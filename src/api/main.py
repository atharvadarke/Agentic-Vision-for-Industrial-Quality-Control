"""
FastAPI Serving Bridge — Agentic Vision for Industrial Quality Control.

This module serves the complete inspection pipeline via a single REST endpoint:
    POST /inspect-part  →  Upload image → Extract features → Detect anomaly → (Agent fallback)

The pipeline is instrumented with Prometheus for latency monitoring:
    GET /metrics  →  Prometheus-compatible metrics endpoint

Architecture:
    - On startup: loads VisionEncoder (ResNet-18) and AnomalyDetector (IsolationForest)
    - Single Uvicorn worker process to prevent RAM duplication on edge devices
    - Agent layer activates ONLY for anomalous frames to conserve compute
    - All business logic and validation delegated to InspectionService (services.py)
    - All config sourced from AppSettings (config.py) — no hardcoded paths/values
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Optional

# Ensure project root is in sys.path for IDE module resolution and standalone execution
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from prometheus_client import Summary, make_asgi_app

# ─── Project Imports ─────────────────────────────────────────────────────────
from src.config import settings
from src.detector import AnomalyDetector
from src.encoder import VisionEncoder
from src.api.schemas import HealthResponse, InspectionResponse
from src.api.services import InspectionService, build_health_response

# ─── Logging Configuration ───────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agentic_vision")

# ─── Prometheus Metrics ──────────────────────────────────────────────────────
INSPECTION_LATENCY = Summary(
    "inspection_latency_seconds",
    "Time spent processing a single /inspect-part request (seconds)",
)

# ─── Global Model & Service References (loaded on startup) ───────────────────
encoder: Optional[VisionEncoder] = None
detector: Optional[AnomalyDetector] = None
inspection_service: Optional[InspectionService] = None


async def startup_event():
    """
    Initialize models and service layer on application startup:
    1. Load VisionEncoder (ResNet-18 backbone) — ~44MB RAM
    2. Load AnomalyDetector (IsolationForest) — if trained model exists
    3. Instantiate InspectionService with both models

    All path and model config is sourced from AppSettings (src/config.py).
    If no trained model exists, the detector is initialized but untrained.
    A training script must be run separately to generate the model weights.
    """
    global encoder, detector, inspection_service

    logger.info("=" * 60)
    logger.info("AGENTIC VISION — STARTING UP")
    logger.info("=" * 60)

    # Step 1: Initialize Vision Encoder
    logger.info("[1/3] Loading VisionEncoder (ResNet-18, CPU-only)...")
    encoder = VisionEncoder()
    logger.info("[1/3] VisionEncoder loaded successfully.")

    # Step 2: Initialize Anomaly Detector
    logger.info("[2/3] Loading AnomalyDetector (IsolationForest)...")
    detector = AnomalyDetector()

    if os.path.exists(settings.model_path):
        detector.load_model(settings.model_path)
        logger.info(f"[2/3] AnomalyDetector loaded from: {settings.model_path}")
    else:
        logger.warning(
            f"[2/3] No trained model found at {settings.model_path}. "
            "Detector initialized but NOT trained. "
            "Run the training script to generate model weights."
        )

    # Step 3: Instantiate InspectionService (business logic + validation layer)
    logger.info("[3/3] Instantiating InspectionService...")
    inspection_service = InspectionService(encoder=encoder, detector=detector)
    logger.info("[3/3] InspectionService ready.")

    # Ensure upload directory exists
    os.makedirs(settings.raw_data_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STARTUP COMPLETE — READY FOR INSPECTION")
    logger.info("=" * 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_event()
    yield


# ─── FastAPI Application ─────────────────────────────────────────────────────
app = FastAPI(
    title="Agentic Vision — Industrial Quality Control",
    description="Edge-deployed vision pipeline for manufacturing defect detection.",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount Prometheus metrics at /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.post(
    "/inspect-part",
    response_model=InspectionResponse,
    summary="Inspect a part image for manufacturing defects",
    response_description="Validated inspection result with detection, optional agent report, and latency metadata.",
)
async def inspect_part(file: UploadFile = File(...)):
    """
    Core inspection endpoint — processes a single part image through the full pipeline.

    Pipeline Flow:
        1. Save uploaded image to data/raw/ (path from AppSettings)
        2. Extract 512-D feature vector via VisionEncoder
        3. Classify as Normal/Anomaly via AnomalyDetector
        4. If Anomaly → activate LangChain agent for validated rejection report
        5. Compose and return a validated InspectionResponse via Pydantic schema

    All outputs are validated through Pydantic schemas in InspectionService
    before reaching this response boundary.
    """
    # ── Guard: models must be loaded ────────────────────────────────────
    if inspection_service is None or encoder is None or detector is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Server is still starting up.",
        )

    if not detector._is_trained:
        raise HTTPException(
            status_code=503,
            detail="AnomalyDetector is not trained. Run the training script first.",
        )

    try:
        # ── Delegate full pipeline to InspectionService ──────────────────
        # InspectionService handles: save → encode → detect → agent → validate
        file_bytes = await file.read()
        response: InspectionResponse = inspection_service.run_full_inspection(
            file_bytes=file_bytes,
            original_filename=file.filename,
        )

        # ── Record Prometheus latency metric ─────────────────────────────
        INSPECTION_LATENCY.observe(response.metadata.processing_time_seconds)

        logger.info(
            f"[INSPECT] {response.status} — "
            f"score={response.detection.anomaly_score:.4f}, "
            f"latency={response.metadata.processing_time_seconds:.3f}s"
        )

        # Return validated Pydantic model (FastAPI serialises via response_model)
        return response

    except HTTPException:
        raise
    except RuntimeError as e:
        # Raised by InspectionService when detector is untrained (safety check)
        logger.error(f"[INSPECT] Runtime error: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"[INSPECT] Unexpected pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service readiness check",
    response_description="Current load status of VisionEncoder and AnomalyDetector.",
)
async def health_check():
    """
    Health check endpoint for Docker/Kubernetes readiness and liveness probes.

    Returns a validated HealthResponse indicating whether the encoder and
    detector are loaded and ready for inference.
    """
    return build_health_response(
        encoder_loaded=encoder is not None,
        detector_loaded=detector is not None,
        detector_trained=detector._is_trained if detector else False,
    )
