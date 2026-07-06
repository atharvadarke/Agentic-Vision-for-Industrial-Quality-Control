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
"""

import os
import sys
import time
import uuid
import logging
from typing import Optional
from contextlib import asynccontextmanager

# Ensure project root is in sys.path for IDE module resolution and standalone execution
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Summary, make_asgi_app

# ─── Project Imports ─────────────────────────────────────────────────────────
from src.encoder import VisionEncoder
from src.detector import AnomalyDetector
from src.agent import run_agent

# ─── Logging Configuration ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("agentic_vision")

# ─── Prometheus Metrics ──────────────────────────────────────────────────────
INSPECTION_LATENCY = Summary(
    "inspection_latency_seconds",
    "Time spent processing a single /inspect-part request (seconds)"
)

# Latency threshold for assembly line safety (500ms)
LATENCY_THRESHOLD_SECONDS = 0.5

# ─── Global Model References (loaded on startup) ─────────────────────────────
encoder: Optional[VisionEncoder] = None
detector: Optional[AnomalyDetector] = None

# Path configuration
BASE_DIR = _PROJECT_ROOT
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_PATH = os.path.join(BASE_DIR, "models", "anomaly_detector.joblib")


async def startup_event():
    """
    Initialize models on application startup:
    1. Load VisionEncoder (ResNet-18 backbone) — ~44MB RAM
    2. Load AnomalyDetector (IsolationForest) — if trained model exists

    If no trained model exists, the detector is initialized but untrained.
    A training script must be run separately to generate the model weights.
    """
    global encoder, detector

    logger.info("=" * 60)
    logger.info("AGENTIC VISION — STARTING UP")
    logger.info("=" * 60)

    # Step 1: Initialize Vision Encoder
    logger.info("[1/2] Loading VisionEncoder (ResNet-18, CPU-only)...")
    encoder = VisionEncoder()
    logger.info("[1/2] VisionEncoder loaded successfully.")

    # Step 2: Initialize Anomaly Detector
    logger.info("[2/2] Loading AnomalyDetector (IsolationForest)...")
    detector = AnomalyDetector()

    if os.path.exists(MODEL_PATH):
        detector.load_model(MODEL_PATH)
        logger.info(f"[2/2] AnomalyDetector loaded from: {MODEL_PATH}")
    else:
        logger.warning(
            f"[2/2] No trained model found at {MODEL_PATH}. "
            "Detector initialized but NOT trained. "
            "Run the training script to generate model weights."
        )

    # Ensure upload directory exists
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

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


@app.post("/inspect-part")
async def inspect_part(file: UploadFile = File(...)):
    """
    Core inspection endpoint — processes a single part image through the pipeline.

    Pipeline Flow:
        1. Save uploaded image to data/raw/
        2. Extract 512-D feature vector via VisionEncoder
        3. Classify as Normal/Anomaly via AnomalyDetector
        4. If Anomaly → activate LangChain agent for rejection report
        5. If Normal → bypass agent, return pass status

    Returns:
        JSON response with inspection results and optional agent analysis.
    """
    # Start latency timer
    start_time = time.time()

    try:
        # ── Validate model state ─────────────────────────────────────────
        if encoder is None or detector is None:
            raise HTTPException(
                status_code=503,
                detail="Models not loaded. Server is still starting up."
            )

        if not detector._is_trained:
            raise HTTPException(
                status_code=503,
                detail="AnomalyDetector is not trained. Run the training script first."
            )

        # ── Step 0: Save uploaded file ───────────────────────────────────
        file_id = str(uuid.uuid4())[:8]
        file_extension = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
        saved_filename = f"part_{file_id}{file_extension}"
        saved_path = os.path.join(RAW_DATA_DIR, saved_filename)

        contents = await file.read()
        with open(saved_path, "wb") as f:
            f.write(contents)

        logger.info(f"[INSPECT] Image saved: {saved_filename}")

        # ── Step 1: Feature Extraction ───────────────────────────────────
        logger.info("[INSPECT] Step 1 — Extracting features via ResNet-18...")
        features = encoder.extract_features(saved_path)
        logger.info(f"[INSPECT] Feature vector shape: {features.shape}")

        # ── Step 2: Anomaly Detection ────────────────────────────────────
        logger.info("[INSPECT] Step 2 — Running IsolationForest prediction...")
        detection_result = detector.predict(features)
        status = detection_result["status"]
        anomaly_score = detection_result["anomaly_score"]
        logger.info(f"[INSPECT] Detection: {status} (score: {anomaly_score:.4f})")

        # ── Step 3: Conditional Agent Activation ─────────────────────────
        agent_analysis = None

        if status == "Anomaly":
            logger.warning("[INSPECT] [ALERT] ANOMALY DETECTED — Activating LangChain agent...")
            agent_analysis = run_agent(saved_path, anomaly_score)
            logger.warning(f"[INSPECT] Agent report generated: {agent_analysis}")
            response_status = "Fail"
        else:
            logger.info("[INSPECT] [OK] Part is NORMAL — Agent bypassed.")
            response_status = "Pass"

        # ── Latency Check ────────────────────────────────────────────────
        elapsed = time.time() - start_time
        INSPECTION_LATENCY.observe(elapsed)

        if elapsed > LATENCY_THRESHOLD_SECONDS:
            logger.critical(
                "[CRITICAL ALERT] ASSEMBLY LINE HALT: "
                f"LATENCY THRESHOLD EXCEEDED ({elapsed:.3f}s > {LATENCY_THRESHOLD_SECONDS}s)"
            )
            print(
                "[CRITICAL ALERT] ASSEMBLY LINE HALT: LATENCY THRESHOLD EXCEEDED"
            )

        # ── Build Response ───────────────────────────────────────────────
        response = {
            "status": response_status,
            "detection": detection_result,
            "agent_analysis": agent_analysis,
            "metadata": {
                "image_file": saved_filename,
                "processing_time_seconds": round(elapsed, 4),
                "latency_alert": elapsed > LATENCY_THRESHOLD_SECONDS,
            }
        }

        logger.info(f"[INSPECT] Response sent in {elapsed:.3f}s")
        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        INSPECTION_LATENCY.observe(elapsed)
        logger.error(f"[INSPECT] Pipeline error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.get("/health")
async def health_check():
    """Simple health check endpoint for Docker/Kubernetes readiness probes."""
    return {
        "status": "healthy",
        "encoder_loaded": encoder is not None,
        "detector_loaded": detector is not None,
        "detector_trained": detector._is_trained if detector else False,
    }
