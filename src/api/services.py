"""
services.py — Business Logic & Data Validation Service Layer.

This module sits between the FastAPI route handlers (main.py) and the core
ML modules (encoder.py, detector.py, agent.py). It is responsible for:

    1. Orchestrating the full inspection pipeline (encode → detect → agent)
    2. Validating all raw ML outputs through Pydantic schemas before they
       reach the API boundary
    3. Enforcing business rules (latency alerts, fail-safe defaults)
    4. Keeping main.py thin — route handlers delegate to this service

Architecture:
    main.py (HTTP boundary)
        └── InspectionService (business logic + validation)
                ├── VisionEncoder      (ML: feature extraction)
                ├── AnomalyDetector    (ML: anomaly classification)
                ├── run_agent()        (LLM: rejection report)
                └── Pydantic schemas   (data validation)
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Optional

import numpy as np
from pydantic import ValidationError

from src.api.schemas import (
    AgentReport,
    DetectionResult,
    ErrorResponse,
    HealthResponse,
    InspectionMetadata,
    InspectionResponse,
)
from src.config import settings

logger = logging.getLogger("agentic_vision")


# ─── Individual Validation Helpers ───────────────────────────────────────────

def validate_detection_result(raw: dict) -> DetectionResult:
    """
    Parse and validate raw dict output from AnomalyDetector.predict()
    into a typed DetectionResult schema.

    Args:
        raw: Raw dict returned by AnomalyDetector.predict(), expected keys:
             {"status": "Normal"|"Anomaly", "anomaly_score": float}

    Returns:
        DetectionResult: Validated and typed detection result.

    Raises:
        ValueError: If the raw dict is missing required keys or contains
                    invalid values (e.g., NaN score, unknown status string).

    Example:
        raw = {"status": "Anomaly", "anomaly_score": -0.312}
        result = validate_detection_result(raw)
        assert result.status == "Anomaly"
    """
    try:
        return DetectionResult.model_validate(raw)
    except ValidationError as e:
        logger.error(f"[SERVICE] DetectionResult validation failed: {e}")
        raise ValueError(
            f"AnomalyDetector returned invalid output: {e.error_count()} validation error(s). "
            f"Raw output was: {raw}"
        ) from e


def validate_agent_report(raw: dict) -> AgentReport:
    """
    Parse and validate raw dict output from run_agent() (LLM response)
    into a typed AgentReport schema.

    Replaces the manual `required_keys.issubset(report.keys())` check in
    agent.py with a proper Pydantic model_validate() call that enforces
    field types, ranges (severity_score ∈ [0, 1]), and minimum lengths.

    Args:
        raw: Raw dict from run_agent(), expected keys:
             {"defect_confirmed": bool, "severity_score": float,
              "historical_analogy": str, "recommended_action": str}
             Optionally: {"_error": str} on LLM fallback responses.

    Returns:
        AgentReport: Validated and typed agent report.

    Raises:
        ValueError: If the raw dict has missing/invalid fields.

    Example:
        raw = {
            "defect_confirmed": True,
            "severity_score": 0.85,
            "historical_analogy": "Weld porosity Q2 2024",
            "recommended_action": "Halt line and inspect gas shielding."
        }
        report = validate_agent_report(raw)
    """
    try:
        return AgentReport.model_validate(raw)
    except ValidationError as e:
        logger.error(f"[SERVICE] AgentReport validation failed: {e}")
        # On validation failure, return a conservative fail-safe report
        # so the part is always rejected when the LLM output is malformed.
        logger.warning(
            "[SERVICE] Returning conservative fail-safe AgentReport "
            "due to LLM output validation failure."
        )
        return AgentReport(
            defect_confirmed=True,
            severity_score=1.0,
            historical_analogy="Validation error — LLM output could not be parsed.",
            recommended_action=(
                "Halt assembly line and perform manual inspection. "
                "LLM agent output failed schema validation."
            ),
            **{"_error": f"Schema validation failed: {e}"},
        )


def build_inspection_response(
    *,
    response_status: str,
    detection: DetectionResult,
    agent_report: Optional[AgentReport],
    image_filename: str,
    elapsed_seconds: float,
) -> InspectionResponse:
    """
    Compose and validate the full InspectionResponse envelope.

    Combines the detection result, optional agent report, and per-request
    metadata into a single validated Pydantic model that is returned by the
    /inspect-part endpoint.

    Args:
        response_status:  "Pass" or "Fail" — overall inspection outcome.
        detection:        Validated DetectionResult from validate_detection_result().
        agent_report:     Validated AgentReport or None (None for passing parts).
        image_filename:   Filename of the saved part image.
        elapsed_seconds:  Total pipeline latency in seconds.

    Returns:
        InspectionResponse: Fully validated top-level response model.

    Raises:
        ValueError: If the response envelope itself fails validation
                    (e.g., invalid status literal).
    """
    latency_alert = elapsed_seconds > settings.latency_threshold_seconds

    metadata = InspectionMetadata(
        image_file=image_filename,
        processing_time_seconds=elapsed_seconds,
        latency_alert=latency_alert,
    )

    try:
        return InspectionResponse(
            status=response_status,  # type: ignore[arg-type]
            detection=detection,
            agent_analysis=agent_report,
            metadata=metadata,
        )
    except ValidationError as e:
        logger.error(f"[SERVICE] InspectionResponse validation failed: {e}")
        raise ValueError(f"Failed to build InspectionResponse: {e}") from e


def build_health_response(
    *,
    encoder_loaded: bool,
    detector_loaded: bool,
    detector_trained: bool,
) -> HealthResponse:
    """
    Build and validate the HealthResponse for GET /health.

    Args:
        encoder_loaded:   True if VisionEncoder is initialized.
        detector_loaded:  True if AnomalyDetector is initialized.
        detector_trained: True if AnomalyDetector has a trained model.

    Returns:
        HealthResponse: Validated health status model.
    """
    return HealthResponse(
        status="healthy",
        encoder_loaded=encoder_loaded,
        detector_loaded=detector_loaded,
        detector_trained=detector_trained,
    )


# ─── Main Inspection Service ──────────────────────────────────────────────────

class InspectionService:
    """
    Orchestrates the full inspection pipeline with validated inputs/outputs.

    This service is instantiated once in main.py and reused across requests.
    It holds references to the loaded ML models (encoder, detector) and
    runs the complete pipeline for each uploaded image:

        1. Save uploaded image bytes to data/raw/
        2. Extract 512-D feature vector via VisionEncoder
        3. Classify via AnomalyDetector → validated DetectionResult
        4. If Anomaly → run_agent() → validated AgentReport
        5. Compose → validated InspectionResponse

    All raw ML outputs are validated through Pydantic schemas before being
    returned, ensuring the API boundary is always type-safe.
    """

    def __init__(self, encoder, detector) -> None:
        """
        Args:
            encoder:  Initialized VisionEncoder instance.
            detector: Initialized and trained AnomalyDetector instance.
        """
        self.encoder = encoder
        self.detector = detector

    def save_upload(self, file_bytes: bytes, original_filename: Optional[str]) -> tuple[str, str]:
        """
        Save uploaded image bytes to the raw data directory.

        Args:
            file_bytes:        Raw bytes of the uploaded file.
            original_filename: Original filename from the upload (for extension).

        Returns:
            tuple[str, str]: (saved_filename, saved_path) — filename and full path.
        """
        os.makedirs(settings.raw_data_dir, exist_ok=True)

        file_id = str(uuid.uuid4())[:8]
        extension = (
            os.path.splitext(original_filename)[1]
            if original_filename
            else ".jpg"
        )
        saved_filename = f"part_{file_id}{extension}"
        saved_path = os.path.join(settings.raw_data_dir, saved_filename)

        with open(saved_path, "wb") as f:
            f.write(file_bytes)

        logger.info(f"[SERVICE] Image saved: {saved_filename} ({len(file_bytes)} bytes)")
        return saved_filename, saved_path

    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract and validate 512-D feature vector from an image.

        Args:
            image_path: Absolute path to the saved image file.

        Returns:
            np.ndarray: Validated feature vector of shape (512,).

        Raises:
            ValueError: If the extracted feature vector has unexpected shape.
        """
        logger.info("[SERVICE] Extracting features via VisionEncoder (ResNet-18)...")
        features = self.encoder.extract_features(image_path)

        # Validate output shape against config
        expected_dim = settings.feature_vector_dim
        if features.shape != (expected_dim,):
            raise ValueError(
                f"VisionEncoder returned unexpected feature shape: {features.shape}. "
                f"Expected: ({expected_dim},). "
                "Check that the ResNet-18 fc layer is correctly replaced with Identity."
            )

        logger.info(f"[SERVICE] Features extracted: shape={features.shape}, "
                    f"L2-norm={float(np.linalg.norm(features)):.4f}")
        return features

    def run_detection(self, features: np.ndarray) -> DetectionResult:
        """
        Run anomaly detection and validate the result through DetectionResult schema.

        Args:
            features: 512-D feature vector from extract_features().

        Returns:
            DetectionResult: Validated detection result.

        Raises:
            RuntimeError: If the detector is not trained.
            ValueError:   If the detector output fails schema validation.
        """
        if not self.detector._is_trained:
            raise RuntimeError(
                "AnomalyDetector is not trained. "
                "Run the training script to generate model weights before serving."
            )

        logger.info("[SERVICE] Running IsolationForest anomaly detection...")
        raw_result = self.detector.predict(features)

        # Validate raw dict → typed schema
        detection = validate_detection_result(raw_result)
        logger.info(
            f"[SERVICE] Detection complete: status={detection.status}, "
            f"score={detection.anomaly_score:.4f}"
        )
        return detection

    def run_agent_analysis(
        self, image_path: str, anomaly_score: float
    ) -> Optional[AgentReport]:
        """
        Activate the LLM agent and validate the rejection report.

        Only called when detection.status == "Anomaly". Validates the raw
        LLM output through AgentReport schema, with a conservative fail-safe
        if the output is malformed.

        Args:
            image_path:    Path to the anomalous image.
            anomaly_score: Raw IsolationForest decision score.

        Returns:
            AgentReport: Validated agent rejection report, or a fail-safe
                         report if the LLM returned malformed output.
        """
        # Import here to avoid circular import and allow agent.py to use settings independently
        from src.agent import run_agent

        logger.warning(
            "[SERVICE] ANOMALY DETECTED — activating LangChain QA agent..."
        )

        raw_report = run_agent(image_path, anomaly_score)

        # Validate raw LLM dict → typed schema
        agent_report = validate_agent_report(raw_report)

        logger.warning(
            f"[SERVICE] Agent report validated: defect_confirmed={agent_report.defect_confirmed}, "
            f"severity={agent_report.severity_score:.4f}"
        )
        return agent_report

    def run_full_inspection(
        self, file_bytes: bytes, original_filename: Optional[str]
    ) -> InspectionResponse:
        """
        Execute the complete end-to-end inspection pipeline.

        This is the primary entry point called by the /inspect-part route handler.
        All outputs are validated through Pydantic schemas at each stage.

        Pipeline:
            1. Save image → saved_filename, saved_path
            2. Extract features → np.ndarray (512,)
            3. Detect anomaly → DetectionResult (validated)
            4. If Anomaly → LLM agent → AgentReport (validated)
            5. Compose response → InspectionResponse (validated)

        Args:
            file_bytes:        Raw bytes from the uploaded UploadFile.
            original_filename: Original filename for extension detection.

        Returns:
            InspectionResponse: Fully validated inspection response envelope.

        Raises:
            RuntimeError: If the detector is untrained.
            ValueError:   If any pipeline output fails validation.
        """
        start_time = time.time()

        # Step 1: Save uploaded image
        saved_filename, saved_path = self.save_upload(file_bytes, original_filename)

        # Step 2: Feature extraction (validated shape)
        features = self.extract_features(saved_path)

        # Step 3: Anomaly detection (validated DetectionResult)
        detection = self.run_detection(features)

        # Step 4: Conditional agent activation
        agent_report: Optional[AgentReport] = None
        if detection.status == "Anomaly":
            agent_report = self.run_agent_analysis(saved_path, detection.anomaly_score)
            response_status = "Fail"
        else:
            logger.info("[SERVICE] Part is NORMAL — agent bypassed.")
            response_status = "Pass"

        # Step 5: Compose validated response envelope
        elapsed = time.time() - start_time

        if elapsed > settings.latency_threshold_seconds:
            logger.critical(
                f"[SERVICE] ASSEMBLY LINE HALT: latency {elapsed:.3f}s exceeded "
                f"threshold {settings.latency_threshold_seconds}s"
            )

        response = build_inspection_response(
            response_status=response_status,
            detection=detection,
            agent_report=agent_report,
            image_filename=saved_filename,
            elapsed_seconds=elapsed,
        )

        logger.info(
            f"[SERVICE] Inspection complete: status={response.status}, "
            f"latency={elapsed:.3f}s"
        )
        return response
