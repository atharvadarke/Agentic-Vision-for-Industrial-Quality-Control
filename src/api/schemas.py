"""
schemas.py — Pydantic Data Models for Agentic Vision API.

Defines strongly-typed request/response contracts for all API endpoints.
FastAPI uses these models to:
    - Auto-validate incoming and outgoing data at the boundary
    - Auto-generate OpenAPI (Swagger) documentation
    - Serialize responses with guaranteed field presence and types

Model hierarchy:
    DetectionResult       ← output of AnomalyDetector.predict()
    AgentReport           ← output of run_agent() (LLM rejection report)
    InspectionMetadata    ← timing and file-level metadata for each request
    InspectionResponse    ← full /inspect-part response envelope
    HealthResponse        ← /health endpoint response
    ErrorResponse         ← standardized error response body
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ─── Detection Layer ──────────────────────────────────────────────────────────

class DetectionResult(BaseModel):
    """
    Validated output from AnomalyDetector.predict().

    Attributes:
        status:        Binary classification — "Normal" or "Anomaly".
        anomaly_score: Raw IsolationForest decision score.
                       More negative = more anomalous (range: roughly -0.5 to 0.5).
    """
    status: Literal["Normal", "Anomaly"] = Field(
        ...,
        description="Binary classification result from IsolationForest."
    )
    anomaly_score: float = Field(
        ...,
        description="Raw decision function score. More negative = more anomalous."
    )

    @field_validator("anomaly_score")
    @classmethod
    def score_must_be_finite(cls, v: float) -> float:
        """Reject NaN or Inf scores that would indicate a pipeline failure."""
        import math
        if not math.isfinite(v):
            raise ValueError(f"anomaly_score must be a finite float, got: {v}")
        return round(v, 6)


# ─── Agent Layer ──────────────────────────────────────────────────────────────

class AgentReport(BaseModel):
    """
    Validated output from run_agent() — the LLM Quality Assurance report.

    This model replaces the manual set-key validation in agent.py with
    a proper Pydantic schema that validates types, ranges, and presence.

    Attributes:
        defect_confirmed:   Whether the LLM confirmed a real defect.
        severity_score:     Severity between 0.0 (minor) and 1.0 (critical).
        historical_analogy: Description of the most similar past defect from DB.
        recommended_action: Specific, immediately actionable mitigation step.
        _error:             Optional internal error detail (only present on fallback).
    """
    defect_confirmed: bool = Field(
        ...,
        description="True if the LLM confirmed a manufacturing defect."
    )
    severity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Severity score between 0.0 (negligible) and 1.0 (critical halt)."
    )
    historical_analogy: str = Field(
        ...,
        min_length=5,
        description="Reference to the most similar past defect from the historical DB."
    )
    recommended_action: str = Field(
        ...,
        min_length=10,
        description="Specific and immediately actionable mitigation step."
    )
    error: Optional[str] = Field(
        default=None,
        alias="_error",
        description="Present only on fallback when LLM parsing failed.",
    )

    model_config = {"populate_by_name": True}

    @field_validator("severity_score")
    @classmethod
    def round_severity(cls, v: float) -> float:
        """Normalize severity to 4 decimal places for consistent reporting."""
        return round(v, 4)

    @model_validator(mode="after")
    def warn_on_fallback(self) -> "AgentReport":
        """
        Cross-field validation: if _error is present, defect must be confirmed
        (conservative fail-safe — never pass a part when the agent failed).
        """
        if self.error is not None and not self.defect_confirmed:
            raise ValueError(
                "AgentReport with a parsing error must have defect_confirmed=True "
                "(fail-safe: when the agent fails, default to rejecting the part)."
            )
        return self


# ─── Metadata ─────────────────────────────────────────────────────────────────

class InspectionMetadata(BaseModel):
    """
    Per-request metadata attached to every /inspect-part response.

    Attributes:
        image_file:               Filename of the saved part image.
        processing_time_seconds:  Total end-to-end pipeline latency.
        latency_alert:            True if latency exceeded the 500ms assembly line threshold.
    """
    image_file: str = Field(
        ...,
        description="Filename of the saved part image in data/raw/."
    )
    processing_time_seconds: float = Field(
        ...,
        ge=0.0,
        description="Total pipeline latency in seconds (encode + detect + optional agent)."
    )
    latency_alert: bool = Field(
        ...,
        description="True if processing exceeded the 500ms assembly line safety threshold."
    )

    @field_validator("processing_time_seconds")
    @classmethod
    def round_latency(cls, v: float) -> float:
        return round(v, 4)


# ─── Top-Level Response Envelopes ─────────────────────────────────────────────

class InspectionResponse(BaseModel):
    """
    Full validated response envelope for POST /inspect-part.

    Attributes:
        status:         "Pass" if part is Normal, "Fail" if Anomaly detected.
        detection:      Typed DetectionResult from the IsolationForest layer.
        agent_analysis: Typed AgentReport if anomaly triggered the LLM agent,
                        None if the part passed the vision check.
        metadata:       Per-request timing and file metadata.
    """
    status: Literal["Pass", "Fail"] = Field(
        ...,
        description="Overall inspection outcome: Pass = conforming, Fail = defect detected."
    )
    detection: DetectionResult = Field(
        ...,
        description="Anomaly detection result from the IsolationForest classifier."
    )
    agent_analysis: Optional[AgentReport] = Field(
        default=None,
        description="LLM agent rejection report. Only present when status is Fail."
    )
    metadata: InspectionMetadata = Field(
        ...,
        description="Per-request latency and file metadata."
    )

    @model_validator(mode="after")
    def agent_present_on_fail(self) -> "InspectionResponse":
        """
        Business rule: a Fail response SHOULD have agent_analysis populated.
        Logs a warning if the agent failed to produce a report on a Fail result
        (does not raise, so the response is still returned to the client).
        """
        if self.status == "Fail" and self.agent_analysis is None:
            import logging
            logging.getLogger("agentic_vision").warning(
                "[SCHEMA] InspectionResponse status=Fail but agent_analysis is None. "
                "Agent may have been skipped or failed silently."
            )
        return self


class HealthResponse(BaseModel):
    """
    Validated response for GET /health.
    Used by Docker/Kubernetes readiness and liveness probes.

    Attributes:
        status:           Always "healthy" when the API is reachable.
        encoder_loaded:   True if VisionEncoder (ResNet-18) is initialized.
        detector_loaded:  True if AnomalyDetector (IsolationForest) is initialized.
        detector_trained: True if the detector has a loaded/trained model ready for inference.
    """
    status: Literal["healthy"] = Field(
        default="healthy",
        description="Service status string. Always 'healthy' if the endpoint responds."
    )
    encoder_loaded: bool = Field(
        ...,
        description="True if VisionEncoder is loaded and ready."
    )
    detector_loaded: bool = Field(
        ...,
        description="True if AnomalyDetector is initialized."
    )
    detector_trained: bool = Field(
        ...,
        description="True if AnomalyDetector has a trained model ready for inference."
    )


# ─── Error Response ───────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """
    Standardized error response body for all HTTP 4xx/5xx responses.

    Attributes:
        detail:     Human-readable error description.
        error_code: Optional machine-readable error code for client handling.
    """
    detail: str = Field(
        ...,
        description="Human-readable description of the error."
    )
    error_code: Optional[str] = Field(
        default=None,
        description="Optional machine-readable error code (e.g., 'MODEL_NOT_READY')."
    )
