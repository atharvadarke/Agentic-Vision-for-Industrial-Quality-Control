"""
tests/test_schemas.py — Pydantic Schema Validation Tests  (HIGH RISK GAP #2)

Covers every validator and model_validator in src/api/schemas.py, including:
  - DetectionResult  : NaN/Inf score rejection, invalid Literal status
  - AgentReport      : severity out-of-range, missing required fields,
                       fail-safe cross-field rule (_error requires defect_confirmed=True)
  - InspectionMetadata : negative latency rejection
  - InspectionResponse : Fail-without-agent warning path, Pass-with-agent guard
  - HealthResponse   : valid construction only

These tests protect the critical API boundary — malformed ML/LLM output must
never reach the client in an unvalidated state.
"""

import math
import pytest
from pydantic import ValidationError

from src.api.schemas import (
    AgentReport,
    DetectionResult,
    ErrorResponse,
    HealthResponse,
    InspectionMetadata,
    InspectionResponse,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DetectionResult
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDetectionResult:

    def test_valid_normal(self):
        r = DetectionResult(status="Normal", anomaly_score=0.0821)
        assert r.status == "Normal"
        assert r.anomaly_score == pytest.approx(0.0821, abs=1e-5)

    def test_valid_anomaly(self):
        r = DetectionResult(status="Anomaly", anomaly_score=-0.3157)
        assert r.status == "Anomaly"
        assert r.anomaly_score < 0

    def test_score_rounded_to_6_decimals(self):
        r = DetectionResult(status="Normal", anomaly_score=0.123456789)
        assert r.anomaly_score == pytest.approx(0.123457, abs=1e-6)

    def test_rejects_nan_score(self):
        with pytest.raises(ValidationError, match="finite"):
            DetectionResult(status="Normal", anomaly_score=float("nan"))

    def test_rejects_positive_inf_score(self):
        with pytest.raises(ValidationError, match="finite"):
            DetectionResult(status="Normal", anomaly_score=math.inf)

    def test_rejects_negative_inf_score(self):
        with pytest.raises(ValidationError, match="finite"):
            DetectionResult(status="Anomaly", anomaly_score=-math.inf)

    def test_rejects_invalid_status_literal(self):
        with pytest.raises(ValidationError):
            DetectionResult(status="Pass", anomaly_score=0.05)  # "Pass" is not a valid status

    def test_rejects_missing_status(self):
        with pytest.raises(ValidationError):
            DetectionResult(anomaly_score=0.05)  # type: ignore[call-arg]

    def test_rejects_missing_score(self):
        with pytest.raises(ValidationError):
            DetectionResult(status="Normal")  # type: ignore[call-arg]

    def test_model_validate_from_raw_dict(self):
        """Simulates exact output format from AnomalyDetector.predict()."""
        raw = {"status": "Anomaly", "anomaly_score": -0.25}
        r = DetectionResult.model_validate(raw)
        assert r.status == "Anomaly"
        assert r.anomaly_score == pytest.approx(-0.25, abs=1e-5)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AgentReport
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VALID_AGENT_REPORT = {
    "defect_confirmed": True,
    "severity_score": 0.87,
    "historical_analogy": "[Q3 2024] surface_scratch (severity: 0.82): Conveyor belt friction.",
    "recommended_action": "Halt conveyor belt and inspect belt tension immediately.",
}


class TestAgentReport:

    def test_valid_report(self):
        r = AgentReport.model_validate(VALID_AGENT_REPORT)
        assert r.defect_confirmed is True
        assert r.severity_score == pytest.approx(0.87, abs=1e-4)
        assert r.error is None

    def test_severity_rounded_to_4_decimals(self):
        data = {**VALID_AGENT_REPORT, "severity_score": 0.876543210}
        r = AgentReport.model_validate(data)
        assert r.severity_score == pytest.approx(0.8765, abs=1e-5)

    def test_rejects_severity_above_1(self):
        with pytest.raises(ValidationError):
            AgentReport.model_validate({**VALID_AGENT_REPORT, "severity_score": 1.5})

    def test_rejects_severity_below_0(self):
        with pytest.raises(ValidationError):
            AgentReport.model_validate({**VALID_AGENT_REPORT, "severity_score": -0.1})

    def test_severity_at_boundary_0(self):
        r = AgentReport.model_validate({**VALID_AGENT_REPORT, "severity_score": 0.0})
        assert r.severity_score == 0.0

    def test_severity_at_boundary_1(self):
        r = AgentReport.model_validate({**VALID_AGENT_REPORT, "severity_score": 1.0})
        assert r.severity_score == 1.0

    def test_rejects_historical_analogy_too_short(self):
        """min_length=5 on historical_analogy."""
        with pytest.raises(ValidationError):
            AgentReport.model_validate({**VALID_AGENT_REPORT, "historical_analogy": "N/A"})

    def test_rejects_recommended_action_too_short(self):
        """min_length=10 on recommended_action."""
        with pytest.raises(ValidationError):
            AgentReport.model_validate({**VALID_AGENT_REPORT, "recommended_action": "Stop."})

    def test_rejects_missing_defect_confirmed(self):
        data = {k: v for k, v in VALID_AGENT_REPORT.items() if k != "defect_confirmed"}
        with pytest.raises(ValidationError):
            AgentReport.model_validate(data)

    def test_rejects_missing_severity_score(self):
        data = {k: v for k, v in VALID_AGENT_REPORT.items() if k != "severity_score"}
        with pytest.raises(ValidationError):
            AgentReport.model_validate(data)

    # ── Fail-safe cross-field rule ───────────────────────────────────────────

    def test_error_field_requires_defect_confirmed_true(self):
        """
        HIGH RISK: When _error is present (LLM parse failure), defect_confirmed
        MUST be True. This is the conservative fail-safe — never pass a part
        when the agent failed. If this breaks, defective parts could slip through.
        """
        with pytest.raises(ValidationError, match="fail-safe"):
            AgentReport.model_validate({
                **VALID_AGENT_REPORT,
                "defect_confirmed": False,   # WRONG — should trigger fail-safe error
                "_error": "json.JSONDecodeError: line 1",
            })

    def test_error_field_with_defect_confirmed_true_is_valid(self):
        """The correct fail-safe path: _error + defect_confirmed=True is allowed."""
        r = AgentReport.model_validate({
            **VALID_AGENT_REPORT,
            "defect_confirmed": True,
            "_error": "LLM output was malformed JSON.",
        })
        assert r.defect_confirmed is True
        assert r.error == "LLM output was malformed JSON."

    def test_no_error_field_is_none_by_default(self):
        r = AgentReport.model_validate(VALID_AGENT_REPORT)
        assert r.error is None

    def test_alias_error_field_populated_by_alias(self):
        """_error alias must be accepted by model_validate."""
        r = AgentReport.model_validate({
            **VALID_AGENT_REPORT,
            "_error": "API timeout",
        })
        assert r.error == "API timeout"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# InspectionMetadata
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestInspectionMetadata:

    def test_valid_metadata(self):
        m = InspectionMetadata(
            image_file="part_abc12345.jpg",
            processing_time_seconds=0.2134,
            latency_alert=False,
        )
        assert m.image_file == "part_abc12345.jpg"
        assert m.latency_alert is False

    def test_latency_rounded_to_4_decimals(self):
        m = InspectionMetadata(
            image_file="part.jpg",
            processing_time_seconds=0.213456789,
            latency_alert=False,
        )
        assert m.processing_time_seconds == pytest.approx(0.2135, abs=1e-5)

    def test_rejects_negative_latency(self):
        with pytest.raises(ValidationError):
            InspectionMetadata(
                image_file="part.jpg",
                processing_time_seconds=-0.1,
                latency_alert=False,
            )

    def test_zero_latency_is_valid(self):
        m = InspectionMetadata(
            image_file="part.jpg",
            processing_time_seconds=0.0,
            latency_alert=False,
        )
        assert m.processing_time_seconds == 0.0

    def test_latency_alert_true(self):
        m = InspectionMetadata(
            image_file="part.jpg",
            processing_time_seconds=0.85,
            latency_alert=True,
        )
        assert m.latency_alert is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# InspectionResponse
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_detection(status="Normal", score=0.08):
    return DetectionResult(status=status, anomaly_score=score)

def _make_metadata(alert=False, latency=0.21):
    return InspectionMetadata(
        image_file="part_test.jpg",
        processing_time_seconds=latency,
        latency_alert=alert,
    )

def _make_agent_report():
    return AgentReport.model_validate(VALID_AGENT_REPORT)


class TestInspectionResponse:

    def test_pass_response_no_agent(self):
        r = InspectionResponse(
            status="Pass",
            detection=_make_detection("Normal", 0.08),
            agent_analysis=None,
            metadata=_make_metadata(),
        )
        assert r.status == "Pass"
        assert r.agent_analysis is None

    def test_fail_response_with_agent(self):
        r = InspectionResponse(
            status="Fail",
            detection=_make_detection("Anomaly", -0.35),
            agent_analysis=_make_agent_report(),
            metadata=_make_metadata(alert=True, latency=1.5),
        )
        assert r.status == "Fail"
        assert r.agent_analysis is not None
        assert r.agent_analysis.defect_confirmed is True

    def test_fail_without_agent_does_not_raise(self):
        """
        Business rule: Fail without agent_analysis logs a warning but does NOT
        raise — the response is still returned so the client sees the Fail status.
        """
        r = InspectionResponse(
            status="Fail",
            detection=_make_detection("Anomaly", -0.15),
            agent_analysis=None,   # agent skipped / failed
            metadata=_make_metadata(),
        )
        assert r.status == "Fail"
        assert r.agent_analysis is None

    def test_rejects_invalid_status_literal(self):
        with pytest.raises(ValidationError):
            InspectionResponse(
                status="Normal",   # not a valid top-level status — must be Pass/Fail
                detection=_make_detection("Normal", 0.05),
                agent_analysis=None,
                metadata=_make_metadata(),
            )

    def test_rejects_missing_detection(self):
        with pytest.raises((ValidationError, TypeError)):
            InspectionResponse(
                status="Pass",
                agent_analysis=None,
                metadata=_make_metadata(),
            )  # type: ignore[call-arg]

    def test_rejects_missing_metadata(self):
        with pytest.raises((ValidationError, TypeError)):
            InspectionResponse(
                status="Pass",
                detection=_make_detection(),
                agent_analysis=None,
            )  # type: ignore[call-arg]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HealthResponse
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestHealthResponse:

    def test_fully_ready(self):
        h = HealthResponse(
            encoder_loaded=True,
            detector_loaded=True,
            detector_trained=True,
        )
        assert h.status == "healthy"
        assert h.detector_trained is True

    def test_partially_ready(self):
        h = HealthResponse(
            encoder_loaded=True,
            detector_loaded=True,
            detector_trained=False,
        )
        assert h.status == "healthy"
        assert h.detector_trained is False

    def test_status_always_healthy(self):
        """The Literal["healthy"] field cannot be set to anything else."""
        with pytest.raises(ValidationError):
            HealthResponse(
                status="unhealthy",   # type: ignore[arg-type]
                encoder_loaded=True,
                detector_loaded=True,
                detector_trained=True,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ErrorResponse
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestErrorResponse:

    def test_valid_with_error_code(self):
        e = ErrorResponse(detail="Model not ready.", error_code="MODEL_NOT_READY")
        assert e.detail == "Model not ready."
        assert e.error_code == "MODEL_NOT_READY"

    def test_valid_without_error_code(self):
        e = ErrorResponse(detail="Unexpected error.")
        assert e.error_code is None

    def test_rejects_missing_detail(self):
        with pytest.raises(ValidationError):
            ErrorResponse()  # type: ignore[call-arg]
