"""
tests/test_services.py — InspectionService Unit Tests  (HIGH RISK GAP #1)

Tests src/api/services.py with fully mocked encoder and detector so:
  - No ResNet-18 is loaded (fast, no GPU/model required)
  - No IsolationForest is needed
  - No Groq API key is consumed
  - No file system side effects (tmp directory used)

Covers:
  - save_upload()          : UUID filename, extension preservation, bytes written
  - extract_features()     : shape validation, wrong-shape rejection
  - run_detection()        : Normal path, Anomaly path, untrained-detector RuntimeError
  - run_agent_analysis()   : LLM fail-safe path (malformed JSON → conservative report)
  - run_full_inspection()  : Normal E2E (no agent), Anomaly E2E (agent activated),
                             latency_alert=True when elapsed > threshold
  - validate_detection_result() : NaN score → ValueError
  - validate_agent_report()     : bad LLM output → fail-safe AgentReport
"""

import os
import math
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.api.services import (
    InspectionService,
    build_health_response,
    build_inspection_response,
    validate_agent_report,
    validate_detection_result,
)
from src.api.schemas import DetectionResult, AgentReport


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NORMAL_FEATURE = np.random.default_rng(42).standard_normal(512).astype(np.float32)
NORMAL_FEATURE /= np.linalg.norm(NORMAL_FEATURE)  # L2-normalised

VALID_AGENT_DICT = {
    "defect_confirmed": True,
    "severity_score": 0.82,
    "historical_analogy": "[Q3 2024] surface_scratch: Conveyor belt friction.",
    "recommended_action": "Halt conveyor belt, inspect belt tension immediately.",
}


def _mock_encoder(feature_vec=None):
    """Return a mock VisionEncoder that returns a fixed feature vector."""
    enc = MagicMock()
    enc.extract_features.return_value = feature_vec if feature_vec is not None else NORMAL_FEATURE
    return enc


def _mock_detector(status="Normal", score=0.09, trained=True):
    """Return a mock AnomalyDetector that returns a fixed prediction."""
    det = MagicMock()
    det._is_trained = trained
    det.predict.return_value = {"status": status, "anomaly_score": score}
    return det


def _make_service(status="Normal", score=0.09, feature_vec=None):
    """Build InspectionService with mock encoder and detector."""
    return InspectionService(
        encoder=_mock_encoder(feature_vec),
        detector=_mock_detector(status, score),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# validate_detection_result()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestValidateDetectionResult:

    def test_valid_normal(self):
        r = validate_detection_result({"status": "Normal", "anomaly_score": 0.09})
        assert isinstance(r, DetectionResult)
        assert r.status == "Normal"

    def test_valid_anomaly(self):
        r = validate_detection_result({"status": "Anomaly", "anomaly_score": -0.32})
        assert r.status == "Anomaly"

    def test_rejects_nan_score(self):
        """NaN from a broken model must raise ValueError, never silently propagate."""
        with pytest.raises(ValueError, match="validation error"):
            validate_detection_result({"status": "Normal", "anomaly_score": float("nan")})

    def test_rejects_inf_score(self):
        with pytest.raises(ValueError, match="validation error"):
            validate_detection_result({"status": "Normal", "anomaly_score": math.inf})

    def test_rejects_bad_status(self):
        with pytest.raises(ValueError, match="validation error"):
            validate_detection_result({"status": "Unknown", "anomaly_score": 0.1})

    def test_rejects_empty_dict(self):
        with pytest.raises(ValueError, match="validation error"):
            validate_detection_result({})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# validate_agent_report()  —  CRITICAL FAIL-SAFE PATH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestValidateAgentReport:

    def test_valid_report(self):
        r = validate_agent_report(VALID_AGENT_DICT)
        assert isinstance(r, AgentReport)
        assert r.defect_confirmed is True
        assert r.severity_score == pytest.approx(0.82, abs=1e-4)

    def test_malformed_llm_output_triggers_failsafe(self):
        """
        HIGH RISK: When run_agent() returns garbage (missing fields / wrong types),
        validate_agent_report() must return the conservative fail-safe AgentReport —
        never raise and never return an invalid object.
        """
        malformed = {"some_unknown_key": "garbage"}
        result = validate_agent_report(malformed)
        # Must still be an AgentReport (not raise)
        assert isinstance(result, AgentReport)
        # Fail-safe: defect must always be confirmed when parsing fails
        assert result.defect_confirmed is True
        assert result.severity_score == 1.0
        # Error detail must be present
        assert result.error is not None

    def test_severity_out_of_range_triggers_failsafe(self):
        """LLM returning severity=5.0 must hit the fail-safe, not propagate."""
        bad_report = {**VALID_AGENT_DICT, "severity_score": 5.0}
        result = validate_agent_report(bad_report)
        assert isinstance(result, AgentReport)
        assert result.defect_confirmed is True
        assert result.severity_score == 1.0

    def test_empty_dict_triggers_failsafe(self):
        result = validate_agent_report({})
        assert isinstance(result, AgentReport)
        assert result.defect_confirmed is True

    def test_valid_report_with_error_field(self):
        """A valid report that includes _error (partial parse success path)."""
        data = {**VALID_AGENT_DICT, "_error": "LLM timed out, using defaults."}
        r = validate_agent_report(data)
        assert r.defect_confirmed is True
        assert r.error == "LLM timed out, using defaults."


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# InspectionService.save_upload()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSaveUpload:

    def test_saves_bytes_to_disk(self, tmp_path):
        svc = _make_service()
        image_bytes = b"\xff\xd8\xff" + b"\x00" * 100  # fake JPEG header

        with patch("src.api.services.settings") as mock_cfg:
            mock_cfg.raw_data_dir = str(tmp_path)
            fname, fpath = svc.save_upload(image_bytes, "part_cam01.jpg")

        assert os.path.exists(fpath)
        assert os.path.getsize(fpath) == len(image_bytes)

    def test_filename_has_part_prefix(self, tmp_path):
        svc = _make_service()
        with patch("src.api.services.settings") as mock_cfg:
            mock_cfg.raw_data_dir = str(tmp_path)
            fname, _ = svc.save_upload(b"data", "image.jpg")
        assert fname.startswith("part_")

    def test_preserves_file_extension(self, tmp_path):
        svc = _make_service()
        with patch("src.api.services.settings") as mock_cfg:
            mock_cfg.raw_data_dir = str(tmp_path)
            fname, _ = svc.save_upload(b"data", "photo.png")
        assert fname.endswith(".png")

    def test_defaults_to_jpg_when_no_filename(self, tmp_path):
        svc = _make_service()
        with patch("src.api.services.settings") as mock_cfg:
            mock_cfg.raw_data_dir = str(tmp_path)
            fname, _ = svc.save_upload(b"data", None)
        assert fname.endswith(".jpg")

    def test_two_uploads_get_different_filenames(self, tmp_path):
        svc = _make_service()
        with patch("src.api.services.settings") as mock_cfg:
            mock_cfg.raw_data_dir = str(tmp_path)
            f1, _ = svc.save_upload(b"data1", "a.jpg")
            f2, _ = svc.save_upload(b"data2", "b.jpg")
        assert f1 != f2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# InspectionService.extract_features()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestExtractFeatures:

    def test_returns_512d_vector(self):
        svc = _make_service()
        with patch("src.api.services.settings") as mock_cfg:
            mock_cfg.feature_vector_dim = 512
            feat = svc.extract_features("/fake/path/part.jpg")
        assert feat.shape == (512,)

    def test_rejects_wrong_shape_from_encoder(self):
        """
        HIGH RISK: If the encoder is misconfigured (fc not stripped correctly),
        it might return shape (1000,) instead of (512,). This must raise, not silently continue.
        """
        wrong_feat = np.zeros(1000, dtype=np.float32)  # 1000-D — wrong
        svc = _make_service(feature_vec=wrong_feat)
        with patch("src.api.services.settings") as mock_cfg:
            mock_cfg.feature_vector_dim = 512
            with pytest.raises(ValueError, match="unexpected feature shape"):
                svc.extract_features("/fake/path/part.jpg")

    def test_encoder_called_with_correct_path(self):
        svc = _make_service()
        path = "/some/path/part_test.jpg"
        with patch("src.api.services.settings") as mock_cfg:
            mock_cfg.feature_vector_dim = 512
            svc.extract_features(path)
        svc.encoder.extract_features.assert_called_once_with(path)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# InspectionService.run_detection()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRunDetection:

    def test_normal_prediction_returns_detection_result(self):
        svc = _make_service(status="Normal", score=0.09)
        result = svc.run_detection(NORMAL_FEATURE)
        assert isinstance(result, DetectionResult)
        assert result.status == "Normal"

    def test_anomaly_prediction_returns_detection_result(self):
        svc = _make_service(status="Anomaly", score=-0.35)
        result = svc.run_detection(NORMAL_FEATURE)
        assert result.status == "Anomaly"
        assert result.anomaly_score < 0

    def test_untrained_detector_raises_runtime_error(self):
        """
        HIGH RISK: Calling predict() on an untrained detector must raise RuntimeError
        immediately — not silently return garbage predictions.
        """
        svc = InspectionService(
            encoder=_mock_encoder(),
            detector=_mock_detector(trained=False),
        )
        with pytest.raises(RuntimeError, match="not trained"):
            svc.run_detection(NORMAL_FEATURE)

    def test_detector_called_with_feature_vector(self):
        svc = _make_service(status="Normal", score=0.07)
        svc.run_detection(NORMAL_FEATURE)
        svc.detector.predict.assert_called_once_with(NORMAL_FEATURE)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# InspectionService.run_agent_analysis()  —  CRITICAL FAIL-SAFE PATH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRunAgentAnalysis:

    def test_valid_llm_output_returns_agent_report(self):
        svc = _make_service()
        # run_agent is a local import inside run_agent_analysis() — patch at source
        with patch("src.agent.run_agent", return_value=VALID_AGENT_DICT):
            report = svc.run_agent_analysis("/path/to/part.jpg", -0.35)
        assert isinstance(report, AgentReport)
        assert report.defect_confirmed is True
        assert 0 <= report.severity_score <= 1

    def test_malformed_llm_output_returns_failsafe_report(self):
        """
        HIGH RISK: When run_agent() returns garbage (missing fields / wrong types),
        run_agent_analysis() must still return a valid AgentReport via the fail-safe —
        never raise and never return an invalid object.
        """
        garbage = {"broken": "response", "_error": "json decode error"}
        svc = _make_service()
        with patch("src.agent.run_agent", return_value=garbage):
            report = svc.run_agent_analysis("/path/to/part.jpg", -0.35)
        assert isinstance(report, AgentReport)
        assert report.defect_confirmed is True   # fail-safe must ALWAYS confirm defect
        assert report.severity_score == 1.0      # fail-safe must always use max severity

    def test_llm_api_error_dict_returns_failsafe(self):
        """Simulates the run_agent() Exception path — API/network error."""
        api_error_dict = {
            "defect_confirmed": True,
            "severity_score": 0.35,   # abs(anomaly_score) capped at 1.0
            "historical_analogy": "Unable to query LLM — using conservative defaults.",
            "recommended_action": "Halt assembly line and escalate to shift supervisor for manual review.",
            "_error": "LLM API error: Connection timeout",
        }
        svc = _make_service()
        with patch("src.agent.run_agent", return_value=api_error_dict):
            report = svc.run_agent_analysis("/path/to/part.jpg", -0.35)
        assert isinstance(report, AgentReport)
        assert report.defect_confirmed is True
        assert report.error is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# InspectionService.run_full_inspection()  —  Full E2E Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRunFullInspection:

    def test_normal_part_returns_pass_no_agent(self, tmp_path):
        """Normal path: detector says Normal → Pass, agent bypassed, agent_analysis=None."""
        svc = _make_service(status="Normal", score=0.09)
        image_bytes = b"\xff\xd8\xff" + b"\x00" * 50

        with patch("src.api.services.settings") as mock_cfg:
            mock_cfg.raw_data_dir = str(tmp_path)
            mock_cfg.feature_vector_dim = 512
            mock_cfg.latency_threshold_seconds = 0.5
            response = svc.run_full_inspection(image_bytes, "part_cam.jpg")

        assert response.status == "Pass"
        assert response.detection.status == "Normal"
        assert response.agent_analysis is None
        assert response.metadata.latency_alert is False

    def test_anomaly_part_returns_fail_with_agent(self, tmp_path):
        """Anomaly path: detector says Anomaly → Fail, agent activated, agent_analysis populated."""
        svc = InspectionService(
            encoder=_mock_encoder(),
            detector=_mock_detector(status="Anomaly", score=-0.35),
        )
        image_bytes = b"\xff\xd8\xff" + b"\x00" * 50

        with patch("src.api.services.settings") as mock_cfg, \
             patch("src.agent.run_agent", return_value=VALID_AGENT_DICT):
            mock_cfg.raw_data_dir = str(tmp_path)
            mock_cfg.feature_vector_dim = 512
            mock_cfg.latency_threshold_seconds = 0.5
            response = svc.run_full_inspection(image_bytes, "defective_part.jpg")

        assert response.status == "Fail"
        assert response.detection.status == "Anomaly"
        assert response.agent_analysis is not None
        assert response.agent_analysis.defect_confirmed is True

    def test_latency_alert_set_when_threshold_exceeded(self, tmp_path):
        """
        When pipeline elapsed time > latency_threshold_seconds,
        metadata.latency_alert must be True.
        """
        svc = _make_service(status="Normal", score=0.09)
        image_bytes = b"\xff\xd8\xff" + b"\x00" * 50

        with patch("src.api.services.settings") as mock_cfg, \
             patch("src.api.services.time") as mock_time:
            # Simulate 0.6s elapsed > 0.5s threshold
            mock_time.time.side_effect = [0.0, 0.6]
            mock_cfg.raw_data_dir = str(tmp_path)
            mock_cfg.feature_vector_dim = 512
            mock_cfg.latency_threshold_seconds = 0.5
            response = svc.run_full_inspection(image_bytes, "part.jpg")

        assert response.metadata.latency_alert is True

    def test_image_file_saved_to_disk(self, tmp_path):
        """After run_full_inspection, the image bytes must be persisted."""
        svc = _make_service(status="Normal", score=0.09)
        image_bytes = b"FAKE_IMAGE_BYTES_1234"

        with patch("src.api.services.settings") as mock_cfg:
            mock_cfg.raw_data_dir = str(tmp_path)
            mock_cfg.feature_vector_dim = 512
            mock_cfg.latency_threshold_seconds = 0.5
            response = svc.run_full_inspection(image_bytes, "cam_part.jpg")

        saved_path = tmp_path / response.metadata.image_file
        assert saved_path.exists()
        assert saved_path.read_bytes() == image_bytes


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# build_health_response()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBuildHealthResponse:

    def test_fully_ready(self):
        h = build_health_response(
            encoder_loaded=True,
            detector_loaded=True,
            detector_trained=True,
        )
        assert h.status == "healthy"
        assert h.encoder_loaded is True
        assert h.detector_trained is True

    def test_detector_not_trained(self):
        h = build_health_response(
            encoder_loaded=True,
            detector_loaded=True,
            detector_trained=False,
        )
        assert h.status == "healthy"
        assert h.detector_trained is False
