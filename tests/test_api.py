"""
tests/test_api.py — FastAPI Endpoint Tests  (MEDIUM RISK GAP #2)

Tests the HTTP layer in src/api/main.py WITHOUT loading the real ResNet-18 model.
Achieved by:
  1. Patching startup_event() to a no-op so lifespan doesn't load any real models.
  2. Directly setting the module-level globals (encoder, detector, inspection_service)
     to mocks via monkeypatch.setattr before each test.

Covers:
  GET  /health
    - All components loaded + detector trained → 200, detector_trained=True
    - Detector loaded but untrained            → 200, detector_trained=False
    - Nothing loaded (startup not yet done)    → 200, all flags=False

  POST /inspect-part
    - Models not loaded (None globals)         → 503
    - Detector loaded but untrained            → 503
    - Normal part (mock returns Normal)        → 200, status="Pass", agent_analysis=None
    - Anomaly part (mock returns Fail)         → 200, status="Fail", agent_analysis populated
    - RuntimeError from pipeline               → 503
    - Unexpected exception from pipeline       → 500

  GET  /metrics
    - Prometheus endpoint exposed              → 200, contains metric name

All tests run synchronously via asyncio.run() — no extra pytest plugins needed.
"""

import asyncio
import io
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
import pytest

import src.api.main as main_module
from src.api.main import app
from src.api.schemas import (
    AgentReport,
    DetectionResult,
    InspectionMetadata,
    InspectionResponse,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures & helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _noop_startup():
    """Async no-op startup replaces the real startup_event to avoid ResNet-18 load."""
    pass


def _fake_image_bytes() -> bytes:
    """Minimal valid JPEG header + padding so the upload is non-empty."""
    return b"\xff\xd8\xff\xe0" + b"\x00" * 64


def _make_pass_response() -> InspectionResponse:
    return InspectionResponse(
        status="Pass",
        detection=DetectionResult(status="Normal", anomaly_score=0.0875),
        agent_analysis=None,
        metadata=InspectionMetadata(
            image_file="part_abc12345.jpg",
            processing_time_seconds=0.0487,
            latency_alert=False,
        ),
    )


def _make_fail_response() -> InspectionResponse:
    return InspectionResponse(
        status="Fail",
        detection=DetectionResult(status="Anomaly", anomaly_score=-0.3512),
        agent_analysis=AgentReport(
            defect_confirmed=True,
            severity_score=0.87,
            historical_analogy="[Q3 2024] surface_scratch severity 0.82.",
            recommended_action="Halt conveyor belt, inspect belt tension immediately.",
        ),
        metadata=InspectionMetadata(
            image_file="part_def67890.jpg",
            processing_time_seconds=1.4823,
            latency_alert=True,
        ),
    )


@pytest.fixture(autouse=True)
def reset_globals(monkeypatch):
    """
    Before each test, reset ALL module-level globals in main.py to None
    AND patch startup_event to a no-op so the lifespan never loads real models.
    After the test, monkeypatch automatically restores everything.
    """
    monkeypatch.setattr(main_module, "encoder", None)
    monkeypatch.setattr(main_module, "detector", None)
    monkeypatch.setattr(main_module, "inspection_service", None)
    monkeypatch.setattr(main_module, "startup_event", _noop_startup)


def _client():
    """Return a synchronous httpx client wrapping the FastAPI ASGI app."""
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


def run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET /health
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestHealthEndpoint:

    def test_all_loaded_and_trained_returns_all_true(self, monkeypatch):
        """Fully ready system: encoder loaded, detector trained."""
        mock_det = MagicMock()
        mock_det._is_trained = True
        monkeypatch.setattr(main_module, "encoder", MagicMock())
        monkeypatch.setattr(main_module, "detector", mock_det)

        async def _test():
            async with _client() as client:
                resp = await client.get("/health")
                return resp

        resp = run(_test())
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["encoder_loaded"] is True
        assert body["detector_loaded"] is True
        assert body["detector_trained"] is True

    def test_detector_loaded_but_untrained(self, monkeypatch):
        """
        MEDIUM RISK: Detector loaded but not yet trained.
        /health must still return 200 (it's a status, not a guard),
        but detector_trained must be False so orchestrators know not to route traffic.
        """
        mock_det = MagicMock()
        mock_det._is_trained = False
        monkeypatch.setattr(main_module, "encoder", MagicMock())
        monkeypatch.setattr(main_module, "detector", mock_det)

        async def _test():
            async with _client() as client:
                resp = await client.get("/health")
                return resp

        resp = run(_test())
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["detector_loaded"] is True
        assert body["detector_trained"] is False  # critical flag

    def test_nothing_loaded_returns_all_false(self):
        """Nothing loaded yet (e.g., startup still in progress): all flags = False."""
        # reset_globals fixture already set all globals to None

        async def _test():
            async with _client() as client:
                resp = await client.get("/health")
                return resp

        resp = run(_test())
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"      # endpoint itself is reachable
        assert body["encoder_loaded"] is False
        assert body["detector_loaded"] is False
        assert body["detector_trained"] is False

    def test_encoder_loaded_detector_not(self, monkeypatch):
        """Partial startup: encoder ready but detector still loading."""
        monkeypatch.setattr(main_module, "encoder", MagicMock())
        # detector stays None (from fixture)

        async def _test():
            async with _client() as client:
                resp = await client.get("/health")
                return resp

        resp = run(_test())
        assert resp.status_code == 200
        body = resp.json()
        assert body["encoder_loaded"] is True
        assert body["detector_loaded"] is False
        assert body["detector_trained"] is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# POST /inspect-part — Guard: models not loaded → 503
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestInspectPartGuards:

    def test_returns_503_when_service_is_none(self):
        """
        MEDIUM RISK: POST /inspect-part when models aren't loaded yet must return 503.
        Without this guard, FastAPI would call None.run_full_inspection() and crash with
        AttributeError, returning 500 instead of the correct 503 "not ready" signal.
        """
        # All globals stay None (from fixture)
        async def _test():
            async with _client() as client:
                resp = await client.post(
                    "/inspect-part",
                    files={"file": ("part.jpg", _fake_image_bytes(), "image/jpeg")},
                )
                return resp

        resp = run(_test())
        assert resp.status_code == 503
        assert "Models not loaded" in resp.json()["detail"]

    def test_returns_503_when_encoder_is_none_only(self, monkeypatch):
        """All three globals must be set — missing encoder alone triggers 503."""
        mock_det = MagicMock()
        mock_det._is_trained = True
        monkeypatch.setattr(main_module, "detector", mock_det)
        monkeypatch.setattr(main_module, "inspection_service", MagicMock())
        # encoder stays None

        async def _test():
            async with _client() as client:
                resp = await client.post(
                    "/inspect-part",
                    files={"file": ("part.jpg", _fake_image_bytes(), "image/jpeg")},
                )
                return resp

        resp = run(_test())
        assert resp.status_code == 503

    def test_returns_503_when_detector_untrained(self, monkeypatch):
        """
        MEDIUM RISK: The untrained-detector guard in /inspect-part.
        If the model file is missing and training hasn't run, detector._is_trained=False.
        This must return 503, not 200 with random predictions.
        """
        mock_det = MagicMock()
        mock_det._is_trained = False  # NOT trained
        monkeypatch.setattr(main_module, "encoder", MagicMock())
        monkeypatch.setattr(main_module, "detector", mock_det)
        monkeypatch.setattr(main_module, "inspection_service", MagicMock())

        async def _test():
            async with _client() as client:
                resp = await client.post(
                    "/inspect-part",
                    files={"file": ("part.jpg", _fake_image_bytes(), "image/jpeg")},
                )
                return resp

        resp = run(_test())
        assert resp.status_code == 503
        assert "not trained" in resp.json()["detail"]

    def test_503_detail_mentions_training_script(self, monkeypatch):
        """Error message must be actionable — tell operator what to run."""
        mock_det = MagicMock()
        mock_det._is_trained = False
        monkeypatch.setattr(main_module, "encoder", MagicMock())
        monkeypatch.setattr(main_module, "detector", mock_det)
        monkeypatch.setattr(main_module, "inspection_service", MagicMock())

        async def _test():
            async with _client() as client:
                resp = await client.post(
                    "/inspect-part",
                    files={"file": ("part.jpg", _fake_image_bytes(), "image/jpeg")},
                )
                return resp

        resp = run(_test())
        detail = resp.json()["detail"]
        # Must contain actionable guidance
        assert any(word in detail.lower() for word in ["training", "script", "train"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# POST /inspect-part — Happy paths
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestInspectPartHappyPaths:

    def _setup_ready_service(self, monkeypatch, inspection_response: InspectionResponse):
        """Wire up all three globals with a service mock that returns the given response."""
        mock_det = MagicMock()
        mock_det._is_trained = True

        mock_svc = MagicMock()
        mock_svc.run_full_inspection.return_value = inspection_response

        monkeypatch.setattr(main_module, "encoder", MagicMock())
        monkeypatch.setattr(main_module, "detector", mock_det)
        monkeypatch.setattr(main_module, "inspection_service", mock_svc)

    def test_normal_part_returns_pass(self, monkeypatch):
        """Normal part → 200 with status=Pass, agent_analysis=None."""
        self._setup_ready_service(monkeypatch, _make_pass_response())

        async def _test():
            async with _client() as client:
                resp = await client.post(
                    "/inspect-part",
                    files={"file": ("normal_part.jpg", _fake_image_bytes(), "image/jpeg")},
                )
                return resp

        resp = run(_test())
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "Pass"
        assert body["detection"]["status"] == "Normal"
        assert body["agent_analysis"] is None
        assert body["metadata"]["latency_alert"] is False

    def test_anomaly_part_returns_fail_with_agent(self, monkeypatch):
        """Anomalous part → 200 with status=Fail and agent_analysis populated."""
        self._setup_ready_service(monkeypatch, _make_fail_response())

        async def _test():
            async with _client() as client:
                resp = await client.post(
                    "/inspect-part",
                    files={"file": ("defect_part.jpg", _fake_image_bytes(), "image/jpeg")},
                )
                return resp

        resp = run(_test())
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "Fail"
        assert body["detection"]["status"] == "Anomaly"
        assert body["detection"]["anomaly_score"] < 0
        assert body["agent_analysis"] is not None
        assert body["agent_analysis"]["defect_confirmed"] is True
        assert 0 <= body["agent_analysis"]["severity_score"] <= 1

    def test_pass_response_has_correct_schema_fields(self, monkeypatch):
        """Response envelope must contain all required fields."""
        self._setup_ready_service(monkeypatch, _make_pass_response())

        async def _test():
            async with _client() as client:
                resp = await client.post(
                    "/inspect-part",
                    files={"file": ("part.jpg", _fake_image_bytes(), "image/jpeg")},
                )
                return resp

        body = run(_test()).json()
        assert "status" in body
        assert "detection" in body
        assert "anomaly_score" in body["detection"]
        assert "agent_analysis" in body
        assert "metadata" in body
        assert "processing_time_seconds" in body["metadata"]
        assert "latency_alert" in body["metadata"]
        assert "image_file" in body["metadata"]

    def test_latency_alert_true_when_flagged(self, monkeypatch):
        """When pipeline was slow, metadata.latency_alert must come through as True."""
        self._setup_ready_service(monkeypatch, _make_fail_response())  # fail has latency_alert=True

        async def _test():
            async with _client() as client:
                resp = await client.post(
                    "/inspect-part",
                    files={"file": ("slow_part.jpg", _fake_image_bytes(), "image/jpeg")},
                )
                return resp

        body = run(_test()).json()
        assert body["metadata"]["latency_alert"] is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# POST /inspect-part — Error paths from pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestInspectPartErrorPaths:

    def test_runtime_error_from_pipeline_returns_503(self, monkeypatch):
        """
        RuntimeError raised by InspectionService (e.g., untrained check inside service)
        must map to 503, not 500. This allows orchestrators to retry on 503 but alert on 500.
        """
        mock_det = MagicMock()
        mock_det._is_trained = True
        mock_svc = MagicMock()
        mock_svc.run_full_inspection.side_effect = RuntimeError(
            "AnomalyDetector is not trained."
        )
        monkeypatch.setattr(main_module, "encoder", MagicMock())
        monkeypatch.setattr(main_module, "detector", mock_det)
        monkeypatch.setattr(main_module, "inspection_service", mock_svc)

        async def _test():
            async with _client() as client:
                resp = await client.post(
                    "/inspect-part",
                    files={"file": ("part.jpg", _fake_image_bytes(), "image/jpeg")},
                )
                return resp

        resp = run(_test())
        assert resp.status_code == 503

    def test_unexpected_exception_returns_500(self, monkeypatch):
        """
        Any unexpected exception (e.g. disk full, corrupt image) must map to 500.
        """
        mock_det = MagicMock()
        mock_det._is_trained = True
        mock_svc = MagicMock()
        mock_svc.run_full_inspection.side_effect = OSError("Disk quota exceeded")
        monkeypatch.setattr(main_module, "encoder", MagicMock())
        monkeypatch.setattr(main_module, "detector", mock_det)
        monkeypatch.setattr(main_module, "inspection_service", mock_svc)

        async def _test():
            async with _client() as client:
                resp = await client.post(
                    "/inspect-part",
                    files={"file": ("part.jpg", _fake_image_bytes(), "image/jpeg")},
                )
                return resp

        resp = run(_test())
        assert resp.status_code == 500
        assert "Pipeline error" in resp.json()["detail"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET /metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestMetricsEndpoint:

    def test_metrics_endpoint_returns_200(self):
        async def _test():
            async with _client() as client:
                resp = await client.get("/metrics/", follow_redirects=True)
                return resp

        resp = run(_test())
        assert resp.status_code == 200

    def test_metrics_contains_inspection_latency(self):
        """Prometheus metric name must be present in the raw text output."""
        async def _test():
            async with _client() as client:
                resp = await client.get("/metrics/", follow_redirects=True)
                return resp

        resp = run(_test())
        assert "inspection_latency_seconds" in resp.text

    def test_metrics_content_type_is_prometheus(self):
        """Response must use the Prometheus text exposition format content-type."""
        async def _test():
            async with _client() as client:
                resp = await client.get("/metrics/", follow_redirects=True)
                return resp

        resp = run(_test())
        assert "text/plain" in resp.headers.get("content-type", "")
