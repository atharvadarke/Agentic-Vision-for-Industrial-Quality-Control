"""
tests/test_config.py — AppSettings Configuration Validation Tests  (MEDIUM RISK GAP #1)

Tests every validator in src/config.py:
  - GROQ_API_KEY validator  : empty string, whitespace-only, leading/trailing space stripping
  - Field range constraints  : contamination (gt=0, lt=0.5), n_estimators (ge=10, le=1000),
                               latency_threshold (gt=0), llm_temperature (ge=0, le=2)
  - Literal constraints      : encoder_device ("cpu"|"cuda"), log_level (5 valid values)
  - Path expansion validator : ~ expansion, relative → absolute, already-absolute unchanged
  - Default values           : verify every default is correct and the right type

Strategy:
  env vars take priority over .env file in pydantic-settings, so all tests use
  monkeypatch.setenv / monkeypatch.delenv to control the GROQ_API_KEY value
  without touching the real .env file.
"""

import os
import pytest
from pydantic import ValidationError


# ─── Helper ───────────────────────────────────────────────────────────────────

def _make_settings(monkeypatch, api_key="gsk_test_key_abc123", **extra_env):
    """
    Build a fresh AppSettings with a fake-but-valid API key.
    Additional env overrides can be passed as keyword arguments.
    Always bypasses the lru_cache singleton so we get a fresh instance.
    """
    monkeypatch.setenv("GROQ_API_KEY", api_key)
    for k, v in extra_env.items():
        monkeypatch.setenv(k, str(v))

    # Import AFTER setting env vars so the fresh instance picks them up
    from src.config import AppSettings
    return AppSettings()  # type: ignore[call-arg]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GROQ_API_KEY validator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGroqApiKeyValidator:

    def test_empty_string_raises(self, monkeypatch):
        """
        Empty GROQ_API_KEY must raise ValidationError on startup.
        If this guard breaks, the app would call Groq with an empty key and
        fail on every anomaly detection — silently passing defective parts.
        """
        monkeypatch.setenv("GROQ_API_KEY", "")
        from src.config import AppSettings
        with pytest.raises(ValidationError, match="empty"):
            AppSettings()  # type: ignore[call-arg]

    def test_whitespace_only_raises(self, monkeypatch):
        """Whitespace-only key (e.g. copied with extra spaces) must be rejected."""
        monkeypatch.setenv("GROQ_API_KEY", "   ")
        from src.config import AppSettings
        with pytest.raises(ValidationError, match="empty"):
            AppSettings()  # type: ignore[call-arg]

    def test_tab_only_raises(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "\t\t")
        from src.config import AppSettings
        with pytest.raises(ValidationError, match="empty"):
            AppSettings()  # type: ignore[call-arg]

    def test_valid_key_accepted(self, monkeypatch):
        s = _make_settings(monkeypatch, api_key="gsk_test_abc_123")
        assert s.groq_api_key == "gsk_test_abc_123"

    def test_leading_trailing_whitespace_stripped(self, monkeypatch):
        """Key with surrounding whitespace must be accepted AND stripped."""
        monkeypatch.setenv("GROQ_API_KEY", "  gsk_real_key  ")
        from src.config import AppSettings
        s = AppSettings()  # type: ignore[call-arg]
        assert s.groq_api_key == "gsk_real_key"
        assert not s.groq_api_key.startswith(" ")
        assert not s.groq_api_key.endswith(" ")

    def test_key_with_special_characters_accepted(self, monkeypatch):
        """API keys often contain underscores and mixed case — must be accepted."""
        key = "gsk_prod_AbCd1234_xYz"
        s = _make_settings(monkeypatch, api_key=key)
        assert s.groq_api_key == key


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# isolation_forest_contamination  (gt=0.0, lt=0.5)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestContaminationConstraint:

    def test_default_is_0_05(self, monkeypatch):
        s = _make_settings(monkeypatch)
        assert s.isolation_forest_contamination == pytest.approx(0.05)

    def test_zero_contamination_raises(self, monkeypatch):
        """contamination=0.0 is invalid (must be strictly > 0)."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setenv("ISOLATION_FOREST_CONTAMINATION", "0.0")
        from src.config import AppSettings
        with pytest.raises(ValidationError):
            AppSettings()  # type: ignore[call-arg]

    def test_contamination_at_0_5_raises(self, monkeypatch):
        """contamination=0.5 is invalid (must be strictly < 0.5)."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setenv("ISOLATION_FOREST_CONTAMINATION", "0.5")
        from src.config import AppSettings
        with pytest.raises(ValidationError):
            AppSettings()  # type: ignore[call-arg]

    def test_contamination_above_0_5_raises(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setenv("ISOLATION_FOREST_CONTAMINATION", "0.9")
        from src.config import AppSettings
        with pytest.raises(ValidationError):
            AppSettings()  # type: ignore[call-arg]

    def test_negative_contamination_raises(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setenv("ISOLATION_FOREST_CONTAMINATION", "-0.1")
        from src.config import AppSettings
        with pytest.raises(ValidationError):
            AppSettings()  # type: ignore[call-arg]

    def test_valid_contamination_accepted(self, monkeypatch):
        s = _make_settings(monkeypatch, ISOLATION_FOREST_CONTAMINATION="0.03")
        assert s.isolation_forest_contamination == pytest.approx(0.03)

    def test_contamination_just_below_0_5_accepted(self, monkeypatch):
        s = _make_settings(monkeypatch, ISOLATION_FOREST_CONTAMINATION="0.499")
        assert s.isolation_forest_contamination == pytest.approx(0.499)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# isolation_forest_n_estimators  (ge=10, le=1000)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestNEstimatorsConstraint:

    def test_default_is_100(self, monkeypatch):
        s = _make_settings(monkeypatch)
        assert s.isolation_forest_n_estimators == 100

    def test_below_10_raises(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setenv("ISOLATION_FOREST_N_ESTIMATORS", "9")
        from src.config import AppSettings
        with pytest.raises(ValidationError):
            AppSettings()  # type: ignore[call-arg]

    def test_above_1000_raises(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setenv("ISOLATION_FOREST_N_ESTIMATORS", "1001")
        from src.config import AppSettings
        with pytest.raises(ValidationError):
            AppSettings()  # type: ignore[call-arg]

    def test_at_lower_boundary_10_accepted(self, monkeypatch):
        s = _make_settings(monkeypatch, ISOLATION_FOREST_N_ESTIMATORS="10")
        assert s.isolation_forest_n_estimators == 10

    def test_at_upper_boundary_1000_accepted(self, monkeypatch):
        s = _make_settings(monkeypatch, ISOLATION_FOREST_N_ESTIMATORS="1000")
        assert s.isolation_forest_n_estimators == 1000


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# latency_threshold_seconds  (gt=0.0)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestLatencyThresholdConstraint:

    def test_default_is_0_5(self, monkeypatch):
        s = _make_settings(monkeypatch)
        assert s.latency_threshold_seconds == pytest.approx(0.5)

    def test_zero_latency_raises(self, monkeypatch):
        """A zero threshold would flag EVERY request as a latency alert — invalid."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setenv("LATENCY_THRESHOLD_SECONDS", "0.0")
        from src.config import AppSettings
        with pytest.raises(ValidationError):
            AppSettings()  # type: ignore[call-arg]

    def test_negative_latency_raises(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setenv("LATENCY_THRESHOLD_SECONDS", "-1.0")
        from src.config import AppSettings
        with pytest.raises(ValidationError):
            AppSettings()  # type: ignore[call-arg]

    def test_custom_latency_accepted(self, monkeypatch):
        s = _make_settings(monkeypatch, LATENCY_THRESHOLD_SECONDS="1.0")
        assert s.latency_threshold_seconds == pytest.approx(1.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# llm_temperature  (ge=0.0, le=2.0)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestLlmTemperatureConstraint:

    def test_default_is_0_1(self, monkeypatch):
        s = _make_settings(monkeypatch)
        assert s.llm_temperature == pytest.approx(0.1)

    def test_negative_temperature_raises(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setenv("LLM_TEMPERATURE", "-0.1")
        from src.config import AppSettings
        with pytest.raises(ValidationError):
            AppSettings()  # type: ignore[call-arg]

    def test_temperature_above_2_raises(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setenv("LLM_TEMPERATURE", "2.1")
        from src.config import AppSettings
        with pytest.raises(ValidationError):
            AppSettings()  # type: ignore[call-arg]

    def test_temperature_at_boundary_0_accepted(self, monkeypatch):
        s = _make_settings(monkeypatch, LLM_TEMPERATURE="0.0")
        assert s.llm_temperature == pytest.approx(0.0)

    def test_temperature_at_boundary_2_accepted(self, monkeypatch):
        s = _make_settings(monkeypatch, LLM_TEMPERATURE="2.0")
        assert s.llm_temperature == pytest.approx(2.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# encoder_device  Literal["cpu", "cuda"]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEncoderDeviceConstraint:

    def test_default_is_cpu(self, monkeypatch):
        s = _make_settings(monkeypatch)
        assert s.encoder_device == "cpu"

    def test_cuda_accepted(self, monkeypatch):
        s = _make_settings(monkeypatch, ENCODER_DEVICE="cuda")
        assert s.encoder_device == "cuda"

    def test_invalid_device_raises(self, monkeypatch):
        """'gpu' is not a valid device identifier — must be exactly 'cpu' or 'cuda'."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setenv("ENCODER_DEVICE", "gpu")
        from src.config import AppSettings
        with pytest.raises(ValidationError):
            AppSettings()  # type: ignore[call-arg]

    def test_mps_device_raises(self, monkeypatch):
        """Apple MPS is not in the allowed Literal — must raise."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setenv("ENCODER_DEVICE", "mps")
        from src.config import AppSettings
        with pytest.raises(ValidationError):
            AppSettings()  # type: ignore[call-arg]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# log_level  Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestLogLevelConstraint:

    def test_default_is_info(self, monkeypatch):
        s = _make_settings(monkeypatch)
        assert s.log_level == "INFO"

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_all_valid_log_levels_accepted(self, monkeypatch, level):
        s = _make_settings(monkeypatch, LOG_LEVEL=level)
        assert s.log_level == level

    def test_invalid_log_level_raises(self, monkeypatch):
        """'VERBOSE' is not a standard Python log level — must raise."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
        from src.config import AppSettings
        with pytest.raises(ValidationError):
            AppSettings()  # type: ignore[call-arg]

    def test_lowercase_log_level_raises(self, monkeypatch):
        """Literal is case-sensitive — 'info' is not the same as 'INFO'."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setenv("LOG_LEVEL", "info")
        from src.config import AppSettings
        with pytest.raises(ValidationError):
            AppSettings()  # type: ignore[call-arg]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Path expansion validator  (raw_data_dir, model_path, db_path)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPathExpansionValidator:

    def test_tilde_expanded_in_raw_data_dir(self, monkeypatch):
        """~/data/raw must be expanded to the user's home directory."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setenv("RAW_DATA_DIR", "~/data/raw")
        from src.config import AppSettings
        s = AppSettings()  # type: ignore[call-arg]
        assert not s.raw_data_dir.startswith("~"), (
            f"Expected ~ to be expanded, got: {s.raw_data_dir}"
        )
        assert os.path.isabs(s.raw_data_dir), (
            f"Expected absolute path, got: {s.raw_data_dir}"
        )

    def test_tilde_expanded_in_model_path(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setenv("MODEL_PATH", "~/models/anomaly_detector.joblib")
        from src.config import AppSettings
        s = AppSettings()  # type: ignore[call-arg]
        assert not s.model_path.startswith("~")
        assert os.path.isabs(s.model_path)

    def test_tilde_expanded_in_db_path(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setenv("DB_PATH", "~/data/defect_history.db")
        from src.config import AppSettings
        s = AppSettings()  # type: ignore[call-arg]
        assert not s.db_path.startswith("~")
        assert os.path.isabs(s.db_path)

    def test_already_absolute_path_unchanged(self, monkeypatch):
        """An already-absolute path must pass through without modification."""
        abs_path = "/tmp/custom_models/anomaly_detector.joblib"
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setenv("MODEL_PATH", abs_path)
        from src.config import AppSettings
        s = AppSettings()  # type: ignore[call-arg]
        assert os.path.isabs(s.model_path)

    def test_defaults_are_absolute_paths(self, monkeypatch):
        """Default paths must all be absolute (no relative paths at startup)."""
        s = _make_settings(monkeypatch)
        assert os.path.isabs(s.raw_data_dir)
        assert os.path.isabs(s.model_path)
        assert os.path.isabs(s.db_path)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Default values sanity check
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDefaultValues:

    def test_all_defaults(self, monkeypatch):
        s = _make_settings(monkeypatch)
        assert s.groq_model == "llama-3.3-70b-versatile"
        assert s.llm_temperature == pytest.approx(0.1)
        assert s.isolation_forest_contamination == pytest.approx(0.05)
        assert s.isolation_forest_n_estimators == 100
        assert s.encoder_device == "cpu"
        assert s.feature_vector_dim == 512
        assert s.log_level == "INFO"
        assert s.latency_threshold_seconds == pytest.approx(0.5)

    def test_feature_vector_dim_default(self, monkeypatch):
        s = _make_settings(monkeypatch)
        assert s.feature_vector_dim == 512

    def test_groq_model_default(self, monkeypatch):
        s = _make_settings(monkeypatch)
        assert "llama" in s.groq_model.lower()
