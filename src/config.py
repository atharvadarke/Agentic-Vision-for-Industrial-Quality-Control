"""
config.py — Centralised Application Configuration via pydantic-settings.

Replaces all scattered os.getenv() and hardcoded path strings across the
codebase with a single, validated, type-safe settings object.

Usage:
    from src.config import settings

    api_key = settings.groq_api_key
    model_path = settings.model_path

Settings are loaded from environment variables or the .env file at the
project root. pydantic-settings handles type coercion, validation, and
clear error messages for missing required values.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─── Resolve Project Root ──────────────────────────────────────────────────────
# This file lives at src/config.py, so root is two levels up.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)


class AppSettings(BaseSettings):
    """
    Application-wide settings loaded from environment variables / .env file.

    All settings with defaults are optional at runtime.
    Settings without defaults MUST be provided in .env or the environment —
    pydantic-settings will raise a clear ValidationError on startup if missing.

    Categories:
        - API / LLM credentials
        - Model configuration
        - Path configuration
        - Pipeline behaviour tuning
        - Logging
    """

    model_config = SettingsConfigDict(
        env_file=os.path.join(_PROJECT_ROOT, ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,       # GROQ_API_KEY == groq_api_key
        extra="ignore",             # Silently ignore unknown .env keys
    )

    # ── LLM / API Credentials ─────────────────────────────────────────────────

    groq_api_key: str = Field(
        ...,
        description="Groq API key for ChatGroq LLM calls. Required.",
        alias="GROQ_API_KEY",
    )

    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model identifier to use for agent inference.",
    )

    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature. Low values produce deterministic QA reports.",
    )

    # ── Path Configuration ────────────────────────────────────────────────────

    project_root: str = Field(
        default=_PROJECT_ROOT,
        description="Absolute path to the project root directory.",
    )

    raw_data_dir: str = Field(
        default=os.path.join(_PROJECT_ROOT, "data", "raw"),
        description="Directory where uploaded part images are saved.",
    )

    model_path: str = Field(
        default=os.path.join(_PROJECT_ROOT, "models", "anomaly_detector.joblib"),
        description="Path to the serialized IsolationForest model file.",
    )

    db_path: str = Field(
        default=os.path.join(_PROJECT_ROOT, "data", "defect_history.db"),
        description="Path to the SQLite historical defect database.",
    )

    # ── Pipeline Tuning ───────────────────────────────────────────────────────

    latency_threshold_seconds: float = Field(
        default=0.5,
        gt=0.0,
        description=(
            "Maximum allowable end-to-end inspection latency in seconds. "
            "Exceeding this triggers a CRITICAL assembly line halt alert."
        ),
    )

    isolation_forest_contamination: float = Field(
        default=0.05,
        gt=0.0,
        lt=0.5,
        description=(
            "Expected fraction of anomalous samples in training data. "
            "0.05 = 5% contamination is the standard industrial QC heuristic."
        ),
    )

    isolation_forest_n_estimators: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of trees in the IsolationForest ensemble.",
    )

    # ── Feature Extraction ────────────────────────────────────────────────────

    encoder_device: Literal["cpu", "cuda"] = Field(
        default="cpu",
        description="Torch device for ResNet-18 feature extraction. Edge devices use 'cpu'.",
    )

    feature_vector_dim: int = Field(
        default=512,
        description="Expected output dimension of the VisionEncoder (ResNet-18 avgpool output).",
    )

    # ── Logging ───────────────────────────────────────────────────────────────

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Python logging level for the agentic_vision logger.",
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("groq_api_key")
    @classmethod
    def api_key_must_not_be_empty(cls, v: str) -> str:
        """Prevent accidental use of a blank API key."""
        if not v or not v.strip():
            raise ValueError(
                "GROQ_API_KEY is empty. Set it in .env or as an environment variable."
            )
        return v.strip()

    @field_validator("raw_data_dir", "model_path", "db_path", mode="before")
    @classmethod
    def expand_paths(cls, v: str) -> str:
        """Expand ~ and resolve relative paths to absolute paths."""
        return os.path.abspath(os.path.expanduser(str(v)))


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """
    Return the singleton AppSettings instance.

    Uses lru_cache so the .env file is read exactly once per process,
    not on every import. Call get_settings() anywhere in the codebase
    to get the same validated settings object.

    Returns:
        AppSettings: The fully validated application configuration.

    Raises:
        pydantic_core.ValidationError: If required settings (e.g., GROQ_API_KEY)
                                       are missing or invalid on first access.

    Example:
        from src.config import get_settings
        settings = get_settings()
        print(settings.groq_api_key)
    """
    return AppSettings()  # type: ignore[call-arg]


# ─── Module-level singleton for convenience imports ───────────────────────────
# Import `settings` directly for most use-cases:
#   from src.config import settings
# Use get_settings() when you need to bypass the cache (e.g., in tests).
settings: AppSettings = get_settings()
