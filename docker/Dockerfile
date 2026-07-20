# =============================================================================
# Dockerfile — FastAPI Backend (Agentic Vision for Industrial Quality Control)
# =============================================================================
# Optimized for edge deployment on CPU-only devices with ≤8GB RAM.
#
# Key constraints:
#   - python:3.10-slim base (minimal footprint)
#   - Single Uvicorn worker (--workers 1) to prevent RAM duplication
#   - PYTHONUNBUFFERED=1 for real-time log streaming to container orchestrators
#   - No GPU dependencies installed
#
# Build:
#   docker build -f docker/Dockerfile -t agentic-vision-api .
#
# Run standalone (without docker-compose):
#   docker run -p 8000:8000 --env-file .env agentic-vision-api
# =============================================================================

FROM python:3.10-slim

# ── Environment Configuration ─────────────────────────────────────────────────
# Disable Python output buffering for real-time Docker log visibility
ENV PYTHONUNBUFFERED=1

# Prevent Python from writing .pyc bytecode files (saves disk on edge devices)
ENV PYTHONDONTWRITEBYTECODE=1

# ── Set Working Directory ─────────────────────────────────────────────────────
WORKDIR /app

# ── Install System Dependencies ───────────────────────────────────────────────
# libgl1-mesa-glx and libglib2.0-0 are needed for OpenCV/Pillow image ops
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python Dependencies ───────────────────────────────────────────────
# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy Application Code ─────────────────────────────────────────────────────
COPY src/ ./src/
COPY .env .

# ── Copy Trained Model ────────────────────────────────────────────────────────
# Bakes the trained IsolationForest model into the image so it works standalone.
# In docker-compose, the model_data named volume overlays this for live updates.
COPY models/ ./models/

# ── Ensure runtime data directories exist ─────────────────────────────────────
# data/ is managed by the app_data named volume at runtime.
RUN mkdir -p data/raw data/processed

# ── Expose API Port ───────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health Check ──────────────────────────────────────────────────────────────
# Docker and Compose use this to determine when the service is ready.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ── Launch Command ────────────────────────────────────────────────────────────
# CRITICAL: --workers 1 prevents Uvicorn from spawning multiple processes,
# which would duplicate the ResNet-18 model in RAM and cause OOM on edge devices.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
