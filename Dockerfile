# syntax=docker/dockerfile:1
FROM python:3.12-slim

LABEL org.opencontainers.image.title="Smart Buoy Command & Control AI Platform" \
      org.opencontainers.image.description="End-to-End Ocean Buoy Telemetry, Spectral Wave Physics, Deep Learning Forecasting & YOLOv11m Marine Computer Vision" \
      org.opencontainers.image.version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=8000

WORKDIR /app

# Install system runtime dependencies required by OpenCV / Ultralytics / Matplotlib and curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (using PyTorch CPU wheel index by default for compact container images)
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy application source, real-world NOAA datasets, H100 pre-trained weights, validation samples, and tests
COPY app ./app
COPY data ./data
COPY weights ./weights
COPY samples ./samples
COPY assets ./assets
COPY tests ./tests
COPY README.md ./README.md

# Create non-root runtime user and ensure writable cache/data directories
RUN useradd --create-home --uid 1000 buoy \
    && mkdir -p /tmp/Ultralytics /tmp/matplotlib /app/data \
    && chown -R buoy:buoy /app /tmp/Ultralytics /tmp/matplotlib

USER buoy

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=25s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/api/status" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
