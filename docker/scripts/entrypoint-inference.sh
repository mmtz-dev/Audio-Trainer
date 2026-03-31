#!/bin/bash
set -e

echo "=== Audio-Trainer: Inference Container ==="
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  No GPU detected"
echo ""

# Ensure output directories exist
mkdir -p /app/outputs /app/logs

# Start both the FastAPI server and GPT-SoVITS WebUI
if [ "$1" = "serve" ]; then
    echo "Starting FastAPI API server on port 8000..."
    echo "Starting GPT-SoVITS WebUI on port 9874..."

    # Launch GPT-SoVITS WebUI in background
    cd /opt/gpt-sovits
    python webui.py &
    WEBUI_PID=$!
    cd /app

    # Launch FastAPI in foreground
    exec python -m uvicorn inference.api:app --host 0.0.0.0 --port 8000 --workers 1

    # Cleanup on exit
    trap "kill $WEBUI_PID 2>/dev/null" EXIT
fi

# API only mode
if [ "$1" = "api" ]; then
    echo "Starting FastAPI API server on port 8000..."
    exec python -m uvicorn inference.api:app --host 0.0.0.0 --port 8000 --workers 1
fi

# WebUI only mode
if [ "$1" = "webui" ]; then
    echo "Starting GPT-SoVITS WebUI on port 9874..."
    cd /opt/gpt-sovits
    exec python webui.py
fi

# Default: show help
if [ "$1" = "--help" ] || [ -z "$1" ]; then
    echo "Usage:"
    echo "  docker compose up inference        # Start API (8000) + WebUI (9874)"
    echo "  docker compose run inference api    # API server only"
    echo "  docker compose run inference webui  # GPT-SoVITS WebUI only"
    exit 0
fi

exec "$@"
