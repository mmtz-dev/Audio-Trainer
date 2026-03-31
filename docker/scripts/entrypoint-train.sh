#!/bin/bash
set -e

echo "=== Audio-Trainer: Training Container ==="
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  No GPU detected"
echo ""

# Ensure data directories exist
mkdir -p /app/data/raw /app/data/processed /app/data/reference
mkdir -p /app/checkpoints/pretrained /app/checkpoints/finetuned
mkdir -p /app/outputs /app/logs

# If first argument is a make target, run make
if [ "$1" = "make" ]; then
    shift
    exec make "$@"
fi

# If first argument is a python command, run it directly
if [ "$1" = "python" ] || [ "$1" = "audio-trainer" ]; then
    exec "$@"
fi

# If "webui" is requested, launch GPT-SoVITS WebUI
if [ "$1" = "webui" ]; then
    echo "Starting GPT-SoVITS WebUI on port 9874..."
    cd /opt/gpt-sovits
    exec python webui.py
fi

# If "setup" is requested, download pretrained models
if [ "$1" = "setup" ]; then
    exec /app/scripts/setup.sh
fi

# Default: show help
if [ "$1" = "--help" ] || [ -z "$1" ]; then
    echo "Usage:"
    echo "  docker compose --profile training run train make preprocess  # Preprocess audio data"
    echo "  docker compose --profile training run train make train       # Train voice model"
    echo "  docker compose --profile training run train make train-quick # Quick training (test)"
    echo "  docker compose --profile training run train setup            # Download pretrained models"
    echo "  docker compose --profile training run train webui            # Launch GPT-SoVITS WebUI"
    echo "  docker compose --profile training run train python -m cli.main --help"
    exit 0
fi

# Otherwise, execute the provided command
exec "$@"
