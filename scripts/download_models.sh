#!/bin/bash
set -e

# Download pretrained models with retry logic
# Used by setup.sh — can also be run standalone

CHECKPOINT_DIR="${AUDIO_TRAINER_CHECKPOINT_DIR:-./checkpoints/pretrained}"
MAX_RETRIES=3

mkdir -p "$CHECKPOINT_DIR"

download_with_retry() {
    local url="$1"
    local output="$2"
    local attempt=1

    while [ $attempt -le $MAX_RETRIES ]; do
        echo "  Downloading $(basename "$output") (attempt $attempt/$MAX_RETRIES)..."
        if wget -q --show-progress -O "$output" "$url"; then
            echo "  Done."
            return 0
        fi
        echo "  Retry..."
        attempt=$((attempt + 1))
    done

    echo "  FAILED after $MAX_RETRIES attempts: $url"
    return 1
}

echo "=== Downloading pretrained models ==="

# GPT-SoVITS base models are managed by GPT-SoVITS itself
# This script handles any additional models we need

# Verify downloads
echo ""
echo "Model download complete."
echo "GPT-SoVITS pretrained weights are managed by its own setup."
echo "Run 'scripts/setup.sh' for full first-time setup."
