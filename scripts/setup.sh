#!/bin/bash
set -e

echo "=== Audio-Trainer: First-Time Setup ==="
echo "Downloading pretrained models..."
echo ""

CHECKPOINT_DIR="${AUDIO_TRAINER_CHECKPOINT_DIR:-./checkpoints/pretrained}"
GPT_SOVITS_DIR="${GPT_SOVITS_DIR:-/opt/gpt-sovits}"

mkdir -p "$CHECKPOINT_DIR"

# Download GPT-SoVITS pretrained weights
echo "[1/3] Downloading GPT-SoVITS pretrained models..."
bash "$GPT_SOVITS_DIR/install.sh" 2>/dev/null || {
    echo "  GPT-SoVITS install script not found or failed."
    echo "  Models will be downloaded on first use."
}

# Download Faster-Whisper Large V3 for ASR transcription
echo "[2/3] Checking Faster-Whisper model availability..."
python -c "
from faster_whisper import WhisperModel
print('  Downloading faster-whisper large-v3 (this may take a while)...')
model = WhisperModel('large-v3', device='cpu', compute_type='int8')
print('  Faster-Whisper model ready.')
" 2>/dev/null || echo "  Whisper model will be downloaded on first use."

# Verify GPU access
echo "[3/3] Verifying GPU access..."
python -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print(f'  GPU: {gpu} ({mem:.1f} GB)')
else:
    print('  WARNING: No GPU detected. Training will be very slow.')
"

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Place voice recordings in data/raw/"
echo "  2. Run: make preprocess"
echo "  3. Run: make train"
