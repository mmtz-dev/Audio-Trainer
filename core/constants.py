"""Audio-Trainer constants and default values."""

from pathlib import Path

# Audio format defaults
SAMPLE_RATE = 32000  # GPT-SoVITS target sample rate
CHANNELS = 1  # Mono
BIT_DEPTH = 16
AUDIO_FORMAT = "wav"

# Preprocessing defaults
MIN_CLIP_DURATION = 0.5  # seconds
MAX_CLIP_DURATION = 30.0  # seconds
IDEAL_CLIP_MIN = 2.0  # seconds
IDEAL_CLIP_MAX = 15.0  # seconds
SILENCE_THRESHOLD_DB = -40
MIN_SILENCE_MS = 300
PEAK_NORMALIZE_DB = -1.0
MIN_SNR_DB = 20.0

# Supported input formats
SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a", ".wma"}

# Default directory structure
DEFAULT_DATA_DIR = Path("./data")
DEFAULT_RAW_DIR = DEFAULT_DATA_DIR / "raw"
DEFAULT_PROCESSED_DIR = DEFAULT_DATA_DIR / "processed"
DEFAULT_REFERENCE_DIR = DEFAULT_DATA_DIR / "reference"
DEFAULT_CHECKPOINT_DIR = Path("./checkpoints")
DEFAULT_PRETRAINED_DIR = DEFAULT_CHECKPOINT_DIR / "pretrained"
DEFAULT_FINETUNED_DIR = DEFAULT_CHECKPOINT_DIR / "finetuned"
DEFAULT_OUTPUT_DIR = Path("./outputs")

# GPT-SoVITS specific
TRANSCRIPT_LIST_FILENAME = "transcripts.list"
WAVS_SUBDIR = "wavs"
