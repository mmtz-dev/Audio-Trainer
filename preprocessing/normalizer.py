"""Audio normalization: resample, mono conversion, peak normalization."""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from core.constants import CHANNELS, PEAK_NORMALIZE_DB, SAMPLE_RATE

logger = logging.getLogger(__name__)


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono by averaging channels."""
    if audio.ndim > 1:
        return np.mean(audio, axis=1)
    return audio


def peak_normalize(audio: np.ndarray, target_db: float = PEAK_NORMALIZE_DB) -> np.ndarray:
    """Peak-normalize audio to target dB level."""
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio  # Silent, don't amplify noise
    target_amplitude = 10 ** (target_db / 20.0)
    return audio * (target_amplitude / peak)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate using librosa."""
    if orig_sr == target_sr:
        return audio
    import librosa

    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def normalize_file(
    input_path: Path,
    output_path: Path,
    target_sr: int = SAMPLE_RATE,
    target_channels: int = CHANNELS,
    target_peak_db: float = PEAK_NORMALIZE_DB,
) -> Path:
    """Normalize a single audio file: resample, mono, peak normalize.

    Returns the output path.
    """
    audio, sr = sf.read(input_path, dtype="float32")

    # Convert to mono
    if target_channels == 1:
        audio = to_mono(audio)

    # Resample
    audio = resample_audio(audio, sr, target_sr)

    # Peak normalize
    audio = peak_normalize(audio, target_peak_db)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, target_sr, subtype="PCM_16")

    logger.debug(f"Normalized {input_path.name} -> {output_path.name} ({sr}Hz -> {target_sr}Hz)")
    return output_path
