"""Post-processing for generated audio: normalization, silence trimming."""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def trim_silence(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -40.0,
    padding_ms: int = 100,
) -> np.ndarray:
    """Trim leading and trailing silence from audio."""
    threshold = 10 ** (threshold_db / 20.0)
    padding_samples = int(sr * padding_ms / 1000)

    # Find first non-silent sample
    abs_audio = np.abs(audio)
    window = int(sr * 0.01)  # 10ms window

    start = 0
    for i in range(0, len(audio) - window, window):
        if np.max(abs_audio[i : i + window]) > threshold:
            start = max(0, i - padding_samples)
            break

    end = len(audio)
    for i in range(len(audio) - window, 0, -window):
        if np.max(abs_audio[i : i + window]) > threshold:
            end = min(len(audio), i + window + padding_samples)
            break

    return audio[start:end]


def normalize_output(
    audio: np.ndarray,
    target_db: float = -1.0,
) -> np.ndarray:
    """Peak-normalize output audio."""
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    target = 10 ** (target_db / 20.0)
    return audio * (target / peak)


def postprocess_file(
    input_path: Path,
    output_path: Path | None = None,
    do_normalize: bool = True,
    do_trim: bool = True,
    target_db: float = -1.0,
) -> Path:
    """Post-process a generated audio file.

    Args:
        input_path: Path to the generated audio.
        output_path: Path to write processed audio. If None, overwrites input.
        do_normalize: Whether to peak-normalize.
        do_trim: Whether to trim silence.
        target_db: Target peak level in dB.

    Returns:
        Path to the processed file.
    """
    if output_path is None:
        output_path = input_path

    audio, sr = sf.read(input_path, dtype="float32")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if do_trim:
        audio = trim_silence(audio, sr)

    if do_normalize:
        audio = normalize_output(audio, target_db)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sr, subtype="PCM_16")

    logger.debug(f"Post-processed: {output_path}")
    return output_path
