"""Validate audio files for quality and format requirements."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from core.constants import (
    MAX_CLIP_DURATION,
    MIN_CLIP_DURATION,
    MIN_SNR_DB,
    SUPPORTED_AUDIO_FORMATS,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    path: Path
    valid: bool
    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    snr_db: float = 0.0
    issues: list[str] | None = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []


def estimate_snr(audio: np.ndarray, frame_length: int = 2048) -> float:
    """Estimate SNR by comparing signal energy to noise floor."""
    if len(audio) < frame_length:
        return 0.0

    # Frame the signal
    n_frames = len(audio) // frame_length
    frames = audio[: n_frames * frame_length].reshape(n_frames, frame_length)
    frame_energy = np.mean(frames**2, axis=1)

    if len(frame_energy) == 0:
        return 0.0

    # Signal energy = top 10% of frames, noise = bottom 10%
    sorted_energy = np.sort(frame_energy)
    n_top = max(1, len(sorted_energy) // 10)
    n_bottom = max(1, len(sorted_energy) // 10)

    signal_energy = np.mean(sorted_energy[-n_top:])
    noise_energy = np.mean(sorted_energy[:n_bottom])

    if noise_energy <= 0:
        return 60.0  # Effectively silent noise floor

    snr = 10 * np.log10(signal_energy / noise_energy)
    return float(snr)


def validate_audio_file(
    path: Path,
    min_duration: float = MIN_CLIP_DURATION,
    max_duration: float = MAX_CLIP_DURATION,
    min_snr: float = MIN_SNR_DB,
) -> ValidationResult:
    """Validate a single audio file."""
    issues = []

    # Check format
    if path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
        return ValidationResult(
            path=path,
            valid=False,
            issues=[f"Unsupported format: {path.suffix}"],
        )

    # Read file
    try:
        audio, sr = sf.read(path, dtype="float32")
    except Exception as e:
        return ValidationResult(
            path=path, valid=False, issues=[f"Cannot read file: {e}"]
        )

    # Handle stereo
    if audio.ndim > 1:
        channels = audio.shape[1]
        audio = np.mean(audio, axis=1)  # Mix to mono for analysis
    else:
        channels = 1

    duration = len(audio) / sr

    # Check duration
    if duration < min_duration:
        issues.append(f"Too short: {duration:.1f}s (min {min_duration}s)")
    if duration > max_duration:
        issues.append(f"Too long: {duration:.1f}s (max {max_duration}s)")

    # Check for silence
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-6:
        issues.append("File is silent")

    # Check SNR
    snr = estimate_snr(audio)
    if snr < min_snr:
        issues.append(f"Low SNR: {snr:.1f} dB (min {min_snr} dB)")

    # Check for clipping
    max_amplitude = np.max(np.abs(audio))
    if max_amplitude >= 0.999:
        issues.append(f"Possible clipping detected (peak: {max_amplitude:.4f})")

    return ValidationResult(
        path=path,
        valid=len(issues) == 0,
        duration=duration,
        sample_rate=sr,
        channels=channels,
        snr_db=snr,
        issues=issues,
    )


def validate_directory(
    directory: Path,
    min_duration: float = MIN_CLIP_DURATION,
    max_duration: float = MAX_CLIP_DURATION,
    min_snr: float = MIN_SNR_DB,
) -> list[ValidationResult]:
    """Validate all audio files in a directory."""
    results = []
    audio_files = sorted(
        f for f in directory.iterdir() if f.suffix.lower() in SUPPORTED_AUDIO_FORMATS
    )

    if not audio_files:
        logger.warning(f"No audio files found in {directory}")
        return results

    for path in audio_files:
        result = validate_audio_file(path, min_duration, max_duration, min_snr)
        if not result.valid:
            logger.warning(f"Validation issues for {path.name}: {result.issues}")
        results.append(result)

    valid_count = sum(1 for r in results if r.valid)
    logger.info(f"Validated {len(results)} files: {valid_count} passed, {len(results) - valid_count} failed")
    return results
