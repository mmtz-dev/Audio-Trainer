"""Silence-based audio slicing into segments."""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from core.constants import IDEAL_CLIP_MAX, IDEAL_CLIP_MIN, MIN_SILENCE_MS, SILENCE_THRESHOLD_DB

logger = logging.getLogger(__name__)


def detect_silence_boundaries(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = SILENCE_THRESHOLD_DB,
    min_silence_ms: int = MIN_SILENCE_MS,
) -> list[tuple[int, int]]:
    """Detect silence regions in audio.

    Returns list of (start_sample, end_sample) tuples for silent regions.
    """
    threshold_amplitude = 10 ** (threshold_db / 20.0)
    min_silence_samples = int(sr * min_silence_ms / 1000)

    # Compute RMS energy in small windows
    window_size = int(sr * 0.02)  # 20ms windows
    hop = window_size // 2

    silent_regions = []
    in_silence = False
    silence_start = 0

    for i in range(0, len(audio) - window_size, hop):
        window = audio[i : i + window_size]
        rms = np.sqrt(np.mean(window**2))

        if rms < threshold_amplitude:
            if not in_silence:
                silence_start = i
                in_silence = True
        else:
            if in_silence:
                silence_end = i
                if (silence_end - silence_start) >= min_silence_samples:
                    silent_regions.append((silence_start, silence_end))
                in_silence = False

    # Handle trailing silence
    if in_silence:
        silence_end = len(audio)
        if (silence_end - silence_start) >= min_silence_samples:
            silent_regions.append((silence_start, silence_end))

    return silent_regions


def find_zero_crossing(audio: np.ndarray, position: int, search_range: int = 1024) -> int:
    """Find the nearest zero crossing to a position."""
    start = max(0, position - search_range)
    end = min(len(audio), position + search_range)
    segment = audio[start:end]

    zero_crossings = np.where(np.diff(np.signbit(segment)))[0]
    if len(zero_crossings) == 0:
        return position

    # Find closest zero crossing to the target position
    target_idx = position - start
    closest = zero_crossings[np.argmin(np.abs(zero_crossings - target_idx))]
    return start + closest


def slice_audio(
    audio: np.ndarray,
    sr: int,
    min_duration: float = IDEAL_CLIP_MIN,
    max_duration: float = IDEAL_CLIP_MAX,
    threshold_db: float = SILENCE_THRESHOLD_DB,
    min_silence_ms: int = MIN_SILENCE_MS,
) -> list[tuple[int, int]]:
    """Slice audio into segments at silence boundaries.

    Returns list of (start_sample, end_sample) tuples for each segment.
    """
    min_samples = int(sr * min_duration)
    max_samples = int(sr * max_duration)

    silent_regions = detect_silence_boundaries(audio, sr, threshold_db, min_silence_ms)

    if not silent_regions:
        # No silence found — return the whole audio if within limits
        if len(audio) <= max_samples:
            return [(0, len(audio))]
        # Force-split at max duration boundaries
        segments = []
        pos = 0
        while pos < len(audio):
            end = min(pos + max_samples, len(audio))
            if end < len(audio):
                end = find_zero_crossing(audio, end)
            segments.append((pos, end))
            pos = end
        return segments

    # Split at silence midpoints
    segments = []
    segment_start = 0

    for silence_start, silence_end in silent_regions:
        split_point = find_zero_crossing(audio, (silence_start + silence_end) // 2)
        segment_length = split_point - segment_start

        # Check if segment is long enough
        if segment_length >= min_samples:
            segments.append((segment_start, split_point))
            segment_start = split_point
        elif segment_length >= max_samples:
            # Segment too long even with this split — force split
            segments.append((segment_start, split_point))
            segment_start = split_point

    # Handle remaining audio
    if segment_start < len(audio):
        remaining = len(audio) - segment_start
        if remaining >= min_samples:
            segments.append((segment_start, len(audio)))
        elif segments:
            # Merge short trailing segment with previous
            prev_start, _ = segments[-1]
            segments[-1] = (prev_start, len(audio))

    return segments


def slice_file(
    input_path: Path,
    output_dir: Path,
    prefix: str = "",
    min_duration: float = IDEAL_CLIP_MIN,
    max_duration: float = IDEAL_CLIP_MAX,
    threshold_db: float = SILENCE_THRESHOLD_DB,
    min_silence_ms: int = MIN_SILENCE_MS,
) -> list[Path]:
    """Slice a single audio file into segments, saving each to output_dir.

    Returns list of output file paths.
    """
    audio, sr = sf.read(input_path, dtype="float32")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    segments = slice_audio(audio, sr, min_duration, max_duration, threshold_db, min_silence_ms)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    for i, (start, end) in enumerate(segments):
        segment = audio[start:end]
        filename = f"{prefix}{i:04d}.wav"
        output_path = output_dir / filename
        sf.write(output_path, segment, sr, subtype="PCM_16")
        output_paths.append(output_path)

    logger.info(f"Sliced {input_path.name} into {len(output_paths)} segments")
    return output_paths
