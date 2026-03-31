"""Audio denoising using Demucs vocal separation."""

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def denoise_with_demucs(
    input_path: Path,
    output_path: Path,
    model: str = "htdemucs",
) -> Path:
    """Separate vocals from background using Demucs, keep vocal track.

    Args:
        input_path: Path to input audio file.
        output_path: Path to write denoised audio.
        model: Demucs model name (htdemucs, htdemucs_ft, mdx_extra).

    Returns:
        Path to denoised audio file.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Run demucs separation
        cmd = [
            "python", "-m", "demucs",
            "--two-stems", "vocals",
            "-n", model,
            "--out", str(tmp_path),
            str(input_path),
        ]

        logger.info(f"Denoising {input_path.name} with {model}...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Demucs failed: {result.stderr}")
            raise RuntimeError(f"Demucs failed for {input_path.name}: {result.stderr}")

        # Find the vocals output
        stem_name = input_path.stem
        vocals_path = tmp_path / model / stem_name / "vocals.wav"

        if not vocals_path.exists():
            # Try without model subdirectory
            for p in tmp_path.rglob("vocals.wav"):
                vocals_path = p
                break

        if not vocals_path.exists():
            raise FileNotFoundError(f"Demucs vocals output not found for {input_path.name}")

        # Copy to output
        audio, sr = sf.read(vocals_path, dtype="float32")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, sr, subtype="PCM_16")

    logger.info(f"Denoised: {input_path.name} -> {output_path.name}")
    return output_path


def denoise_simple(
    input_path: Path,
    output_path: Path,
    noise_reduce_strength: float = 0.5,
) -> Path:
    """Simple noise reduction using spectral gating (fallback if Demucs unavailable).

    Uses a basic noise profile estimation from the quietest segments.
    """
    audio, sr = sf.read(input_path, dtype="float32")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Simple spectral noise gate
    import librosa

    stft = librosa.stft(audio)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Estimate noise floor from quietest 10% of frames
    frame_energy = np.mean(magnitude**2, axis=0)
    noise_threshold = np.percentile(frame_energy, 10)
    noise_frames = magnitude[:, frame_energy <= noise_threshold]

    if noise_frames.size > 0:
        noise_profile = np.mean(noise_frames, axis=1, keepdims=True)
    else:
        noise_profile = np.min(magnitude, axis=1, keepdims=True)

    # Spectral subtraction
    cleaned_magnitude = np.maximum(
        magnitude - noise_reduce_strength * noise_profile, 0
    )

    # Reconstruct
    cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
    cleaned_audio = librosa.istft(cleaned_stft, length=len(audio))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, cleaned_audio, sr, subtype="PCM_16")

    logger.info(f"Denoised (simple): {input_path.name} -> {output_path.name}")
    return output_path
