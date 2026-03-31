"""TTS synthesis wrapper around GPT-SoVITS inference."""

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from core.config import InferenceConfig

logger = logging.getLogger(__name__)

GPT_SOVITS_DIR = Path(os.environ.get("GPT_SOVITS_DIR", "/opt/gpt-sovits"))


class Synthesizer:
    """GPT-SoVITS TTS synthesizer."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self._model_loaded = False
        self._gpt_model_path: Path | None = None
        self._sovits_model_path: Path | None = None

    def load_model(
        self,
        gpt_checkpoint: Path,
        sovits_checkpoint: Path,
        device: str = "cuda",
    ):
        """Load GPT and SoVITS model checkpoints."""
        if not gpt_checkpoint.exists():
            raise FileNotFoundError(f"GPT checkpoint not found: {gpt_checkpoint}")
        if not sovits_checkpoint.exists():
            raise FileNotFoundError(f"SoVITS checkpoint not found: {sovits_checkpoint}")

        self._gpt_model_path = gpt_checkpoint
        self._sovits_model_path = sovits_checkpoint
        self._device = device
        self._model_loaded = True

        logger.info(f"Model loaded: GPT={gpt_checkpoint.name}, SoVITS={sovits_checkpoint.name}")

    def load_speaker(
        self,
        speaker_name: str,
        checkpoint_dir: Path,
        device: str = "cuda",
    ):
        """Load the latest checkpoints for a speaker by name."""
        speaker_dir = checkpoint_dir / speaker_name

        # Find latest SoVITS checkpoint
        sovits_dir = speaker_dir / "sovits"
        sovits_ckpts = sorted(sovits_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime) if sovits_dir.exists() else []
        if not sovits_ckpts:
            raise FileNotFoundError(f"No SoVITS checkpoints found for speaker '{speaker_name}'")

        # Find latest GPT checkpoint
        gpt_dir = speaker_dir / "gpt"
        gpt_ckpts = sorted(gpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime) if gpt_dir.exists() else []
        if not gpt_ckpts:
            raise FileNotFoundError(f"No GPT checkpoints found for speaker '{speaker_name}'")

        self.load_model(gpt_ckpts[-1], sovits_ckpts[-1], device)

    def synthesize(
        self,
        text: str,
        output_path: Path,
        reference_audio: Path | None = None,
        reference_text: str | None = None,
        speed: float = 1.0,
    ) -> Path:
        """Generate speech from text using the loaded model.

        Args:
            text: Text to speak.
            output_path: Path to write generated audio.
            reference_audio: Reference audio clip for voice matching.
            reference_text: Transcript of the reference audio.
            speed: Speech speed multiplier (0.5 to 2.0).

        Returns:
            Path to the generated audio file.
        """
        if not self._model_loaded:
            raise RuntimeError("No model loaded. Call load_model() or load_speaker() first.")

        ref_audio = reference_audio
        if ref_audio and isinstance(ref_audio, str):
            ref_audio = Path(ref_audio)

        # Build GPT-SoVITS inference command
        cmd = [
            sys.executable,
            str(GPT_SOVITS_DIR / "GPT_SoVITS" / "inference_cli.py"),
            "--gpt_model", str(self._gpt_model_path),
            "--sovits_model", str(self._sovits_model_path),
            "--text", text,
            "--output", str(output_path),
        ]

        if ref_audio and ref_audio.exists():
            cmd.extend(["--ref_audio", str(ref_audio)])
        if reference_text:
            cmd.extend(["--ref_text", reference_text])
        if speed != 1.0:
            cmd.extend(["--speed", str(speed)])

        logger.info(f"Synthesizing: '{text[:60]}...' -> {output_path.name}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(GPT_SOVITS_DIR)},
        )

        if result.returncode != 0:
            logger.error(f"Synthesis failed: {result.stderr[-500:]}")
            raise RuntimeError(f"GPT-SoVITS inference failed:\n{result.stderr[-500:]}")

        if not output_path.exists():
            raise FileNotFoundError(f"Expected output not found: {output_path}")

        logger.info(f"Generated: {output_path}")
        return output_path

    def synthesize_batch(
        self,
        texts: list[str],
        output_dir: Path,
        reference_audio: Path | None = None,
        reference_text: str | None = None,
        speed: float = 1.0,
    ) -> list[Path]:
        """Generate speech for multiple text inputs.

        Returns list of output file paths.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        for i, text in enumerate(texts):
            output_path = output_dir / f"output_{i:04d}.wav"
            try:
                self.synthesize(text, output_path, reference_audio, reference_text, speed)
                results.append(output_path)
            except Exception as e:
                logger.error(f"Failed to synthesize item {i}: {e}")

        logger.info(f"Batch complete: {len(results)}/{len(texts)} succeeded")
        return results


def list_speakers(checkpoint_dir: Path) -> list[dict]:
    """List all available trained speakers.

    Returns list of dicts with speaker info.
    """
    finetuned_dir = checkpoint_dir / "finetuned"
    if not finetuned_dir.exists():
        return []

    speakers = []
    for speaker_dir in sorted(finetuned_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue

        sovits_dir = speaker_dir / "sovits"
        gpt_dir = speaker_dir / "gpt"

        sovits_ckpts = sorted(sovits_dir.glob("*.pth")) if sovits_dir.exists() else []
        gpt_ckpts = sorted(gpt_dir.glob("*.ckpt")) if gpt_dir.exists() else []

        if sovits_ckpts and gpt_ckpts:
            speakers.append({
                "name": speaker_dir.name,
                "sovits_checkpoint": str(sovits_ckpts[-1]),
                "gpt_checkpoint": str(gpt_ckpts[-1]),
                "ready": True,
            })

    return speakers
