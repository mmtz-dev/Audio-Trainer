"""Batch TTS generation from text files."""

import logging
from pathlib import Path

from .postprocess import postprocess_file
from .synthesizer import Synthesizer

logger = logging.getLogger(__name__)


def generate_from_file(
    synthesizer: Synthesizer,
    text_file: Path,
    output_dir: Path,
    reference_audio: Path | None = None,
    reference_text: str | None = None,
    speed: float = 1.0,
    postprocess: bool = True,
) -> list[Path]:
    """Generate speech for each line in a text file.

    Each non-empty line becomes a separate audio file.

    Returns list of generated file paths.
    """
    if not text_file.exists():
        raise FileNotFoundError(f"Text file not found: {text_file}")

    lines = [
        line.strip()
        for line in text_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not lines:
        raise ValueError(f"No text found in {text_file}")

    logger.info(f"Generating {len(lines)} utterances from {text_file.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, text in enumerate(lines):
        output_path = output_dir / f"utterance_{i:04d}.wav"
        try:
            synthesizer.synthesize(
                text=text,
                output_path=output_path,
                reference_audio=reference_audio,
                reference_text=reference_text,
                speed=speed,
            )

            if postprocess:
                postprocess_file(output_path)

            results.append(output_path)
            logger.info(f"  [{i + 1}/{len(lines)}] {text[:50]}...")

        except Exception as e:
            logger.error(f"  [{i + 1}/{len(lines)}] Failed: {e}")

    logger.info(f"Batch complete: {len(results)}/{len(lines)} generated")
    return results
