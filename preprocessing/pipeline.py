"""Preprocessing pipeline orchestrator.

Runs the full preprocessing pipeline: validate → normalize → denoise → slice → transcribe.
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from core.config import PreprocessingConfig
from core.constants import SUPPORTED_AUDIO_FORMATS, WAVS_SUBDIR

from .audio_slicer import slice_file
from .denoiser import denoise_simple, denoise_with_demucs
from .normalizer import normalize_file
from .transcriber import transcribe_directory
from .validator import validate_directory

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Orchestrates the full audio preprocessing pipeline."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        speaker_name: str | None = None,
        language: str | None = None,
        denoise: bool | None = None,
    ) -> dict:
        """Run the full preprocessing pipeline.

        Args:
            input_dir: Directory containing raw audio files.
            output_dir: Directory to write processed output.
            speaker_name: Override speaker name from config.
            language: Override language from config.
            denoise: Override denoise flag from config.

        Returns:
            Summary dict with statistics about the processed dataset.
        """
        speaker = speaker_name or self.config.speaker_name
        lang = language or self.config.language
        do_denoise = denoise if denoise is not None else self.config.denoise

        logger.info(f"Starting preprocessing pipeline for speaker '{speaker}'")
        logger.info(f"  Input:  {input_dir}")
        logger.info(f"  Output: {output_dir}")
        logger.info(f"  Denoise: {do_denoise}, Language: {lang}")

        # Create output subdirectories
        normalized_dir = output_dir / "_normalized"
        denoised_dir = output_dir / "_denoised"
        wavs_dir = output_dir / WAVS_SUBDIR
        for d in [normalized_dir, denoised_dir, wavs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Stage 1: Validate
        logger.info("Stage 1/5: Validating input audio...")
        results = validate_directory(
            input_dir,
            min_duration=0.5,  # Accept anything for initial validation
            max_duration=600,  # Accept long files (will be sliced)
            min_snr=0,  # Don't reject on SNR (will denoise)
        )

        audio_files = [r.path for r in results if r.valid]
        skipped = [r for r in results if not r.valid]

        if skipped:
            logger.warning(f"Skipping {len(skipped)} invalid files:")
            for r in skipped:
                logger.warning(f"  {r.path.name}: {r.issues}")

        if not audio_files:
            raise ValueError(f"No valid audio files found in {input_dir}")

        logger.info(f"  {len(audio_files)} valid files, {len(skipped)} skipped")

        # Stage 2: Normalize (resample, mono, peak normalize)
        logger.info("Stage 2/5: Normalizing audio...")
        normalized_files = []
        for path in audio_files:
            out = normalized_dir / f"{path.stem}.wav"
            normalize_file(
                path,
                out,
                target_sr=self.config.sample_rate,
                target_peak_db=self.config.peak_normalize_db,
            )
            normalized_files.append(out)
        logger.info(f"  Normalized {len(normalized_files)} files to {self.config.sample_rate}Hz")

        # Stage 3: Denoise (optional)
        if do_denoise:
            logger.info("Stage 3/5: Denoising audio...")
            source_files = []
            for path in normalized_files:
                out = denoised_dir / path.name
                try:
                    denoise_with_demucs(path, out)
                except Exception as e:
                    logger.warning(f"Demucs failed for {path.name}, using simple denoiser: {e}")
                    denoise_simple(path, out)
                source_files.append(out)
            logger.info(f"  Denoised {len(source_files)} files")
        else:
            logger.info("Stage 3/5: Skipping denoising (disabled)")
            source_files = normalized_files

        # Stage 4: Slice into segments
        logger.info("Stage 4/5: Slicing into segments...")
        all_segments = []
        for path in source_files:
            segments = slice_file(
                path,
                wavs_dir,
                prefix=f"{path.stem}_",
                min_duration=self.config.min_clip_duration,
                max_duration=self.config.max_clip_duration,
                threshold_db=self.config.silence_threshold_db,
                min_silence_ms=self.config.min_silence_ms,
            )
            all_segments.extend(segments)
        logger.info(f"  Created {len(all_segments)} segments")

        # Stage 5: Transcribe
        logger.info("Stage 5/5: Transcribing audio segments...")
        transcripts_path = transcribe_directory(
            wavs_dir,
            output_dir,
            model_size=self.config.whisper_model,
            device=self.config.whisper_device,
            compute_type=self.config.whisper_compute_type,
            language=lang,
            speaker_name=speaker,
        )
        logger.info(f"  Transcripts written to {transcripts_path}")

        # Compute total duration
        import soundfile as sf
        total_duration = 0.0
        for seg_path in all_segments:
            info = sf.info(str(seg_path))
            total_duration += info.duration

        # Generate metadata summary
        summary = {
            "speaker_name": speaker,
            "language": lang,
            "input_files": len(audio_files),
            "skipped_files": len(skipped),
            "total_segments": len(all_segments),
            "total_duration_seconds": round(total_duration, 2),
            "total_duration_minutes": round(total_duration / 60, 2),
            "sample_rate": self.config.sample_rate,
            "denoised": do_denoise,
            "transcripts_path": str(transcripts_path),
            "wavs_dir": str(wavs_dir),
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }

        # Write metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Cleanup intermediate directories
        import shutil
        shutil.rmtree(normalized_dir, ignore_errors=True)
        if do_denoise:
            shutil.rmtree(denoised_dir, ignore_errors=True)

        logger.info(f"Pipeline complete: {len(all_segments)} segments, {summary['total_duration_minutes']} minutes")
        return summary
