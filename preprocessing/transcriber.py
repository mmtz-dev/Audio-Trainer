"""ASR transcription using Faster-Whisper."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def transcribe_file(
    audio_path: Path,
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    language: str = "en",
) -> str:
    """Transcribe a single audio file using Faster-Whisper.

    Returns the transcription text.
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(str(audio_path), language=language, beam_size=5)

    text = " ".join(segment.text.strip() for segment in segments)
    logger.debug(f"Transcribed {audio_path.name}: {text[:80]}...")
    return text


def transcribe_directory(
    audio_dir: Path,
    output_dir: Path,
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    language: str = "en",
    speaker_name: str = "speaker",
) -> Path:
    """Transcribe all WAV files in a directory.

    Creates .lab files alongside each WAV and a master transcripts.list file.
    Returns path to the transcripts.list file.
    """
    from faster_whisper import WhisperModel

    logger.info(f"Loading Whisper model: {model_size} on {device}...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {audio_dir}")

    lang_code = language.upper()[:2]
    transcript_entries = []

    for wav_path in wav_files:
        # Transcribe
        segments, info = model.transcribe(str(wav_path), language=language, beam_size=5)
        text = " ".join(segment.text.strip() for segment in segments).strip()

        if not text:
            logger.warning(f"Empty transcription for {wav_path.name}, skipping")
            continue

        # Write .lab file alongside WAV
        lab_path = wav_path.with_suffix(".lab")
        lab_path.write_text(text, encoding="utf-8")

        # Add to transcript list (GPT-SoVITS format)
        # Format: audio_path|speaker_name|language_code|transcription
        transcript_entries.append(f"{wav_path}|{speaker_name}|{lang_code}|{text}")

        logger.debug(f"Transcribed {wav_path.name}: {text[:60]}...")

    # Write master transcripts.list
    output_dir.mkdir(parents=True, exist_ok=True)
    transcripts_path = output_dir / "transcripts.list"
    transcripts_path.write_text("\n".join(transcript_entries) + "\n", encoding="utf-8")

    logger.info(f"Transcribed {len(transcript_entries)} files -> {transcripts_path}")
    return transcripts_path
