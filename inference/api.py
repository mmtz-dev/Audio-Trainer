"""FastAPI REST API for TTS inference."""

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from core.config import load_config
from .postprocess import postprocess_file
from .synthesizer import Synthesizer, list_speakers

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Audio-Trainer TTS API",
    description="Text-to-speech API powered by GPT-SoVITS",
    version="0.1.0",
)

# Global state
_config = None
_synthesizer = None


def get_config():
    global _config
    if _config is None:
        config_path = Path("configs/inference.yaml")
        _config = load_config(config_path if config_path.exists() else None)
    return _config


def get_synthesizer() -> Synthesizer:
    global _synthesizer
    if _synthesizer is None:
        config = get_config()
        _synthesizer = Synthesizer(config.inference)
    return _synthesizer


class TTSRequest(BaseModel):
    text: str = Field(..., max_length=5000, description="Text to synthesize")
    speaker: str = Field(default="speaker", description="Speaker name")
    reference_audio: str | None = Field(default=None, description="Path to reference audio clip")
    reference_text: str | None = Field(default=None, description="Transcript of reference audio")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier")
    output_format: str = Field(default="wav", description="Output audio format")


class TTSBatchRequest(BaseModel):
    texts: list[str] = Field(..., max_items=100, description="List of texts to synthesize")
    speaker: str = Field(default="speaker", description="Speaker name")
    reference_audio: str | None = Field(default=None, description="Path to reference audio clip")
    reference_text: str | None = Field(default=None, description="Transcript of reference audio")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier")


@app.get("/v1/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "engine": "gpt-sovits"}


@app.get("/v1/speakers")
async def speakers():
    """List available trained speakers."""
    config = get_config()
    available = list_speakers(config.checkpoint_dir)
    return {"speakers": available}


@app.post("/v1/tts")
async def tts(request: TTSRequest):
    """Generate speech from text.

    Returns the generated audio file.
    """
    config = get_config()
    synth = get_synthesizer()

    # Load speaker model if not already loaded
    try:
        synth.load_speaker(request.speaker, config.finetuned_dir)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Generate
    output_dir = config.output_dir / "api"
    output_dir.mkdir(parents=True, exist_ok=True)

    import uuid
    output_path = output_dir / f"{uuid.uuid4().hex}.wav"

    ref_audio = Path(request.reference_audio) if request.reference_audio else None

    try:
        synth.synthesize(
            text=request.text,
            output_path=output_path,
            reference_audio=ref_audio,
            reference_text=request.reference_text,
            speed=request.speed,
        )

        # Post-process
        if config.inference.normalize_output or config.inference.trim_silence:
            postprocess_file(
                output_path,
                do_normalize=config.inference.normalize_output,
                do_trim=config.inference.trim_silence,
            )

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"tts_{output_path.stem}.wav",
        )

    except Exception as e:
        logger.error(f"TTS failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/tts/batch")
async def tts_batch(request: TTSBatchRequest):
    """Generate speech for multiple texts.

    Returns JSON with paths to generated files.
    """
    config = get_config()
    synth = get_synthesizer()

    try:
        synth.load_speaker(request.speaker, config.finetuned_dir)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    ref_audio = Path(request.reference_audio) if request.reference_audio else None

    output_dir = config.output_dir / "api" / "batch"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, text in enumerate(request.texts):
        output_path = output_dir / f"batch_{i:04d}.wav"
        try:
            synth.synthesize(
                text=text,
                output_path=output_path,
                reference_audio=ref_audio,
                reference_text=request.reference_text,
                speed=request.speed,
            )
            results.append({"index": i, "status": "ok", "path": str(output_path)})
        except Exception as e:
            results.append({"index": i, "status": "error", "error": str(e)})

    return {
        "total": len(request.texts),
        "succeeded": sum(1 for r in results if r["status"] == "ok"),
        "results": results,
    }
