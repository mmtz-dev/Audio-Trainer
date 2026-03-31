"""Audio-Trainer CLI — preprocess, train, generate, serve."""

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

app = typer.Typer(
    name="audio-trainer",
    help="Train and fine-tune TTS voice models using GPT-SoVITS",
    no_args_is_help=True,
)
console = Console()


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@app.command()
def preprocess(
    input_dir: Path = typer.Option(
        "./data/raw", "--input-dir", "-i", help="Directory with raw audio files"
    ),
    output_dir: Path = typer.Option(
        "./data/processed", "--output-dir", "-o", help="Directory for processed output"
    ),
    speaker_name: str = typer.Option(
        "speaker", "--speaker", "-s", help="Speaker name"
    ),
    language: str = typer.Option(
        "en", "--language", "-l", help="Language code (en, zh, ja, etc.)"
    ),
    denoise: bool = typer.Option(
        True, "--denoise/--no-denoise", help="Enable/disable denoising"
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to preprocessing config YAML"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Preprocess raw audio: validate, normalize, denoise, slice, transcribe."""
    setup_logging(verbose)

    from core.config import load_config
    from preprocessing.pipeline import PreprocessingPipeline

    config_path = config or Path("configs/preprocessing.yaml")
    app_config = load_config(config_path if config_path.exists() else None)

    pipeline = PreprocessingPipeline(app_config.preprocessing)
    summary = pipeline.run(
        input_dir=input_dir,
        output_dir=output_dir,
        speaker_name=speaker_name,
        language=language,
        denoise=denoise,
    )

    console.print(f"\n[bold green]Preprocessing complete![/]")
    console.print(f"  Segments: {summary['total_segments']}")
    console.print(f"  Duration: {summary['total_duration_minutes']} minutes")
    console.print(f"  Output:   {summary['wavs_dir']}")


@app.command()
def train(
    data_dir: Path = typer.Option(
        "./data/processed", "--data-dir", "-d", help="Directory with processed data"
    ),
    checkpoint_dir: Path = typer.Option(
        "./checkpoints/finetuned", "--checkpoint-dir", help="Directory for model checkpoints"
    ),
    speaker_name: str = typer.Option(
        "speaker", "--speaker", "-s", help="Speaker name"
    ),
    preset: str = typer.Option(
        "default", "--preset", "-p", help="Training preset: quick, default, quality"
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to training config YAML"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Train GPT-SoVITS voice model (feature extraction + SoVITS + GPT)."""
    setup_logging(verbose)

    from core.config import load_config
    from training.trainer import GPTSoVITSTrainer

    # Load preset config if specified
    preset_path = Path(f"training/configs/{preset}.yaml")
    config_path = config or (preset_path if preset_path.exists() else Path("configs/training.yaml"))
    app_config = load_config(config_path if config_path.exists() else None)

    trainer = GPTSoVITSTrainer(app_config.training)
    result = trainer.train(
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        speaker_name=speaker_name,
    )

    console.print(f"\n[bold green]Training complete![/]")
    console.print(f"  Speaker:  {result['speaker_name']}")
    console.print(f"  SoVITS:   {result['sovits_checkpoint']}")
    console.print(f"  GPT:      {result['gpt_checkpoint']}")
    console.print(f"  Duration: {result['training_duration_seconds']:.0f}s")


@app.command()
def generate(
    text: str = typer.Option(..., "--text", "-t", help="Text to synthesize"),
    speaker: str = typer.Option(
        "speaker", "--speaker", "-s", help="Speaker name"
    ),
    output_dir: Path = typer.Option(
        None, "--output-dir", "-o", help="Output directory (default: from config or ./outputs)"
    ),
    reference_audio: Optional[Path] = typer.Option(
        None, "--reference-audio", "-r", help="Reference audio for voice matching"
    ),
    reference_text: Optional[str] = typer.Option(
        None, "--reference-text", help="Transcript of reference audio"
    ),
    speed: float = typer.Option(1.0, "--speed", help="Speech speed (0.5-2.0)"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to inference config YAML"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Generate speech from text using a trained voice model."""
    setup_logging(verbose)

    from core.config import load_config
    from inference.synthesizer import Synthesizer
    from inference.postprocess import postprocess_file

    config_path = config or Path("configs/inference.yaml")
    app_config = load_config(
        config_path if config_path.exists() else None,
        output_dir=output_dir,
    )

    resolved_output_dir = output_dir or app_config.output_dir
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    synth = Synthesizer(app_config.inference)
    synth.load_speaker(speaker, app_config.finetuned_dir)

    output_path = resolved_output_dir / f"{speaker}_output.wav"
    synth.synthesize(
        text=text,
        output_path=output_path,
        reference_audio=reference_audio,
        reference_text=reference_text,
        speed=speed,
    )

    if app_config.inference.normalize_output or app_config.inference.trim_silence:
        postprocess_file(
            output_path,
            do_normalize=app_config.inference.normalize_output,
            do_trim=app_config.inference.trim_silence,
        )

    console.print(f"\n[bold green]Generated:[/] {output_path}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="API host"),
    port: int = typer.Option(8000, "--port", help="API port"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to inference config YAML"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Start the FastAPI TTS server."""
    setup_logging(verbose)
    import uvicorn

    console.print(f"Starting API server on {host}:{port}")
    uvicorn.run("inference.api:app", host=host, port=port, workers=1)


@app.command()
def setup(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Download pretrained models (first-time setup)."""
    setup_logging(verbose)
    import subprocess

    console.print("Running first-time setup...")
    result = subprocess.run(["bash", "scripts/setup.sh"], check=False)
    if result.returncode == 0:
        console.print("[bold green]Setup complete![/]")
    else:
        console.print("[bold red]Setup encountered errors. Check output above.[/]")


@app.command()
def evaluate(
    speaker: str = typer.Option(
        "speaker", "--speaker", "-s", help="Speaker name to evaluate"
    ),
    checkpoint_dir: Path = typer.Option(
        "./checkpoints/finetuned", "--checkpoint-dir", help="Checkpoint directory"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Evaluate a trained voice model."""
    setup_logging(verbose)

    from training.evaluate import evaluate_training_run

    result = evaluate_training_run(checkpoint_dir, speaker)

    console.print(f"\n[bold]Evaluation: {speaker}[/]")
    console.print(f"  Status: {result['status']}")
    console.print(f"  SoVITS checkpoints: {len(result['sovits_checkpoints'])}")
    console.print(f"  GPT checkpoints:    {len(result['gpt_checkpoints'])}")


if __name__ == "__main__":
    app()
