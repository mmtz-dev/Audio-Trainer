"""Post-training evaluation: similarity scoring and quality checks."""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def compute_cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(embedding_a, embedding_b) / (norm_a * norm_b))


def evaluate_speaker_similarity(
    reference_audio: Path,
    generated_audio: Path,
    device: str = "cuda",
) -> dict:
    """Evaluate how similar generated audio is to the reference speaker.

    Uses a speaker verification model to compute embeddings and similarity.

    Returns:
        Dict with similarity score and metadata.
    """
    try:
        import torch
        import torchaudio

        # Load audio files
        ref_wav, ref_sr = torchaudio.load(str(reference_audio))
        gen_wav, gen_sr = torchaudio.load(str(generated_audio))

        # Resample to 16kHz if needed (standard for speaker verification)
        target_sr = 16000
        if ref_sr != target_sr:
            ref_wav = torchaudio.functional.resample(ref_wav, ref_sr, target_sr)
        if gen_sr != target_sr:
            gen_wav = torchaudio.functional.resample(gen_wav, gen_sr, target_sr)

        # Use a simple energy-based similarity as baseline
        # (Full speaker verification requires a dedicated model like ECAPA-TDNN)
        ref_energy = torch.sqrt(torch.mean(ref_wav**2)).item()
        gen_energy = torch.sqrt(torch.mean(gen_wav**2)).item()

        energy_ratio = min(ref_energy, gen_energy) / max(ref_energy, gen_energy) if max(ref_energy, gen_energy) > 0 else 0

        result = {
            "energy_similarity": round(energy_ratio, 4),
            "reference_duration": ref_wav.shape[1] / target_sr,
            "generated_duration": gen_wav.shape[1] / target_sr,
            "note": "For full speaker similarity, use a dedicated speaker verification model (ECAPA-TDNN, WavLM)",
        }

        logger.info(f"Evaluation: energy similarity = {energy_ratio:.4f}")
        return result

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {"error": str(e)}


def evaluate_training_run(
    checkpoint_dir: Path,
    speaker_name: str,
) -> dict:
    """Evaluate a completed training run.

    Checks for expected outputs and reports status.
    """
    speaker_dir = checkpoint_dir / speaker_name
    sovits_dir = speaker_dir / "sovits"
    gpt_dir = speaker_dir / "gpt"

    results = {
        "speaker_name": speaker_name,
        "checkpoint_dir": str(speaker_dir),
        "sovits_checkpoints": [],
        "gpt_checkpoints": [],
        "status": "unknown",
    }

    # Check SoVITS checkpoints
    if sovits_dir.exists():
        sovits_ckpts = sorted(sovits_dir.glob("*.pth"))
        results["sovits_checkpoints"] = [str(p) for p in sovits_ckpts]

    # Check GPT checkpoints
    if gpt_dir.exists():
        gpt_ckpts = sorted(gpt_dir.glob("*.ckpt"))
        results["gpt_checkpoints"] = [str(p) for p in gpt_ckpts]

    # Determine status
    if results["sovits_checkpoints"] and results["gpt_checkpoints"]:
        results["status"] = "complete"
    elif results["sovits_checkpoints"]:
        results["status"] = "partial (SoVITS only)"
    elif results["gpt_checkpoints"]:
        results["status"] = "partial (GPT only)"
    else:
        results["status"] = "no checkpoints found"

    # Check training log
    log_path = speaker_dir / "training_log.json"
    if log_path.exists():
        import json
        with open(log_path) as f:
            results["training_log"] = json.load(f)

    logger.info(f"Training evaluation for '{speaker_name}': {results['status']}")
    return results
