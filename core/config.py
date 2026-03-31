"""Centralized configuration with YAML loading and env var support.

Resolution order: CLI flag > env var > config YAML > default
"""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.constants import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    IDEAL_CLIP_MAX,
    IDEAL_CLIP_MIN,
    MIN_SNR_DB,
    PEAK_NORMALIZE_DB,
    SAMPLE_RATE,
    SILENCE_THRESHOLD_DB,
)


class PreprocessingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AUDIO_TRAINER_PREPROCESS_")

    sample_rate: int = SAMPLE_RATE
    peak_normalize_db: float = PEAK_NORMALIZE_DB
    min_clip_duration: float = IDEAL_CLIP_MIN
    max_clip_duration: float = IDEAL_CLIP_MAX
    silence_threshold_db: float = SILENCE_THRESHOLD_DB
    min_silence_ms: int = 300
    min_snr_db: float = MIN_SNR_DB
    denoise: bool = True
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    language: str = "en"
    speaker_name: str = "speaker"


class TrainingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AUDIO_TRAINER_TRAIN_")

    speaker_name: str = "speaker"
    preset: str = "default"
    sovits_epochs: int = 8
    sovits_batch_size: int = 4
    sovits_lr: float = 0.0001
    sovits_save_every_epoch: int = 2
    gpt_epochs: int = 15
    gpt_batch_size: int = 4
    gpt_lr: float = 0.00015
    gpt_save_every_epoch: int = 5
    precision: str = "fp16"
    gradient_checkpointing: bool = True
    device: str = "cuda:0"


class InferenceConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AUDIO_TRAINER_INFERENCE_")

    engine: str = "gpt-sovits"
    speaker_name: str = "speaker"
    reference_audio: Optional[str] = None
    reference_text: Optional[str] = None
    output_format: str = "wav"
    output_sample_rate: int = SAMPLE_RATE
    normalize_output: bool = True
    trim_silence: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 8000


class AppConfig(BaseSettings):
    """Top-level application config. All paths are configurable."""

    model_config = SettingsConfigDict(env_prefix="AUDIO_TRAINER_")

    data_dir: Path = DEFAULT_DATA_DIR
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR

    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def reference_dir(self) -> Path:
        return self.data_dir / "reference"

    @property
    def pretrained_dir(self) -> Path:
        return self.checkpoint_dir / "pretrained"

    @property
    def finetuned_dir(self) -> Path:
        return self.checkpoint_dir / "finetuned"


def load_config(config_path: Optional[Path] = None, **overrides) -> AppConfig:
    """Load config from YAML file, env vars, and overrides.

    Resolution order: overrides (CLI) > env vars > YAML > defaults
    """
    yaml_data = {}
    if config_path and config_path.exists():
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f) or {}

    # Merge YAML sections into sub-configs
    preprocessing_data = yaml_data.pop("preprocessing", {})
    training_data = yaml_data.pop("training", {})
    inference_data = yaml_data.pop("inference", {})

    # Build sub-configs
    preprocessing = PreprocessingConfig(**preprocessing_data)
    training = TrainingConfig(**training_data)
    inference = InferenceConfig(**inference_data)

    # Build top-level config
    top_level = {k: v for k, v in yaml_data.items() if v is not None}
    top_level.update({k: v for k, v in overrides.items() if v is not None})

    return AppConfig(
        preprocessing=preprocessing,
        training=training,
        inference=inference,
        **top_level,
    )
