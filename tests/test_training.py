"""Tests for training configuration and setup."""

from pathlib import Path

import pytest
import yaml

from core.config import TrainingConfig, load_config


class TestTrainingConfig:
    def test_default_config(self):
        config = TrainingConfig()
        assert config.sovits_epochs == 8
        assert config.gpt_epochs == 15
        assert config.sovits_batch_size == 4
        assert config.precision == "fp16"
        assert config.device == "cuda:0"

    def test_config_override(self):
        config = TrainingConfig(sovits_epochs=20, gpt_lr=0.0001)
        assert config.sovits_epochs == 20
        assert config.gpt_lr == 0.0001

    def test_preset_quick_exists(self):
        path = Path("training/configs/quick.yaml")
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
            training = data.get("training", {})
            assert training.get("sovits_epochs", 0) < 8
            assert training.get("gpt_epochs", 0) < 15

    def test_preset_quality_exists(self):
        path = Path("training/configs/quality.yaml")
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
            training = data.get("training", {})
            assert training.get("sovits_epochs", 0) > 8
            assert training.get("gpt_epochs", 0) > 15


class TestAppConfig:
    def test_load_config_defaults(self):
        config = load_config()
        assert config.data_dir == Path("./data")
        assert config.output_dir == Path("./outputs")
        assert config.checkpoint_dir == Path("./checkpoints")

    def test_load_config_with_overrides(self):
        config = load_config(output_dir=Path("/tmp/custom_output"))
        assert config.output_dir == Path("/tmp/custom_output")

    def test_config_properties(self):
        config = load_config()
        assert config.raw_dir == Path("./data/raw")
        assert config.processed_dir == Path("./data/processed")
        assert config.finetuned_dir == Path("./checkpoints/finetuned")
