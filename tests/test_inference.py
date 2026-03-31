"""Tests for inference modules."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from inference.postprocess import trim_silence, normalize_output, postprocess_file
from inference.synthesizer import list_speakers


def _make_wav(path: Path, audio: np.ndarray, sr: int = 32000):
    sf.write(path, audio, sr, subtype="PCM_16")


class TestPostprocess:
    def test_trim_silence_leading(self):
        sr = 16000
        silence = np.zeros(sr, dtype=np.float32)
        signal = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr, dtype=np.float32))
        audio = np.concatenate([silence, signal])

        trimmed = trim_silence(audio, sr, threshold_db=-40)
        assert len(trimmed) < len(audio)
        assert len(trimmed) > len(signal) * 0.8  # Should keep most of signal

    def test_trim_silence_trailing(self):
        sr = 16000
        signal = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr, dtype=np.float32))
        silence = np.zeros(sr, dtype=np.float32)
        audio = np.concatenate([signal, silence])

        trimmed = trim_silence(audio, sr, threshold_db=-40)
        assert len(trimmed) < len(audio)

    def test_normalize_output(self):
        audio = np.array([0.0, 0.1, -0.2, 0.15], dtype=np.float32)
        normalized = normalize_output(audio, target_db=-1.0)
        target = 10 ** (-1.0 / 20.0)
        assert abs(np.max(np.abs(normalized)) - target) < 0.01

    def test_postprocess_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Create a test file with silence padding
            sr = 16000
            silence = np.zeros(sr // 2, dtype=np.float32)
            signal = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr, dtype=np.float32))
            audio = np.concatenate([silence, signal, silence])

            input_path = Path(tmp) / "input.wav"
            output_path = Path(tmp) / "output.wav"
            _make_wav(input_path, audio, sr)

            result = postprocess_file(input_path, output_path)
            assert result.exists()

            # Output should be shorter (trimmed) and louder (normalized)
            out_audio, out_sr = sf.read(output_path, dtype="float32")
            assert len(out_audio) < len(audio)


class TestListSpeakers:
    def test_list_speakers_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = list_speakers(Path(tmp))
            assert result == []

    def test_list_speakers_no_finetuned_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = list_speakers(Path(tmp))
            assert result == []
