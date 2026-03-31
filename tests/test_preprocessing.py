"""Tests for preprocessing modules."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from core.constants import SAMPLE_RATE
from preprocessing.normalizer import peak_normalize, to_mono, resample_audio
from preprocessing.validator import estimate_snr, validate_audio_file
from preprocessing.audio_slicer import detect_silence_boundaries, slice_audio


def _make_test_wav(path: Path, duration: float = 2.0, sr: int = 44100, silence: bool = False):
    """Create a test WAV file."""
    samples = int(sr * duration)
    if silence:
        audio = np.zeros(samples, dtype=np.float32)
    else:
        # Generate a simple sine wave with some noise
        t = np.linspace(0, duration, samples, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.01 * np.random.randn(samples).astype(np.float32)
    sf.write(path, audio, sr, subtype="PCM_16")
    return audio, sr


class TestNormalizer:
    def test_to_mono_stereo(self):
        stereo = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mono = to_mono(stereo)
        assert mono.ndim == 1
        assert len(mono) == 3
        np.testing.assert_allclose(mono, [1.5, 3.5, 5.5])

    def test_to_mono_already_mono(self):
        mono = np.array([1.0, 2.0, 3.0])
        result = to_mono(mono)
        np.testing.assert_array_equal(result, mono)

    def test_peak_normalize(self):
        audio = np.array([0.0, 0.25, -0.5, 0.3], dtype=np.float32)
        normalized = peak_normalize(audio, target_db=-1.0)
        target_amplitude = 10 ** (-1.0 / 20.0)
        assert abs(np.max(np.abs(normalized)) - target_amplitude) < 0.01

    def test_peak_normalize_silent(self):
        audio = np.zeros(100, dtype=np.float32)
        result = peak_normalize(audio)
        np.testing.assert_array_equal(result, audio)


class TestValidator:
    def test_estimate_snr_clean_signal(self):
        sr = 16000
        t = np.linspace(0, 1.0, sr, dtype=np.float32)
        clean = np.sin(2 * np.pi * 440 * t)
        snr = estimate_snr(clean)
        assert snr > 0  # Clean signal should have positive SNR

    def test_estimate_snr_silent(self):
        audio = np.zeros(16000, dtype=np.float32)
        snr = estimate_snr(audio)
        assert snr == 0.0

    def test_validate_audio_file_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.wav"
            _make_test_wav(path, duration=3.0)
            result = validate_audio_file(path, min_duration=0.5, max_duration=30.0, min_snr=0)
            assert result.valid
            assert result.duration > 2.5

    def test_validate_audio_file_too_short(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "short.wav"
            _make_test_wav(path, duration=0.1)
            result = validate_audio_file(path, min_duration=1.0)
            assert not result.valid
            assert any("short" in issue.lower() for issue in result.issues)

    def test_validate_audio_file_unsupported_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.xyz"
            path.write_text("not audio")
            result = validate_audio_file(path)
            assert not result.valid


class TestAudioSlicer:
    def test_detect_silence(self):
        sr = 16000
        # 1 second of audio, 0.5s silence, 1 second of audio
        audio_part = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr, dtype=np.float32))
        silence = np.zeros(sr // 2, dtype=np.float32)
        audio = np.concatenate([audio_part, silence, audio_part])

        regions = detect_silence_boundaries(audio, sr, threshold_db=-40, min_silence_ms=200)
        assert len(regions) >= 1

    def test_slice_audio_no_silence(self):
        sr = 16000
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        segments = slice_audio(audio, sr, min_duration=1.0, max_duration=10.0)
        assert len(segments) >= 1
        # All audio should be covered
        total = sum(end - start for start, end in segments)
        assert total == len(audio)

    def test_slice_audio_with_silence(self):
        sr = 16000
        # Create audio with a clear silence gap
        part1 = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 3.0, sr * 3, dtype=np.float32))
        silence = np.zeros(sr, dtype=np.float32)  # 1 second silence
        part2 = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 3.0, sr * 3, dtype=np.float32))
        audio = np.concatenate([part1, silence, part2])

        segments = slice_audio(audio, sr, min_duration=1.0, max_duration=10.0)
        assert len(segments) >= 2
