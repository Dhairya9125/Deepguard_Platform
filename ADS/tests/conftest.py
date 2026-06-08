"""Shared test fixtures and configuration."""

import os
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_waveform():
    import torch
    return torch.randn(1, 16000)


@pytest.fixture
def sample_audio_path(tmp_path):
    import torchaudio
    import torch
    path = tmp_path / "test_sample.wav"
    waveform = torch.randn(1, 16000 * 3)
    torchaudio.save(str(path), waveform, 16000)
    return str(path)
