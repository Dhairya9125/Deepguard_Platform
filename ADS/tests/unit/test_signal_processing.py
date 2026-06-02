"""Unit tests for signal processing features."""

import pytest
import torch


class TestSignalProcessor:
    def test_initialization(self):
        from ads.signal_processing.features import SignalProcessor
        sp = SignalProcessor()
        assert sp.sample_rate == 16000
        assert sp.n_mfcc == 40

    def test_mel_spectrogram(self):
        from ads.signal_processing.features import SignalProcessor
        sp = SignalProcessor()
        waveform = torch.randn(16000)
        mel = sp.compute_mel_spectrogram(waveform)
        assert mel.dim() == 2
        assert mel.shape[0] == 128

    def test_mfcc(self):
        from ads.signal_processing.features import SignalProcessor
        sp = SignalProcessor()
        waveform = torch.randn(16000)
        mfcc = sp.compute_mfcc(waveform)
        assert mfcc.dim() == 2
        assert mfcc.shape[0] == 40

    def test_lfcc(self):
        from ads.signal_processing.features import SignalProcessor
        sp = SignalProcessor()
        waveform = torch.randn(16000)
        lfcc = sp.compute_lfcc(waveform)
        assert lfcc.dim() == 2

    def test_cqcc(self):
        from ads.signal_processing.features import SignalProcessor
        sp = SignalProcessor()
        waveform = torch.randn(16000)
        cqcc = sp.compute_cqcc(waveform)
        assert cqcc.dim() == 2

    def test_chroma(self):
        from ads.signal_processing.features import SignalProcessor
        sp = SignalProcessor()
        waveform = torch.randn(16000)
        chroma = sp.compute_chroma(waveform)
        assert chroma.shape[0] == 12

    def test_spectral_contrast(self):
        from ads.signal_processing.features import SignalProcessor
        sp = SignalProcessor()
        waveform = torch.randn(16000)
        contrast = sp.compute_spectral_contrast(waveform)
        assert contrast.dim() == 1

    def test_pitch(self):
        from ads.signal_processing.features import SignalProcessor
        sp = SignalProcessor()
        waveform = torch.randn(16000)
        pitch = sp.compute_pitch(waveform)
        assert pitch.dim() == 1

    def test_harmonic_features(self):
        from ads.signal_processing.features import SignalProcessor
        sp = SignalProcessor()
        waveform = torch.randn(16000)
        harmonic, percussive = sp.compute_harmonic_features(waveform)
        assert harmonic.dim() == 1
        assert percussive.dim() == 1

    def test_compute_all(self):
        from ads.signal_processing.features import SignalProcessor
        sp = SignalProcessor()
        waveform = torch.randn(1, 16000)
        features = sp.compute_all_features(waveform)
        assert features.mfcc is not None
        assert features.mel_spectrogram is not None
        assert features.lfcc is not None
        assert features.waveform is not None

    def test_stack_features(self):
        from ads.signal_processing.features import SignalProcessor
        from ads.domain.entities import SignalFeatures
        sp = SignalProcessor()
        features = SignalFeatures(
            mfcc=torch.randn(40, 100),
            lfcc=torch.randn(40, 100),
            cqcc=torch.randn(40, 100),
            chroma=torch.randn(12, 100),
        )
        stacked = SignalProcessor.stack_features(features)
        assert stacked.dim() == 3
        assert stacked.shape[0] == 4
