"""Unit tests for model branches."""

import pytest
import torch


class TestWavLMBranch:
    def test_forward_pass(self):
        from ads.models.wavlm_branch.model import WavLMBranch
        model = WavLMBranch()
        waveform = torch.randn(1, 16000)
        output = model(waveform)
        assert "logits" in output
        assert "embedding" in output
        assert "confidence" in output
        assert output["logits"].shape == (1, 1)

    def test_forward_1d_input(self):
        from ads.models.wavlm_branch.model import WavLMBranch
        model = WavLMBranch()
        waveform = torch.randn(16000)
        output = model(waveform)
        assert output["logits"].shape == (1, 1)

    def test_empty_input(self):
        from ads.models.wavlm_branch.model import WavLMBranch
        model = WavLMBranch()
        waveform = torch.randn(1, 0)
        output = model(waveform)
        assert output["logits"].shape == (1, 1)


class TestXLSRBranch:
    def test_forward_pass(self):
        from ads.models.xlsr_branch.model import XLSRBranch
        model = XLSRBranch()
        waveform = torch.randn(1, 16000)
        output = model(waveform)
        assert "logits" in output


class TestSpectralCNN:
    def test_forward_pass(self):
        from ads.models.spectral_cnn_branch.model import SpectralCNN
        model = SpectralCNN()
        features = torch.randn(1, 3, 128, 128)
        output = model(features)
        assert "logits" in output

    def test_2d_input(self):
        from ads.models.spectral_cnn_branch.model import SpectralCNN
        model = SpectralCNN()
        features = torch.randn(40, 100)
        output = model(features)
        assert "logits" in output


class TestVoiceBiometrics:
    def test_forward_pass(self):
        from ads.models.voice_biometrics.model import VoiceBiometricsBranch
        model = VoiceBiometricsBranch()
        features = torch.randn(1, 100, 80)
        output = model(features)
        assert "logits" in output
        assert "speaker_consistency" in output

    def test_speaker_similarity(self):
        from ads.models.voice_biometrics.model import VoiceBiometricsBranch
        model = VoiceBiometricsBranch()
        emb1 = torch.randn(1, 192)
        emb2 = torch.randn(1, 192)
        sim = model.compute_speaker_similarity(emb1, emb2)
        assert 0 <= sim.item() <= 1

    def test_drift_detection(self):
        from ads.models.voice_biometrics.model import VoiceBiometricsBranch
        model = VoiceBiometricsBranch()
        embs = [torch.randn(192) for _ in range(3)]
        result = model.detect_speaker_drift(embs)
        assert "drift_score" in result
        assert "is_consistent" in result


class TestTemporalSplice:
    def test_forward_pass(self):
        from ads.models.temporal_splice.model import TemporalSpliceBranch
        model = TemporalSpliceBranch()
        features = torch.randn(1, 50, 1024)
        output = model(features)
        assert "frame_logits" in output
        assert "frame_probs" in output

    def test_decode_frames(self):
        from ads.models.temporal_splice.model import TemporalSpliceBranch
        model = TemporalSpliceBranch()
        probs = torch.randn(1, 50, 2).softmax(dim=-1)
        labels = model.decode_frames(probs, threshold=0.5)
        assert labels.shape == (1, 50)

    def test_manipulated_regions(self):
        from ads.models.temporal_splice.model import TemporalSpliceBranch
        model = TemporalSpliceBranch()
        labels = torch.zeros(1, 100).long()
        labels[0, 20:40] = 1
        labels[0, 60:80] = 1
        regions = model.get_manipulated_regions(labels, hop_time=0.01)
        assert len(regions) == 1
        starts, ends = regions[0]
        assert len(starts) == 2


class TestDiffusionDetector:
    def test_forward_pass(self):
        from ads.models.diffusion_detector.model import DiffusionDetectorBranch
        model = DiffusionDetectorBranch()
        waveform = torch.randn(1, 16000)
        output = model(waveform)
        assert "logits" in output
        assert "diffusion_probability" in output


class TestAdversarialDetector:
    def test_forward_pass(self):
        from ads.models.adversarial_detector.model import AdversarialDetectorBranch
        model = AdversarialDetectorBranch()
        waveform = torch.randn(1, 16000)
        output = model(waveform)
        assert "logits" in output
        assert "robustness_score" in output


class TestFusionEngine:
    def test_forward_pass(self):
        from ads.fusion.engine import FusionTransformer
        model = FusionTransformer()
        embeddings = {
            "wavlm": torch.randn(1, 256),
            "xlsr": torch.randn(1, 256),
            "spectral": torch.randn(1, 256),
            "voice_biometrics": torch.randn(1, 192),
            "temporal": torch.randn(1, 256),
            "diffusion": torch.randn(1, 128),
            "adversarial": torch.randn(1, 128),
        }
        output = model(embeddings)
        assert "probability" in output
        assert "confidence" in output
        assert "uncertainty" in output

    def test_partial_branches(self):
        from ads.fusion.engine import FusionTransformer
        model = FusionTransformer()
        embeddings = {
            "wavlm": torch.randn(1, 256),
            "spectral": torch.randn(1, 256),
        }
        output = model(embeddings)
        assert "probability" in output


class TestLocalization:
    def test_forward_pass(self):
        from ads.localization.engine import LocalizationEngine
        engine = LocalizationEngine()
        features = torch.randn(1, 100, 256)
        logits = torch.randn(1, 100, 2)
        result = engine.forward(features, logits, hop_time=0.01)
        assert "manipulated_segments" in result
        assert "timeline" in result
        assert "frame_predictions" in result
