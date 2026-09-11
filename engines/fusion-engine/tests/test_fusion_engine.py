"""
Tests for Layer 3 — Cross-Modal Fusion Engine.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest
import torch

# Add engines/fusion-engine/ root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.fusion_transformer import MultimodalFusionTransformer
from core.multimodal_fusion_engine import CrossModalFusionEngine
from result_types import FusionResult


# ===========================================================================
# Mock Layer 2 Stub Data Classes
# ===========================================================================

class MockFrameVisualResult:
    def __init__(self, artifact_embedding=None, branch_embeddings=None):
        self.artifact_embedding = artifact_embedding
        self.branch_embeddings = branch_embeddings


class MockBranchAResult:
    def __init__(self, frame_results):
        self.frame_results = frame_results

    @property
    def unique_track_ids(self):
        return list(self.frame_results.keys())


class MockTrackSyncMetrics:
    def __init__(self, syncnet_confidence=0.5, syncnet_offset=0, av_hubert_score=0.1,
                 phoneme_viseme_inconsistency=0.2, audio_video_delay=0.0, emotion_mismatch_score=0.1):
        self.syncnet_confidence = syncnet_confidence
        self.syncnet_offset = syncnet_offset
        self.av_hubert_score = av_hubert_score
        self.phoneme_viseme_inconsistency = phoneme_viseme_inconsistency
        self.audio_video_delay = audio_video_delay
        self.emotion_mismatch_score = emotion_mismatch_score


class MockBranchDResult:
    def __init__(self, track_metrics):
        self.track_metrics = track_metrics


class MockTrackMotionMetrics:
    def __init__(self, timesformer_score=0.1, videomae_score=0.2, slowfast_score=0.1,
                 jitter_entropy=1.2, landmark_velocity_variance=0.02, blink_asymmetry=0.05,
                 blink_duration_anomaly=0.1, frame_inconsistency_score=0.05):
        self.timesformer_score = timesformer_score
        self.videomae_score = videomae_score
        self.slowfast_score = slowfast_score
        self.jitter_entropy = jitter_entropy
        self.landmark_velocity_variance = landmark_velocity_variance
        self.blink_asymmetry = blink_asymmetry
        self.blink_duration_anomaly = blink_duration_anomaly
        self.frame_inconsistency_score = frame_inconsistency_score


class MockBranchCResult:
    def __init__(self, track_metrics):
        self.track_metrics = track_metrics


class MockTrackTemporalMetrics:
    def __init__(self, fake_prob_mean=0.1, fake_prob_variance=0.01, embedding_drift_mean=0.02, heatmap_flux_mean=0.01):
        self.fake_prob_mean = fake_prob_mean
        self.fake_prob_variance = fake_prob_variance
        self.embedding_drift_mean = embedding_drift_mean
        self.heatmap_flux_mean = heatmap_flux_mean


class MockBranchBResult:
    def __init__(self, track_metrics):
        self.track_metrics = track_metrics


class MockTrackRPPGMetrics:
    def __init__(self, physnet_score=0.1, deepphys_score=0.1, pulse_rate_mean=72.0,
                 pulse_rate_std=1.2, pulse_snr=12.0, spatial_correlation_mean=0.85):
        self.physnet_score = physnet_score
        self.deepphys_score = deepphys_score
        self.pulse_rate_mean = pulse_rate_mean
        self.pulse_rate_std = pulse_rate_std
        self.pulse_snr = pulse_snr
        self.spatial_correlation_mean = spatial_correlation_mean


class MockBranchEResult:
    def __init__(self, track_metrics):
        self.track_metrics = track_metrics


class MockTrackContinuityMetrics:
    def __init__(self, identity_drift_mean=0.01, geometry_instability=0.02):
        self.identity_drift_mean = identity_drift_mean
        self.geometry_instability = geometry_instability


class MockBranchFResult:
    def __init__(self, track_metrics):
        self.track_metrics = track_metrics


# ===========================================================================
# Unit and Integration Tests (12 Tests total)
# ===========================================================================

def test_1_transformer_dimensions():
    """1. Verifies projection and attention shape output of MultimodalFusionTransformer."""
    model = MultimodalFusionTransformer(
        audio_dim=128, visual_dim=2560, temporal_dim=128, biological_dim=128, fused_dim=256
    )
    
    B, Ta, Tv, Tt, Tb = 2, 16, 24, 16, 16
    audio_feat = torch.randn(B, Ta, 128)
    visual_feat = torch.randn(B, Tv, 2560)
    temporal_feat = torch.randn(B, Tt, 128)
    biological_feat = torch.randn(B, Tb, 128)

    logits, attribution, localization = model(
        audio_feat=audio_feat,
        visual_feat=visual_feat,
        temporal_feat=temporal_feat,
        biological_feat=biological_feat,
    )

    assert logits.shape == (B, 2)
    assert attribution.shape == (B, 4)
    assert localization.shape == (B, Tv)


def test_2_transformer_missing_modalities():
    """2. Verifies MultimodalFusionTransformer works when some modalities are None."""
    model = MultimodalFusionTransformer()
    B, Tv = 2, 16
    visual_feat = torch.randn(B, Tv, 2560)

    # Only visual is provided
    logits, attribution, localization = model(visual_feat=visual_feat)

    assert logits.shape == (B, 2)
    assert attribution.shape == (B, 4)
    assert localization.shape == (B, Tv)


def test_3_transformer_invalid_input():
    """3. Verifies error is raised if all modalities are None."""
    model = MultimodalFusionTransformer()
    with pytest.raises(ValueError, match="requires at least one modality input"):
        model()


def test_4_transformer_truncation():
    """4. Verifies sequence truncation when sequence length > max_seq_len."""
    max_len = 10
    model = MultimodalFusionTransformer(max_seq_len=max_len)
    B, Tv = 1, 25
    visual_feat = torch.randn(B, Tv, 2560)

    logits, attribution, localization = model(visual_feat=visual_feat)
    
    # Should truncate visual timeline to max_len (10)
    assert localization.shape == (B, max_len)


def test_5_localization_shape():
    """5. Verifies temporal localization output shape is (B, Tv)."""
    model = MultimodalFusionTransformer()
    B, Tv = 3, 30
    visual_feat = torch.randn(B, Tv, 2560)
    _, _, localization = model(visual_feat=visual_feat)
    
    assert localization is not None
    assert localization.shape == (3, 30)


def test_6_attribution_scores():
    """6. Verifies attribution output shape is (B, 4) and contains values in [0, 1]."""
    model = MultimodalFusionTransformer()
    B = 2
    visual_feat = torch.randn(B, 15, 2560)
    _, attribution, _ = model(visual_feat=visual_feat)

    assert attribution.shape == (B, 4)
    assert torch.all(attribution >= 0.0)
    assert torch.all(attribution <= 1.0)


def test_7_orchestrator_initialization():
    """7. Verifies CrossModalFusionEngine initializes on CPU with random/fallback weights."""
    engine = CrossModalFusionEngine(device="cpu")
    assert engine.device.type == "cpu"
    assert isinstance(engine.model, MultimodalFusionTransformer)


def test_8_orchestrator_mock_run():
    """8. Verifies processing runs with mock Layer 2 outputs."""
    engine = CrossModalFusionEngine(device="cpu")

    # Construct mock data for track 'track_01'
    # FrameVisualResults: 5 frames
    frame_results = [
        MockFrameVisualResult(artifact_embedding=np.random.randn(2560).astype(np.float32))
        for _ in range(5)
    ]
    branch_a = MockBranchAResult({"track_01": frame_results})

    branch_b = MockBranchBResult({"track_01": MockTrackTemporalMetrics()})
    branch_c = MockBranchCResult({"track_01": MockTrackMotionMetrics()})
    branch_d = MockBranchDResult({"track_01": MockTrackSyncMetrics()})
    branch_e = MockBranchEResult({"track_01": MockTrackRPPGMetrics()})
    branch_f = MockBranchFResult({"track_01": MockTrackContinuityMetrics()})
    
    result = engine.process_layer2_results(
        branch_a_result=branch_a,
        branch_b_result=branch_b,
        branch_c_result=branch_c,
        branch_d_result=branch_d,
        branch_e_result=branch_e,
        branch_f_result=branch_f,
        branch_g_result=None,  # Missing G
    )

    assert isinstance(result, FusionResult)
    assert "track_01" in result.track_results
    assert result.n_tracks == 1
    assert result.overall_verdict in ["REAL", "FAKE", "UNCERTAIN"]
    assert len(result.track_results["track_01"].localization.temporal_timeline) == 5


def test_9_orchestrator_missing_branches():
    """9. Verifies warnings are emitted and engine runs when some branches are missing."""
    engine = CrossModalFusionEngine()
    
    # Run with only Branch A provided, others None
    frame_results = [MockFrameVisualResult(artifact_embedding=np.random.randn(2560).astype(np.float32))]
    branch_a = MockBranchAResult({"track_02": frame_results})

    result = engine.process_layer2_results(branch_a_result=branch_a)
    
    assert result.n_tracks == 1
    assert len(result.processing_warnings) > 0
    # Should warn about Branch B, C, D, E, F, G missing
    assert any("Branch B" in w for w in result.processing_warnings)


def test_10_orchestrator_no_tracks():
    """10. Verifies graceful response (uncertain verdict) when no tracks are found in the inputs."""
    engine = CrossModalFusionEngine()
    result = engine.process_layer2_results()
    
    assert result.overall_verdict == "UNCERTAIN"
    assert result.overall_confidence == 0.0
    assert any("No face tracks detected" in w for w in result.processing_warnings)


def test_11_verdict_thresholds():
    """11. Verifies FAKE/REAL/UNCERTAIN verdicts are correctly mapped based on different fake probability ranges."""
    engine = CrossModalFusionEngine(threshold_fake=0.7, threshold_real=0.3)
    
    # We will temporarily mock the forward output of the model to control the logits
    # Logits mapping:
    #   [1.0, -1.0] -> Softmax -> FAKE prob approx 0.12 (should be REAL since 0.12 <= 0.3)
    #   [-1.0, 1.0] -> Softmax -> FAKE prob approx 0.88 (should be FAKE since 0.88 >= 0.7)
    #   [0.0, 0.0]  -> Softmax -> FAKE prob 0.5 (should be UNCERTAIN since 0.3 < 0.5 < 0.7)

    # 1. REAL case
    real_logits = torch.tensor([[1.0, -1.0]])
    attrib = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
    loc = torch.tensor([[0.1, 0.1, 0.1]])
    
    with torch.no_grad():
        # Test mapping directly with a patch/mock on model forward
        original_forward = engine.model.forward
        try:
            # Mock model forward to return REAL logits
            engine.model.forward = lambda **kwargs: (real_logits, attrib, loc)
            res = engine.process_layer2_results(
                branch_a_result=MockBranchAResult({"tr": [MockFrameVisualResult(np.zeros(2560))] * 3})
            )
            assert res.track_results["tr"].verdict == "REAL"
            
            # Mock model forward to return FAKE logits
            engine.model.forward = lambda **kwargs: (torch.tensor([[-1.0, 1.0]]), attrib, loc)
            res = engine.process_layer2_results(
                branch_a_result=MockBranchAResult({"tr": [MockFrameVisualResult(np.zeros(2560))] * 3})
            )
            assert res.track_results["tr"].verdict == "FAKE"

            # Mock model forward to return UNCERTAIN logits
            engine.model.forward = lambda **kwargs: (torch.tensor([[0.0, 0.0]]), attrib, loc)
            res = engine.process_layer2_results(
                branch_a_result=MockBranchAResult({"tr": [MockFrameVisualResult(np.zeros(2560))] * 3})
            )
            assert res.track_results["tr"].verdict == "UNCERTAIN"
        finally:
            engine.model.forward = original_forward


def test_12_localization_timeline_mapping():
    """12. Verifies localized timelines and culprit rankings match expectations."""
    engine = CrossModalFusionEngine()
    
    # Mock model to return specific attribution: high visual (index 1) and biological (index 3)
    # attribution tensor: [audio, visual, temporal, biological]
    mock_attrib = torch.tensor([[0.1, 0.9, 0.2, 0.8]])
    mock_loc = torch.tensor([[0.1, 0.2, 0.9, 0.95, 0.3]])
    mock_logits = torch.tensor([[-2.0, 2.0]]) # FAKE
    
    original_forward = engine.model.forward
    try:
        engine.model.forward = lambda **kwargs: (mock_logits, mock_attrib, mock_loc)
        
        res = engine.process_layer2_results(
            branch_a_result=MockBranchAResult({"tr": [MockFrameVisualResult(np.zeros(2560))] * 5})
        )
        
        track_res = res.track_results["tr"]
        assert track_res.verdict == "FAKE"
        # Check timeline mapping
        assert len(track_res.localization.temporal_timeline) == 5
        assert track_res.localization.temporal_timeline[3] == pytest.approx(0.95)
        
        # Check culprit ordering (visual, then biological, then temporal, then audio)
        culprits = list(track_res.modality_culprits.keys())
        assert culprits[0] == "visual"
        assert culprits[1] == "biological"
        assert culprits[2] == "temporal"
        assert culprits[3] == "audio"
    finally:
        engine.model.forward = original_forward
