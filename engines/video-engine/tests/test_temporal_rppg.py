"""
Tests for VDS Layer 2, Branch E — rPPG Biological Signal Analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import numpy as np
import pytest
import torch

from feature_extraction.result_types_l2e import BranchEResult, TrackRPPGMetrics
from feature_extraction.rppg_signals import (
    compute_rppg_chrom,
    compute_rppg_pos,
    estimate_heart_rate,
    evaluate_blood_flow_realism,
    evaluate_pulse_consistency,
    extract_skin_rois,
)
from feature_extraction.physnet_model import PhysNetActiveModel
from feature_extraction.deepphys_model import DeepPhysActiveModel
from feature_extraction.temporal_rppg_analyzer import TemporalRPPGAnalyzer


# ===========================================================================
# Helpers
# ===========================================================================

def _make_dummy_landmarks(
    n_frames: int,
    pulse_freq: float = 1.5,
    fps: float = 30.0
):
    """Generate mock MediaPipe landmarks where forehead and cheeks exist."""
    forehead_indices = [54, 67, 103, 109, 10, 338, 297, 332, 284, 251]
    left_cheek_indices = [116, 118, 123, 50, 205, 207, 101]
    right_cheek_indices = [345, 347, 352, 280, 425, 427, 330]

    lm_list = []
    try:
        from preprocessing.result_types import FaceLandmarkResult
    except ImportError:
        # Mock class if not importable
        class FaceLandmarkResult:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

    for i in range(n_frames):
        lm_points = np.zeros((478, 3), dtype=np.float32)
        
        # Forehead center (112, 30)
        for j, idx in enumerate(forehead_indices):
            angle = 2 * np.pi * j / len(forehead_indices)
            lm_points[idx] = [112 + 15 * np.cos(angle), 30 + 15 * np.sin(angle), 0.0]
            
        # Left cheek center (60, 150)
        for j, idx in enumerate(left_cheek_indices):
            angle = 2 * np.pi * j / len(left_cheek_indices)
            lm_points[idx] = [60 + 15 * np.cos(angle), 150 + 15 * np.sin(angle), 0.0]
            
        # Right cheek center (162, 150)
        for j, idx in enumerate(right_cheek_indices):
            angle = 2 * np.pi * j / len(right_cheek_indices)
            lm_points[idx] = [162 + 15 * np.cos(angle), 150 + 15 * np.sin(angle), 0.0]
            
        lm_list.append(FaceLandmarkResult(
            track_id="track_001",
            frame_id=i,
            timestamp_ms=float(i * 1000.0 / fps),
            landmarks_478=lm_points,
            smoothed_landmarks=lm_points,
            jaw_open_ratio=0.1,
            eye_blink_l=0.2,
            eye_blink_r=0.2,
            detection_success=True
        ))
        
    return lm_list


def _make_layer1_result(
    track_ids: List[str],
    n_frames: int,
    fps: float = 30.0,
    pulse_freq: float = 1.5
):
    """Build a synthetic VideoPreprocessingResult stub for Branch E."""
    try:
        from preprocessing.result_types import (
            FramePacket,
            SceneBoundary,
            TrackedFace,
            VideoMetadata,
            VideoPreprocessingResult,
        )
    except ImportError:
        pytest.skip("preprocessing.result_types not importable")

    meta = VideoMetadata(
        source_path=Path("synthetic.mp4"),
        duration_s=n_frames / fps,
        native_fps=fps,
        width=320,
        height=240,
        total_frames_native=n_frames,
        total_frames_extracted=n_frames,
        target_fps=fps,
        audio_path=None,
        has_audio=False,
        audio_sample_rate=16000,
    )

    frames = []
    for i in range(n_frames):
        tracked = []
        for tid in track_ids:
            # Vary green channel over time to inject pulse frequency
            crop = np.zeros((224, 224, 3), dtype=np.uint8)
            green_val = int(128 + 20 * np.sin(2 * np.pi * pulse_freq * (i / fps)))
            crop[:, :, 1] = green_val  # Set G channel
            crop[:, :, 0] = 150        # Set R channel (skin color base)
            crop[:, :, 2] = 110        # Set B channel
            
            tracked.append(TrackedFace(
                track_id=tid,
                track_age=i,
                detection_score=0.99,
                bbox=(10, 10, 210, 210),
                frame_id=i,
                timestamp_ms=float(i * (1000.0 / fps)),
                aligned_crop=crop,
            ))
            
        fp = FramePacket(
            frame_id=i,
            timestamp_ms=float(i * (1000.0 / fps)),
            rgb_array=None,
            tracked_faces=tracked,
            scene_id=0,
        )
        frames.append(fp)

    scenes = [SceneBoundary(0, 0, n_frames - 1, 0.0, float(n_frames * 1000.0 / fps), n_frames)]

    landmarks = {}
    for tid in track_ids:
        landmarks[tid] = _make_dummy_landmarks(n_frames, pulse_freq, fps)

    return VideoPreprocessingResult(
        metadata=meta,
        frames=frames,
        scenes=scenes,
        landmarks=landmarks,
        unique_track_ids=track_ids,
    )


# ===========================================================================
# rppg_signals.py — Unit Tests
# ===========================================================================

class TestRPPGSignals:

    def test_extract_skin_rois_shapes(self):
        crop = np.zeros((224, 224, 3), dtype=np.uint8)
        crop[:, :, 1] = 120
        landmarks = _make_dummy_landmarks(1, pulse_freq=1.5, fps=30.0)[0].smoothed_landmarks
        
        rois = extract_skin_rois(crop, landmarks)
        assert "forehead" in rois
        assert "left_cheek" in rois
        assert "right_cheek" in rois
        
        # Since crop was entirely solid values, averages should match
        assert rois["forehead"].shape == (3,)
        assert rois["left_cheek"][1] == pytest.approx(120.0)

    def test_rppg_pos_sine_wave(self):
        fps = 30.0
        T = 150
        t = np.linspace(0, T / fps, T)
        # Create RGB averaged signal with a strong sinusoidal fluctuation in G channel
        rgb = np.zeros((T, 3), dtype=np.float32)
        rgb[:, 0] = 150.0  # R
        rgb[:, 1] = 120.0 + 10.0 * np.sin(2 * np.pi * 1.5 * t)  # G (1.5 Hz)
        rgb[:, 2] = 100.0  # B
        
        pulse = compute_rppg_pos(rgb, fps)
        assert len(pulse) == T
        
        # Heart rate estimate should be close to 1.5 Hz * 60 = 90 BPM
        hr, snr = estimate_heart_rate(pulse, fps)
        assert hr == pytest.approx(90.0, abs=5.0)
        assert snr > 1.0

    def test_rppg_chrom_sine_wave(self):
        fps = 30.0
        T = 150
        t = np.linspace(0, T / fps, T)
        rgb = np.zeros((T, 3), dtype=np.float32)
        rgb[:, 0] = 150.0
        rgb[:, 1] = 120.0 + 10.0 * np.sin(2 * np.pi * 1.33 * t)  # G (1.33 Hz -> ~80 BPM)
        rgb[:, 2] = 100.0
        
        pulse = compute_rppg_chrom(rgb, fps)
        assert len(pulse) == T
        
        hr, snr = estimate_heart_rate(pulse, fps)
        assert hr == pytest.approx(80.0, abs=5.0)

    def test_evaluate_pulse_consistency(self):
        fps = 30.0
        T = 180
        t = np.linspace(0, T / fps, T)
        
        # Constant frequency (1.5 Hz -> 90 BPM)
        pulse = np.sin(2 * np.pi * 1.5 * t)
        mean_hr, std_hr = evaluate_pulse_consistency(pulse, fps, window_size_s=4.0)
        assert mean_hr == pytest.approx(90.0, abs=5.0)
        assert std_hr < 2.0  # highly consistent

    def test_evaluate_blood_flow_realism(self):
        T = 100
        # Highly correlated
        s1 = np.sin(np.linspace(0, 10, T))
        s2 = np.sin(np.linspace(0, 10, T) + 0.1)  # small phase shift
        s3 = np.sin(np.linspace(0, 10, T) - 0.1)
        
        corr_real = evaluate_blood_flow_realism(s1, s2, s3)
        assert corr_real > 0.8
        
        # Uncorrelated noise
        n1 = np.random.normal(0, 1, T)
        n2 = np.random.normal(0, 1, T)
        n3 = np.random.normal(0, 1, T)
        
        corr_fake = evaluate_blood_flow_realism(n1, n2, n3)
        assert corr_fake < 0.3


# ===========================================================================
# Model Wrapper Unit Tests
# ===========================================================================

class TestRPPGModels:

    def test_physnet_model_shape(self):
        model = PhysNetActiveModel()
        
        # Video sequence input: (B, T=16, 3, H=128, W=128)
        video_in = torch.randn(2, 16, 3, 128, 128)
        pulse = model(video_in)
        
        assert pulse.shape == (2, 16)
        
        probs = model.predict_probability(video_in, fps=30.0)
        assert probs.shape == (2,)
        assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

    def test_deepphys_model_shape(self):
        model = DeepPhysActiveModel()
        
        # Video sequence input: (B, T=16, 3, H=128, W=128)
        video_in = torch.randn(2, 16, 3, 128, 128)
        pulse = model(video_in)
        
        assert pulse.shape == (2, 16)
        
        probs = model.predict_probability(video_in, fps=30.0)
        assert probs.shape == (2,)
        assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


# ===========================================================================
# TemporalRPPGAnalyzer Integration Tests
# ===========================================================================

class TestTemporalRPPGAnalyzerIntegration:

    def test_repr(self):
        analyzer = TemporalRPPGAnalyzer(device="cpu")
        assert "TemporalRPPGAnalyzer" in repr(analyzer)

    def test_analyze_empty_layer1_result(self):
        try:
            from preprocessing.result_types import VideoMetadata, VideoPreprocessingResult
        except ImportError:
            pytest.skip("preprocessing.result_types not importable")

        meta = VideoMetadata(
            source_path=Path("s.mp4"), duration_s=0.0, native_fps=25.0,
            width=320, height=240, total_frames_native=0, total_frames_extracted=0,
            target_fps=30.0, audio_path=None, has_audio=False, audio_sample_rate=16000,
        )
        layer1 = VideoPreprocessingResult(
            metadata=meta, frames=[], scenes=[], landmarks={}, unique_track_ids=[]
        )
        
        analyzer = TemporalRPPGAnalyzer(device="cpu")
        res = analyzer.analyze(layer1)
        assert isinstance(res, BranchEResult)
        assert res.n_tracks == 0
        assert len(res.processing_warnings) > 0

    def test_analyze_integration(self):
        # Generate mock Layer 1 preprocessing result with G channel color oscillation at 1.5 Hz (90 BPM)
        layer1 = _make_layer1_result(track_ids=["track_001"], n_frames=90, fps=30.0, pulse_freq=1.5)
        
        analyzer = TemporalRPPGAnalyzer(device="cpu")
        res = analyzer.analyze(layer1)
        
        assert isinstance(res, BranchEResult)
        assert res.n_tracks == 1
        assert res.unique_track_ids == ["track_001"]
        
        m = res.metrics_for_track("track_001")
        assert isinstance(m, TrackRPPGMetrics)
        assert m.track_id == "track_001"
        assert m.n_frames == 90
        assert m.n_valid_frames == 90
        
        # BPM should be reasonably near 90 BPM
        assert 45.0 <= m.pulse_rate_mean <= 180.0
        assert 0.0 <= m.pulse_rate_std <= 20.0
        assert m.pulse_snr >= 0.0
        assert -1.0 <= m.spatial_correlation_mean <= 1.0
        assert m.rppg_verdict in ["REAL", "FAKE", "UNCERTAIN"]
        
        analyzer.unload()
        assert analyzer._models_loaded is False
