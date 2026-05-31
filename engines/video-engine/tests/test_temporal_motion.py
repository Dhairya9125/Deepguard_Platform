"""
Tests for VDS Layer 2, Branch C — Temporal Motion Modeling.

Test strategy:
  - motion_signals.py: Primitives (jitter, velocity variance, blinks, inconsistency) 
    tested with known mathematical inputs.
  - Model wrappers: Verify instantiation, fallback detection, and forward pass shapes.
  - TemporalMotionAnalyzer: Test orchestration and verdict logic with synthetic stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pytest
import torch

from feature_extraction.motion_signals import (
    compute_blink_anomalies,
    compute_frame_inconsistency,
    compute_jitter_entropy,
    compute_velocity_variance,
)
from feature_extraction.result_types_l2c import BranchCResult, TrackMotionMetrics
from feature_extraction.slowfast_model import SlowFastMotionModel
from feature_extraction.temporal_motion_analyzer import TemporalMotionAnalyzer
from feature_extraction.timesformer_model import TimeSformerMotionModel
from feature_extraction.videomae_model import VideoMAEMotionModel


# ===========================================================================
# Helpers
# ===========================================================================

def _make_layer1_result(
    track_ids: List[str],
    n_frames: int,
    fps: float = 8.0,
    ear_values: Optional[Dict[str, List[float]]] = None,
    include_landmarks: bool = True,
):
    """Build a synthetic VideoPreprocessingResult stub for Branch C."""
    try:
        from preprocessing.result_types import (
            FaceLandmarkResult,
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
        native_fps=25.0,
        width=320,
        height=240,
        total_frames_native=n_frames * 3,
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
            # Create dummy face crop (224x224x3)
            crop = np.zeros((224, 224, 3), dtype=np.uint8)
            tracked.append(TrackedFace(
                track_id=tid,
                track_age=i,
                detection_score=0.98,
                bbox=(10, 10, 110, 110),
                frame_id=i,
                timestamp_ms=float(i * (1000.0 / fps)),
                aligned_crop=crop,
            ))
        fp = FramePacket(
            frame_id=i,
            timestamp_ms=float(i * (1000.0 / fps)),
            rgb_array=np.zeros((240, 320, 3), dtype=np.uint8),
            tracked_faces=tracked,
            scene_id=0,
        )
        frames.append(fp)

    scenes = [SceneBoundary(0, 0, n_frames - 1, 0.0, float(n_frames * 1000.0 / fps), n_frames)]

    landmarks = {}
    if include_landmarks:
        for tid in track_ids:
            ears = (ear_values or {}).get(tid, [0.30] * n_frames)
            # Create mock face landmarks timeline (T, 478, 3)
            # Alternate values slightly to generate non-zero displacement velocity
            track_landmarks = []
            for i in range(n_frames):
                lm_points = np.zeros((478, 3), dtype=np.float32)
                # Let's shift landmarks coordinate slightly over frames
                lm_points += float(i) * 0.05
                track_landmarks.append(FaceLandmarkResult(
                    track_id=tid,
                    frame_id=i,
                    timestamp_ms=float(i * (1000.0 / fps)),
                    landmarks_478=lm_points,
                    smoothed_landmarks=lm_points,
                    jaw_open_ratio=0.05,
                    eye_blink_l=ears[i] if i < len(ears) else 0.30,
                    eye_blink_r=ears[i] if i < len(ears) else 0.30,
                    detection_success=True,
                ))
            landmarks[tid] = track_landmarks

    return VideoPreprocessingResult(
        metadata=meta,
        frames=frames,
        scenes=scenes,
        landmarks=landmarks,
        unique_track_ids=track_ids,
    )


# ===========================================================================
# motion_signals.py — Unit Tests
# ===========================================================================

class TestJitterEntropy:

    def test_smooth_landmarks_low_entropy(self):
        # 10 frames of constant linear motion
        landmarks = np.zeros((10, 478, 3), dtype=np.float32)
        for i in range(10):
            landmarks[i] += float(i) * 0.1
        entropy = compute_jitter_entropy(landmarks)
        # Bins will have all displacements identical, resulting in 0 entropy
        assert entropy == pytest.approx(0.0, abs=1e-3)

    def test_erratic_landmarks_high_entropy(self):
        # Displacements are randomly varying
        np.random.seed(42)
        landmarks = np.zeros((20, 478, 3), dtype=np.float32)
        for i in range(20):
            landmarks[i] = np.random.normal(float(i) * 0.1, 0.5, size=(478, 3))
        entropy = compute_jitter_entropy(landmarks)
        assert entropy > 0.0

    def test_short_timeline_returns_zero(self):
        landmarks = np.zeros((2, 478, 3), dtype=np.float32)
        assert compute_jitter_entropy(landmarks) == 0.0


class TestVelocityVariance:

    def test_linear_displacement_zero_variance(self):
        landmarks = np.zeros((10, 478, 3), dtype=np.float32)
        for i in range(10):
            landmarks[i] += float(i) * 0.5
        variance = compute_velocity_variance(landmarks)
        assert variance == pytest.approx(0.0, abs=1e-6)

    def test_fluctuating_displacement_positive_variance(self):
        landmarks = np.zeros((10, 478, 3), dtype=np.float32)
        # Shift coordinate non-linearly to generate varying velocities
        for i in range(10):
            landmarks[i] += float(i * i) * 0.1
        variance = compute_velocity_variance(landmarks)
        assert variance > 0.0


class TestBlinkAnomalies:

    def test_symmetric_open_eyes_zero_asymmetry(self):
        ear_l = np.array([0.35] * 20)
        ear_r = np.array([0.35] * 20)
        asym, duration_anom = compute_blink_anomalies(ear_l, ear_r, fps=10.0)
        assert asym == pytest.approx(0.0)
        # No blinking at all in constant open eye is flagged as duration anomaly (1.0)
        assert duration_anom == 1.0

    def test_asymmetric_ear_positive_asymmetry(self):
        ear_l = np.array([0.35] * 10 + [0.10] * 2 + [0.35] * 8)
        ear_r = np.array([0.35] * 20)
        asym, _ = compute_blink_anomalies(ear_l, ear_r, fps=10.0)
        assert asym > 0.0

    def test_normal_blink_duration_profile(self):
        # 10 fps. 2 frames of blink = 0.2s duration (which is normal [0.1s, 0.4s])
        ear_l = np.array([0.35] * 8 + [0.10, 0.10] + [0.35] * 10)
        ear_r = np.array([0.35] * 8 + [0.10, 0.10] + [0.35] * 10)
        _, duration_anom = compute_blink_anomalies(ear_l, ear_r, fps=10.0)
        assert duration_anom == 0.0

    def test_too_slow_blink_duration_anomaly(self):
        # 10 fps. 8 frames of blink = 0.8s duration (>0.4s) -> anomaly
        ear_l = np.array([0.35] * 5 + [0.10] * 8 + [0.35] * 7)
        ear_r = np.array([0.35] * 5 + [0.10] * 8 + [0.35] * 7)
        _, duration_anom = compute_blink_anomalies(ear_l, ear_r, fps=10.0)
        assert duration_anom > 0.0


class TestFrameInconsistency:

    def test_identical_crops_zero_inconsistency(self):
        crops = [np.ones((224, 224, 3), dtype=np.uint8) * 128] * 10
        score = compute_frame_inconsistency(crops)
        assert score == pytest.approx(0.0)

    def test_changing_crops_positive_inconsistency(self):
        c1 = np.zeros((224, 224, 3), dtype=np.uint8)
        c2 = np.ones((224, 224, 3), dtype=np.uint8) * 255
        score = compute_frame_inconsistency([c1, c2])
        # Max difference = 1.0
        assert score == pytest.approx(1.0)


# ===========================================================================
# Model Wrapper Unit Tests
# ===========================================================================

class TestModelWrappers:

    def test_timesformer_wrapper_fallback_shape(self):
        # Test loading fallback timesformer
        model = TimeSformerMotionModel(model_name="nonexistent-model-to-trigger-fallback")
        assert model.is_fallback is True
        
        # Verify shape propagation: (B, T, C, H, W) -> (B, 2)
        dummy_input = torch.randn(2, 8, 3, 224, 224)
        logits = model(dummy_input)
        assert logits.shape == (2, 2)
        
        probs = model.predict_probability(dummy_input)
        assert probs.shape == (2,)
        assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

    def test_videomae_wrapper_fallback_shape(self):
        model = VideoMAEMotionModel(model_name="nonexistent-model-to-trigger-fallback")
        assert model.is_fallback is True
        
        dummy_input = torch.randn(2, 8, 3, 224, 224)
        logits = model(dummy_input)
        assert logits.shape == (2, 2)
        
        probs = model.predict_probability(dummy_input)
        assert probs.shape == (2,)

    def test_slowfast_model_architecture(self):
        model = SlowFastMotionModel()
        
        # Shape verification. SlowFast requires temporal dimension to be divisible by 4 
        # or handles it gracefully. 16 frames: T_slow=4, T_fast=16.
        dummy_input = torch.randn(2, 16, 3, 224, 224)
        logits = model(dummy_input)
        assert logits.shape == (2, 2)
        
        probs = model.predict_probability(dummy_input)
        assert probs.shape == (2,)


# ===========================================================================
# TemporalMotionAnalyzer Integration Tests
# ===========================================================================

class TestTemporalMotionAnalyzerIntegration:

    def test_repr(self):
        analyzer = TemporalMotionAnalyzer(device="cpu", sequence_length=8)
        r = repr(analyzer)
        # Fallback representation check if custom not defined, or standard class name
        assert "TemporalMotionAnalyzer" in r

    def test_analyze_empty_layer1_result(self):
        # Empty preprocessing result
        try:
            from preprocessing.result_types import VideoMetadata, VideoPreprocessingResult
        except ImportError:
            pytest.skip("preprocessing.result_types not importable")
            
        meta = VideoMetadata(
            source_path=Path("s.mp4"), duration_s=0.0, native_fps=25.0,
            width=320, height=240, total_frames_native=0, total_frames_extracted=0,
            target_fps=8.0, audio_path=None, has_audio=False, audio_sample_rate=16000,
        )
        layer1 = VideoPreprocessingResult(
            metadata=meta, frames=[], scenes=[], landmarks={}, unique_track_ids=[]
        )
        
        analyzer = TemporalMotionAnalyzer(
            device="cpu",
            timesformer_model_name="dummy-timesformer",
            videomae_model_name="dummy-videomae"
        )
        res = analyzer.analyze(layer1)
        assert isinstance(res, BranchCResult)
        assert res.n_tracks == 0
        assert len(res.processing_warnings) > 0

    def test_analyze_returns_valid_branch_c_result(self):
        layer1 = _make_layer1_result(track_ids=["track_001", "track_002"], n_frames=16)
        
        # Load analyzer with fallback weights config and dummy model names to trigger fallbacks instantly
        analyzer = TemporalMotionAnalyzer(
            device="cpu",
            timesformer_model_name="dummy-timesformer",
            videomae_model_name="dummy-videomae",
            sequence_length=16,
            timesformer_checkpoint=None,
            videomae_checkpoint=None,
            slowfast_checkpoint=None
        )
        
        res = analyzer.analyze(layer1)
        assert isinstance(res, BranchCResult)
        assert res.n_tracks == 2
        assert sorted(res.unique_track_ids) == ["track_001", "track_002"]
        
        # Check track metrics
        for tid in ["track_001", "track_002"]:
            m = res.metrics_for_track(tid)
            assert isinstance(m, TrackMotionMetrics)
            assert m.track_id == tid
            assert m.n_frames == 16
            assert m.n_valid_frames == 16
            assert 0.0 <= m.timesformer_score <= 1.0
            assert 0.0 <= m.videomae_score <= 1.0
            assert 0.0 <= m.slowfast_score <= 1.0
            assert m.motion_verdict in ["REAL", "FAKE", "UNCERTAIN"]
            assert 0.0 <= m.motion_confidence <= 1.0

        # Unload models
        analyzer.unload()
        assert analyzer._models_loaded is False
        assert analyzer.timesformer is None
