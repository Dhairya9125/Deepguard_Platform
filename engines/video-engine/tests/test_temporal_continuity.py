"""
Tests for VDS Layer 2, Branch F — Identity Continuity Tracking.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import numpy as np
import pytest
import torch

from feature_extraction.result_types_l2f import BranchFResult, TrackContinuityMetrics
from feature_extraction.continuity_signals import (
    compute_geometry_instability,
    compute_identity_drift,
    compute_trajectory_jumps,
)
from feature_extraction.facenet_model import FacenetActiveModel
from feature_extraction.temporal_continuity_analyzer import TemporalContinuityAnalyzer


# ===========================================================================
# Helpers
# ===========================================================================

def _make_dummy_landmarks(n_frames: int, instability: bool = False):
    """Generate mock landmarks for rigid skull geometry ratios."""
    try:
        from preprocessing.result_types import FaceLandmarkResult
    except ImportError:
        class FaceLandmarkResult:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

    lm_list = []
    for i in range(n_frames):
        lm_points = np.zeros((478, 3), dtype=np.float32)
        
        # Rigids coordinates:
        # Iris right: 468, iris left: 473
        # Face width outer right: 234, left: 454
        # Nose tip: 1, chin mid: 152, forehead: 10
        
        # Basic distances: eye_dist = 40, face_width = 100
        # If instability is active, fluctuate them over time
        jitter = 5.0 * np.sin(2 * np.pi * i / 5.0) if instability else 0.0
        
        lm_points[468] = [80.0, 100.0, 0.0]
        lm_points[473] = [120.0 + jitter, 100.0, 0.0]  # eye distance varies if unstable
        lm_points[234] = [50.0, 100.0, 0.0]
        lm_points[454] = [150.0, 100.0, 0.0]
        
        # height: forehead-to-chin = 120, nose-to-chin = 50
        lm_points[10] = [100.0, 40.0, 0.0]
        lm_points[152] = [100.0, 160.0 + jitter, 0.0]  # chin varies if unstable
        lm_points[1] = [100.0, 110.0, 0.0]

        lm_list.append(FaceLandmarkResult(
            track_id="track_001",
            frame_id=i,
            timestamp_ms=float(i * 1000.0 / 30.0),
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
    instability: bool = False,
    swap_face_at: int | None = None
):
    """Build a synthetic VideoPreprocessingResult stub for Branch F."""
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
            crop = np.zeros((224, 224, 3), dtype=np.uint8)
            # Simulating identity swap by changing crop color features mid-way
            if swap_face_at is not None and i >= swap_face_at:
                crop[:, :, 0] = 200  # distinct face representation color signature
                crop[:, :, 1] = 50
            else:
                crop[:, :, 0] = 100
                crop[:, :, 1] = 100
            
            # Simple trajectory: slow movement
            # If i = 15, inject a non-physical jump
            offset_x = 0
            if swap_face_at is not None and i == swap_face_at:
                offset_x = 80  # trajectory jump
                
            x1 = 10 + i + offset_x
            y1 = 10 + i
            bbox = (x1, y1, x1 + 100, y1 + 100)
            
            tracked.append(TrackedFace(
                track_id=tid,
                track_age=i,
                detection_score=0.99,
                bbox=bbox,
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
        landmarks[tid] = _make_dummy_landmarks(n_frames, instability)

    return VideoPreprocessingResult(
        metadata=meta,
        frames=frames,
        scenes=scenes,
        landmarks=landmarks,
        unique_track_ids=track_ids,
    )


# ===========================================================================
# continuity_signals.py — Unit Tests
# ===========================================================================

class TestContinuitySignals:

    def test_compute_identity_drift(self):
        # 1. Constant embeddings -> zero drift
        embs_constant = np.ones((5, 128))
        drift_max, drift_var = compute_identity_drift(embs_constant)
        assert drift_max == pytest.approx(0.0, abs=1e-5)
        assert drift_var == pytest.approx(0.0, abs=1e-5)

        # 2. Distinct embeddings -> positive drift
        embs_distinct = np.random.normal(0, 1, (10, 128))
        drift_max, drift_var = compute_identity_drift(embs_distinct)
        assert drift_max > 0.0
        assert drift_var > 0.0

    def test_compute_geometry_instability(self):
        # 1. Rigid coordinates -> zero instability
        landmarks_rigid = np.array([lm.smoothed_landmarks for lm in _make_dummy_landmarks(5, instability=False)])
        instability_low = compute_geometry_instability(landmarks_rigid)
        assert instability_low == pytest.approx(0.0, abs=1e-5)

        # 2. Fluctuating coordinates -> high instability
        landmarks_wobbly = np.array([lm.smoothed_landmarks for lm in _make_dummy_landmarks(5, instability=True)])
        instability_high = compute_geometry_instability(landmarks_wobbly)
        assert instability_high > 0.01

    def test_compute_trajectory_jumps(self):
        # 1. Continuous movement -> zero jumps
        bboxes_smooth = [(i, 10, i + 100, 110) for i in range(10)]
        max_jump, jump_count = compute_trajectory_jumps(bboxes_smooth, fps=30.0, frame_width=320)
        assert max_jump == pytest.approx(1.0)
        assert jump_count == 0

        # 2. Sudden jump -> flags jump count
        bboxes_jumpy = bboxes_smooth.copy()
        bboxes_jumpy[5] = (200, 10, 300, 110)  # sudden leap
        max_jump, jump_count = compute_trajectory_jumps(bboxes_jumpy, fps=30.0, frame_width=320)
        assert max_jump > 50.0
        assert jump_count > 0


# ===========================================================================
# Model Wrapper Unit Tests
# ===========================================================================

class TestContinuityModels:

    def test_facenet_model_shape(self):
        model = FacenetActiveModel()
        
        # Input shape: (B, C=3, H=160, W=160)
        inputs = torch.randn(2, 3, 160, 160)
        embeddings = model(inputs)
        
        assert embeddings.shape == (2, 128)
        # Verify L2 normalization
        norms = torch.linalg.norm(embeddings, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms))

    def test_facenet_model_deterministic_fallback(self):
        model = FacenetActiveModel()  # has_weights = False
        
        # Test that two identical face crops yield identical representation embeddings
        crop1 = torch.zeros((1, 3, 160, 160))
        crop1[:, 0, :, :] = 1.0  # solid red channel signature
        
        emb1 = model(crop1)
        emb2 = model(crop1.clone())
        
        assert torch.allclose(emb1, emb2)


# ===========================================================================
# TemporalContinuityAnalyzer Integration Tests
# ===========================================================================

class TestTemporalContinuityAnalyzerIntegration:

    def test_repr(self):
        analyzer = TemporalContinuityAnalyzer(device="cpu")
        assert "TemporalContinuityAnalyzer" in repr(analyzer)

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
        
        analyzer = TemporalContinuityAnalyzer(device="cpu")
        res = analyzer.analyze(layer1)
        assert isinstance(res, BranchFResult)
        assert res.n_tracks == 0
        assert len(res.processing_warnings) > 0

    def test_analyze_integration(self):
        # 1. Normal consistent track -> REAL/UNCERTAIN with zero jump count
        layer1_real = _make_layer1_result(track_ids=["track_001"], n_frames=30, fps=30.0, instability=False)
        analyzer = TemporalContinuityAnalyzer(device="cpu")
        res_real = analyzer.analyze(layer1_real)
        
        assert res_real.n_tracks == 1
        m_real = res_real.metrics_for_track("track_001")
        assert m_real.trajectory_jump_count == 0
        assert m_real.geometry_instability_cv == pytest.approx(0.0, abs=1e-4)

        # 2. Track with face swap & trajectory jump mid-way -> FAKE verdict
        layer1_fake = _make_layer1_result(
            track_ids=["track_001"], n_frames=30, fps=30.0, instability=True, swap_face_at=15
        )
        res_fake = analyzer.analyze(layer1_fake)
        m_fake = res_fake.metrics_for_track("track_001")
        
        assert m_fake.facenet_drift_max > 0.05
        assert m_fake.geometry_instability_cv > 0.01
        assert m_fake.trajectory_jump_count > 0
        assert m_fake.continuity_verdict in ["FAKE", "UNCERTAIN"]
        
        analyzer.unload()
        assert analyzer._models_loaded is False
