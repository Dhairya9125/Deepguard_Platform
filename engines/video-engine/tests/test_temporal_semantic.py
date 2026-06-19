"""
Tests for VDS Layer 2, Branch G — Scene Semantic Consistency.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import numpy as np
import pytest
import torch

from vid_feature_extraction.result_types_l2g import BranchGResult, SceneSemanticMetrics
from vid_feature_extraction.semantic_signals import (
    compute_background_inconsistency,
    compute_environment_instability,
    compute_lighting_discrepancy,
)
from vid_feature_extraction.world_model_net import WorldModelActiveClassifier
from vid_feature_extraction.temporal_semantic_analyzer import TemporalSemanticAnalyzer


# ===========================================================================
# Helpers
# ===========================================================================

def _make_layer1_result(
    n_frames: int,
    fps: float = 30.0,
    flicker: bool = False,
    warp: bool = False,
    lighting_shift: bool = False
):
    """Build a synthetic VideoPreprocessingResult stub for Branch G."""
    try:
        from vid_preprocessing.result_types import (
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
        # Generate full frame
        frame = np.ones((240, 320, 3), dtype=np.float32) * 128.0
        
        # Inject global flicker
        if flicker:
            frame += 10.0 * np.sin(2 * np.pi * i / 3.0)
            
        # Inject background warp: change pixels outside face bbox (e.g. at bottom right)
        if warp and i >= n_frames // 2:
            frame[200:, 200:, :] = 200.0
            
        frame = np.clip(frame, 0, 255).astype(np.uint8)
            
        # Face bbox
        x1, y1, x2, y2 = 10, 10, 110, 110
        crop = np.ones((224, 224, 3), dtype=np.uint8) * 150
        
        # Inject lighting shift on face only
        if lighting_shift and i >= n_frames // 2:
            crop += 50

        tracked = [TrackedFace(
            track_id="track_001",
            track_age=i,
            detection_score=0.99,
            bbox=(x1, y1, x2, y2),
            frame_id=i,
            timestamp_ms=float(i * (1000.0 / fps)),
            aligned_crop=crop,
        )]

        fp = FramePacket(
            frame_id=i,
            timestamp_ms=float(i * (1000.0 / fps)),
            rgb_array=frame,  # store in-memory full frame
            tracked_faces=tracked,
            scene_id=0,
        )
        frames.append(fp)

    scenes = [SceneBoundary(0, 0, n_frames - 1, 0.0, float(n_frames * 1000.0 / fps), n_frames)]

    # Mock landmarks (not heavily used in Branch G except as stubs)
    landmarks = {"track_001": []}

    return VideoPreprocessingResult(
        metadata=meta,
        frames=frames,
        scenes=scenes,
        landmarks=landmarks,
        unique_track_ids=["track_001"],
    )


# ===========================================================================
# semantic_signals.py — Unit Tests
# ===========================================================================

class TestSemanticSignals:

    def test_compute_lighting_discrepancy(self):
        T = 10
        # 1. Consistent lighting -> zero variance
        face_crops = [np.ones((224, 224, 3), dtype=np.uint8) * 150 for _ in range(T)]
        full_frames = [np.ones((240, 320, 3), dtype=np.uint8) * 120 for _ in range(T)]
        bboxes = [(10, 10, 110, 110) for _ in range(T)]
        
        var_consistent = compute_lighting_discrepancy(face_crops, full_frames, bboxes)
        assert var_consistent == pytest.approx(0.0, abs=1e-5)

        # 2. Shifted lighting on face only -> positive variance
        face_crops_shifted = face_crops.copy()
        face_crops_shifted[5] = np.ones((224, 224, 3), dtype=np.uint8) * 200
        var_discrepant = compute_lighting_discrepancy(face_crops_shifted, full_frames, bboxes)
        assert var_discrepant > 0.0

    def test_compute_background_inconsistency(self):
        T = 5
        # 1. Identical background -> zero MSE
        frames_steady = [np.ones((240, 320, 3), dtype=np.uint8) * 128 for _ in range(T)]
        bboxes = [(10, 10, 110, 110) for _ in range(T)]
        
        mse_steady = compute_background_inconsistency(frames_steady, bboxes)
        assert mse_steady == pytest.approx(0.0, abs=1e-5)

        # 2. Changing background -> positive MSE
        frames_jumpy = frames_steady.copy()
        # Change background region (e.g. lower-right corner)
        frames_jumpy[3] = frames_steady[3].copy()
        frames_jumpy[3][200:, 200:, :] = 200
        mse_jumpy = compute_background_inconsistency(frames_jumpy, bboxes)
        assert mse_jumpy > 0.0

    def test_compute_environment_instability(self):
        # 1. Steady intensities -> zero instability
        frames_steady = [np.ones((240, 320, 3), dtype=np.uint8) * 128 for _ in range(5)]
        inst_steady = compute_environment_instability(frames_steady)
        assert inst_steady == pytest.approx(0.0, abs=1e-5)

        # 2. Flickering intensity -> positive instability
        frames_flickering = [np.ones((240, 320, 3), dtype=np.uint8) * (128 + (i % 2) * 10) for i in range(5)]
        inst_flicker = compute_environment_instability(frames_flickering)
        assert inst_flicker > 0.0


# ===========================================================================
# Model Wrapper Unit Tests
# ===========================================================================

class TestWorldModelNet:

    def test_world_model_shape(self):
        model = WorldModelActiveClassifier()
        
        # Input shape: (B, T=16, C=3, H=112, W=112)
        inputs = torch.randn(2, 16, 3, 112, 112)
        realism = model(inputs)
        
        assert realism.shape == (2,)
        
        probs = model.predict_probability(inputs)
        assert probs.shape == (2,)
        assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


# ===========================================================================
# TemporalSemanticAnalyzer Integration Tests
# ===========================================================================

class TestTemporalSemanticAnalyzerIntegration:

    def test_repr(self):
        analyzer = TemporalSemanticAnalyzer(device="cpu")
        assert "TemporalSemanticAnalyzer" in repr(analyzer)

    def test_analyze_empty_layer1_result(self):
        try:
            from vid_preprocessing.result_types import VideoMetadata, VideoPreprocessingResult
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
        
        analyzer = TemporalSemanticAnalyzer(device="cpu")
        res = analyzer.analyze(layer1)
        assert isinstance(res, BranchGResult)
        assert res.n_scenes == 0
        assert len(res.processing_warnings) > 0

    def test_analyze_integration(self):
        # 1. Consistent video -> REAL overall verdict
        layer1_real = _make_layer1_result(n_frames=10, fps=30.0, flicker=False, warp=False, lighting_shift=False)
        analyzer = TemporalSemanticAnalyzer(device="cpu")
        res_real = analyzer.analyze(layer1_real)
        
        assert res_real.n_scenes == 1
        m_real = res_real.metrics_for_scene(0)
        assert m_real.lighting_discrepancy_var == pytest.approx(0.0, abs=1e-5)
        assert m_real.background_inconsistency_score == pytest.approx(0.0, abs=1e-5)
        assert m_real.environment_instability_score == pytest.approx(0.0, abs=1e-5)
        assert res_real.overall_verdict == "REAL"

        # 2. Decoupled lighting / warping video -> FAKE/UNCERTAIN overall verdict
        layer1_fake = _make_layer1_result(n_frames=10, fps=30.0, flicker=True, warp=True, lighting_shift=True)
        res_fake = analyzer.analyze(layer1_fake)
        m_fake = res_fake.metrics_for_scene(0)
        
        assert m_fake.lighting_discrepancy_var > 0.001
        assert m_fake.background_inconsistency_score > 0.0
        assert m_fake.environment_instability_score > 0.0
        assert m_fake.semantic_verdict in ["FAKE", "UNCERTAIN"]
        
        analyzer.unload()
        assert analyzer._models_loaded is False
