"""
Tests for VDS Layer 2, Branch D — Cross-Modal Synchronization.

Test strategy:
  - sync_signals.py: Primitives (audio energy, cross-correlation delay, mismatches)
    tested with mathematical inputs.
  - Model wrappers: Verify shape propagation in SyncNet and AV-HuBERT.
  - TemporalSyncAnalyzer: Orchestration integration using synthetic stubs and mocked audio reader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch
import numpy as np
import pytest
import torch

from vid_feature_extraction.motion_signals import compute_blink_anomalies
from vid_feature_extraction.result_types_l2d import BranchDResult, TrackSyncMetrics
from vid_feature_extraction.sync_signals import (
    compute_audio_energy,
    compute_av_delay,
    compute_emotion_mismatch,
    compute_phoneme_viseme_mismatch,
)
from vid_feature_extraction.syncnet_model import SyncNetMotionModel
from vid_feature_extraction.av_hubert_model import AVHuBERTActiveModel
from vid_feature_extraction.temporal_sync_analyzer import TemporalSyncAnalyzer


# ===========================================================================
# Helpers
# ===========================================================================

def _make_layer1_result(
    track_ids: List[str],
    n_frames: int,
    fps: float = 8.0,
    include_landmarks: bool = True,
):
    """Build a synthetic VideoPreprocessingResult stub for Branch D."""
    try:
        from vid_preprocessing.result_types import (
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
        audio_path=Path("synthetic_audio.wav"),
        has_audio=True,
        audio_sample_rate=16000,
    )

    frames = []
    for i in range(n_frames):
        tracked = []
        for tid in track_ids:
            crop = np.zeros((224, 224, 3), dtype=np.uint8)
            tracked.append(TrackedFace(
                track_id=tid,
                track_age=i,
                detection_score=0.99,
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
            track_landmarks = []
            for i in range(n_frames):
                lm_points = np.zeros((478, 3), dtype=np.float32)
                # Shift eyebrows slightly over time
                lm_points[70] += float(i) * 0.01
                lm_points[300] += float(i) * 0.01
                
                # Alternate jaw open ratio to simulate mouth opening
                jaw_val = 0.05 + 0.3 * np.sin(2 * np.pi * i / 8.0)
                
                track_landmarks.append(FaceLandmarkResult(
                    track_id=tid,
                    frame_id=i,
                    timestamp_ms=float(i * (1000.0 / fps)),
                    landmarks_478=lm_points,
                    smoothed_landmarks=lm_points,
                    jaw_open_ratio=jaw_val,
                    eye_blink_l=0.3,
                    eye_blink_r=0.3,
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
# sync_signals.py — Unit Tests
# ===========================================================================

class TestAudioEnergy:

    def test_silence_returns_zeros(self):
        signal = np.zeros(16000, dtype=np.float32)
        energy = compute_audio_energy(signal, sample_rate=16000.0, target_len=10, fps=10.0)
        assert len(energy) == 10
        assert np.all(energy == 0.0)

    def test_sine_wave_positive_energy(self):
        t = np.linspace(0, 1.0, 16000)
        signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        energy = compute_audio_energy(signal, sample_rate=16000.0, target_len=10, fps=10.0)
        assert len(energy) == 10
        assert np.all(energy > 0.0)


class TestAVDelay:

    def test_zero_delay_identical_signals(self):
        # Create matching sine timelines
        t = np.linspace(0, 1.0, 50)
        sig1 = 0.5 + 0.5 * np.sin(2 * np.pi * t)
        sig2 = sig1.copy()
        
        delay, conf = compute_av_delay(sig1, sig2, fps=10.0)
        assert delay == pytest.approx(0.0)
        assert conf > 0.4

    def test_shifted_delay_detection(self):
        t = np.linspace(0, 1.0, 50)
        sig1 = 0.5 + 0.5 * np.sin(2 * np.pi * t)
        # Shift signal 2 by 2 frames (0.2s delay at 10 fps)
        sig2 = np.roll(sig1, 2)
        
        delay, conf = compute_av_delay(sig1, sig2, fps=10.0)
        assert delay == pytest.approx(-0.2)
        assert conf > 0.3


class TestMismatchPrimitives:

    def test_perfect_phoneme_viseme_match(self):
        sig = np.array([0.1, 0.8, 0.9, 0.2])
        jaw = sig.copy()
        mismatch = compute_phoneme_viseme_mismatch(sig, jaw)
        # Perfect correlation -> r=1.0 -> mismatch=0.0
        assert mismatch == pytest.approx(0.0)

    def test_negative_phoneme_viseme_correlation_high_mismatch(self):
        sig = np.array([0.1, 0.8, 0.9, 0.2])
        jaw = 1.0 - sig  # perfectly anti-correlated
        mismatch = compute_phoneme_viseme_mismatch(sig, jaw)
        # r=-1.0 -> mismatch=1.0
        assert mismatch == pytest.approx(1.0)

    def test_emotion_mismatch_calculation(self):
        audio = np.random.normal(0, 0.1, 16000)
        # landmarks (T, 478, 3)
        landmarks = np.zeros((10, 478, 3), dtype=np.float32)
        
        mismatch = compute_emotion_mismatch(audio, landmarks, fps=10.0)
        assert 0.0 <= mismatch <= 1.0


# ===========================================================================
# Model Wrapper Unit Tests
# ===========================================================================

class TestSyncModels:

    def test_syncnet_wrapper_shapes(self):
        model = SyncNetMotionModel()
        
        # Audio spectrogram input: (B, 1, 13, 20)
        audio_in = torch.randn(2, 1, 13, 20)
        # Mouth crops input: (B, 3, 5, 112, 112)
        video_in = torch.randn(2, 3, 5, 112, 112)
        
        dist = model(audio_in, video_in)
        assert dist.shape == (2,)
        
        conf = model.predict_sync_confidence(audio_in, video_in)
        assert conf.shape == (2,)
        assert torch.all(conf >= 0.0) and torch.all(conf <= 10.0)

    def test_av_hubert_wrapper_shapes(self):
        model = AVHuBERTActiveModel()
        
        # Mouth crops: (B, T=16, 3, 112, 112)
        video_in = torch.randn(2, 16, 3, 112, 112)
        # Audio samples: (2, 32000)
        audio_in = torch.randn(2, 32000)
        
        logits = model(video_in, audio_in)
        assert logits.shape == (2, 2)
        
        probs = model.predict_probability(video_in, audio_in)
        assert probs.shape == (2,)
        assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


# ===========================================================================
# TemporalSyncAnalyzer Integration Tests
# ===========================================================================

class TestTemporalSyncAnalyzerIntegration:

    def test_repr(self):
        analyzer = TemporalSyncAnalyzer(device="cpu")
        assert "TemporalSyncAnalyzer" in repr(analyzer)

    def test_analyze_empty_layer1_result(self):
        try:
            from vid_preprocessing.result_types import VideoMetadata, VideoPreprocessingResult
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
        
        analyzer = TemporalSyncAnalyzer(device="cpu")
        res = analyzer.analyze(layer1)
        assert isinstance(res, BranchDResult)
        assert res.n_tracks == 0
        assert len(res.processing_warnings) > 0

    @patch("scipy.io.wavfile.read")
    def test_analyze_returns_valid_branch_d_result(self, mock_wav_read):
        # Mock scipy wav read to return a dummy sample rate & stereo waveform
        mock_wav_read.return_value = (16000, np.zeros((32000, 2), dtype=np.int16))
        
        layer1 = _make_layer1_result(track_ids=["track_001"], n_frames=16, fps=8.0)
        
        analyzer = TemporalSyncAnalyzer(device="cpu")
        res = analyzer.analyze(layer1)
        
        assert isinstance(res, BranchDResult)
        assert res.n_tracks == 1
        assert res.unique_track_ids == ["track_001"]
        
        m = res.metrics_for_track("track_001")
        assert isinstance(m, TrackSyncMetrics)
        assert m.track_id == "track_001"
        assert m.n_frames == 16
        assert m.n_valid_frames == 16
        assert 0.0 <= m.syncnet_confidence <= 10.0
        assert -5 <= m.syncnet_offset <= 5
        assert 0.0 <= m.av_hubert_score <= 1.0
        assert 0.0 <= m.phoneme_viseme_inconsistency <= 1.0
        assert 0.0 <= m.emotion_mismatch_score <= 1.0
        assert m.sync_verdict in ["REAL", "FAKE", "UNCERTAIN"]
        assert 0.0 <= m.sync_confidence <= 1.0

        # Unload models
        analyzer.unload()
        assert analyzer._models_loaded is False
