"""
Tests for VDS Layer 2, Branch B — Temporal Consistency Analysis.

Test strategy:
  - temporal_signals.py: All primitives tested with known mathematical inputs
    (zero-dependency, pure NumPy — always run)
  - TrackTemporalMetrics / BranchBResult: Structural dataclass tests (zero-dep)
  - TemporalConsistencyAnalyzer: Uses synthetic Branch A + Layer 1 stubs —
    no IDS model loading required
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest


# ===========================================================================
# Helpers
# ===========================================================================

def _make_branch_a_result(
    track_ids:     List[str],
    n_frames:      int,
    fake_probs:    Optional[Dict[str, List[float]]] = None,
    include_embs:  bool = True,
    include_maps:  bool = True,
):
    """
    Build a synthetic BranchAResult with controlled fake_prob timelines.
    Embeddings are random (2560,) float32 arrays; heatmaps are random (224,224).
    """
    try:
        from vid_feature_extraction.result_types_l2 import BranchAResult, FrameVisualResult  # noqa
    except ImportError:
        pytest.skip("result_types_l2 not importable")

    frame_results = {}
    for tid in track_ids:
        probs = (fake_probs or {}).get(tid, [0.3] * n_frames)
        results = []
        for i, prob in enumerate(probs):
            results.append(FrameVisualResult(
                track_id=tid,
                frame_id=i,
                timestamp_ms=float(i * 125),
                fake_probability=prob,
                ood_score=0.01,
                artifact_embedding=(
                    np.random.rand(2560).astype(np.float32)
                    if include_embs else None
                ),
                manipulation_heatmap=(
                    np.random.rand(224, 224).astype(np.float32) * 0.3
                    if include_maps else None
                ),
                detection_success=True,
            ))
        frame_results[tid] = results

    return BranchAResult(
        frame_results=frame_results,
        n_frames_processed=n_frames,
        n_faces_total=n_frames * len(track_ids),
        n_faces_successful=n_frames * len(track_ids),
    )


def _make_layer1_result(
    track_ids: List[str],
    n_frames:  int,
    fps:       float = 8.0,
    jaw_ratios: Optional[Dict[str, List[float]]] = None,
    ear_values: Optional[Dict[str, List[float]]] = None,
):
    """
    Build a synthetic VideoPreprocessingResult stub for Branch B.
    Does not require FFmpeg, DeepSORT, or MediaPipe.
    """
    try:
        from vid_preprocessing.result_types import (  # noqa
            FaceLandmarkResult, FramePacket, SceneBoundary,
            VideoMetadata, VideoPreprocessingResult,
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
    # Frames — alternating two scenes
    frames = [
        FramePacket(
            frame_id=i,
            timestamp_ms=float(i * (1000.0 / fps)),
            scene_id=(0 if i < n_frames // 2 else 1),
        )
        for i in range(n_frames)
    ]
    scenes = [
        SceneBoundary(0, 0, n_frames // 2 - 1, 0.0, float(n_frames // 2 * 1000.0 / fps), n_frames // 2),
        SceneBoundary(1, n_frames // 2, n_frames - 1, float(n_frames // 2 * 1000.0 / fps), float(n_frames * 1000.0 / fps), n_frames // 2),
    ]

    # Landmarks
    landmarks = {}
    for tid in track_ids:
        jaws = (jaw_ratios or {}).get(tid, [0.05] * n_frames)
        ears = (ear_values or {}).get(tid, [0.30] * n_frames)
        landmarks[tid] = [
            FaceLandmarkResult(
                track_id=tid,
                frame_id=i,
                timestamp_ms=float(i * (1000.0 / fps)),
                jaw_open_ratio=jaws[i] if i < len(jaws) else 0.05,
                eye_blink_l=ears[i] if i < len(ears) else 0.30,
                eye_blink_r=ears[i] if i < len(ears) else 0.30,
                detection_success=True,
            )
            for i in range(n_frames)
        ]

    return VideoPreprocessingResult(
        metadata=meta,
        frames=frames,
        scenes=scenes,
        landmarks=landmarks,
        unique_track_ids=track_ids,
    )


# ===========================================================================
# temporal_signals.py — pure-NumPy unit tests
# ===========================================================================

class TestRollingStd:

    def test_constant_signal_returns_zero(self):
        from vid_feature_extraction.temporal_signals import rolling_std  # noqa
        assert rolling_std([0.5] * 20, window=5) == pytest.approx(0.0)

    def test_alternating_signal_high(self):
        from vid_feature_extraction.temporal_signals import rolling_std  # noqa
        values = [0.0, 1.0] * 10
        result = rolling_std(values, window=4)
        assert result > 0.3

    def test_shorter_than_window_returns_zero(self):
        from vid_feature_extraction.temporal_signals import rolling_std  # noqa
        assert rolling_std([0.1, 0.2], window=5) == pytest.approx(0.0)


class TestLinearTrend:

    def test_rising_trend_positive_slope(self):
        from vid_feature_extraction.temporal_signals import linear_trend  # noqa
        values = [float(i) * 0.05 for i in range(20)]
        slope  = linear_trend(values)
        assert slope > 0.04

    def test_falling_trend_negative_slope(self):
        from vid_feature_extraction.temporal_signals import linear_trend  # noqa
        values = [1.0 - float(i) * 0.05 for i in range(20)]
        slope  = linear_trend(values)
        assert slope < -0.04

    def test_constant_slope_is_zero(self):
        from vid_feature_extraction.temporal_signals import linear_trend  # noqa
        slope = linear_trend([0.5] * 10)
        assert abs(slope) < 1e-10

    def test_single_value_returns_zero(self):
        from vid_feature_extraction.temporal_signals import linear_trend  # noqa
        assert linear_trend([0.7]) == pytest.approx(0.0)


class TestPeakCount:

    def test_no_peaks_in_constant_signal(self):
        from vid_feature_extraction.temporal_signals import peak_count  # noqa
        assert peak_count([0.3] * 20) == 0

    def test_isolated_spike_detected(self):
        from vid_feature_extraction.temporal_signals import peak_count  # noqa
        values = [0.2] * 10 + [0.85] + [0.2] * 10
        assert peak_count(values, high_threshold=0.70, low_threshold=0.50) == 1

    def test_two_spikes(self):
        from vid_feature_extraction.temporal_signals import peak_count  # noqa
        values = [0.2] * 5 + [0.85] + [0.2] * 5 + [0.90] + [0.2] * 5
        assert peak_count(values) == 2

    def test_plateau_not_counted_as_peak(self):
        from vid_feature_extraction.temporal_signals import peak_count  # noqa
        # A block of high values should not produce a peak
        values = [0.2] * 5 + [0.85] * 5 + [0.2] * 5
        # Middle of plateau: no low neighbour within min_gap=2
        result = peak_count(values, high_threshold=0.70, low_threshold=0.50, min_gap=2)
        # Only edge frames of the plateau may count (within gap to low)
        assert result <= 2


class TestCosineDriftTimeline:

    def test_identical_embeddings_zero_drift(self):
        from vid_feature_extraction.temporal_signals import cosine_drift_timeline  # noqa
        e = np.ones(512, dtype=np.float32)
        drifts = cosine_drift_timeline([e, e, e])
        assert len(drifts) == 2
        for d in drifts:
            assert d == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_embeddings_drift_one(self):
        from vid_feature_extraction.temporal_signals import cosine_drift_timeline  # noqa
        e1 = np.zeros(4, dtype=np.float32)
        e2 = np.zeros(4, dtype=np.float32)
        e1[0] = 1.0
        e2[1] = 1.0
        drifts = cosine_drift_timeline([e1, e2])
        assert len(drifts) == 1
        assert drifts[0] == pytest.approx(1.0, abs=1e-6)

    def test_opposite_embeddings_drift_two(self):
        from vid_feature_extraction.temporal_signals import cosine_drift_timeline  # noqa
        e1 = np.array([1.0, 0.0], dtype=np.float32)
        e2 = np.array([-1.0, 0.0], dtype=np.float32)
        drifts = cosine_drift_timeline([e1, e2])
        assert drifts[0] == pytest.approx(2.0, abs=1e-6)

    def test_single_embedding_returns_empty(self):
        from vid_feature_extraction.temporal_signals import cosine_drift_timeline  # noqa
        e = np.ones(512, dtype=np.float32)
        assert cosine_drift_timeline([e]) == []

    def test_zero_embedding_handled(self):
        from vid_feature_extraction.temporal_signals import cosine_drift_timeline  # noqa
        e1 = np.zeros(512, dtype=np.float32)
        e2 = np.ones(512, dtype=np.float32)
        drifts = cosine_drift_timeline([e1, e2])
        # Zero-norm embedding → distance 0.0 by convention
        assert drifts[0] == pytest.approx(0.0, abs=1e-6)

    def test_output_length_n_minus_one(self):
        from vid_feature_extraction.temporal_signals import cosine_drift_timeline  # noqa
        embs = [np.random.rand(256).astype(np.float32) for _ in range(10)]
        drifts = cosine_drift_timeline(embs)
        assert len(drifts) == 9

    def test_drift_values_in_valid_range(self):
        from vid_feature_extraction.temporal_signals import cosine_drift_timeline  # noqa
        embs = [np.random.rand(256).astype(np.float32) for _ in range(8)]
        drifts = cosine_drift_timeline(embs)
        for d in drifts:
            assert 0.0 <= d <= 2.0


class TestHeatmapFluxTimeline:

    def test_identical_heatmaps_zero_flux(self):
        from vid_feature_extraction.temporal_signals import heatmap_flux_timeline  # noqa
        h = np.zeros((224, 224), dtype=np.float32)
        flux = heatmap_flux_timeline([h, h, h])
        assert len(flux) == 2
        for f in flux:
            assert f == pytest.approx(0.0, abs=1e-8)

    def test_single_heatmap_returns_empty(self):
        from vid_feature_extraction.temporal_signals import heatmap_flux_timeline  # noqa
        h = np.zeros((224, 224), dtype=np.float32)
        assert heatmap_flux_timeline([h]) == []

    def test_high_flux_on_opposite_heatmaps(self):
        from vid_feature_extraction.temporal_signals import heatmap_flux_timeline  # noqa
        h1 = np.ones((224, 224), dtype=np.float32)
        h2 = np.zeros((224, 224), dtype=np.float32)
        flux = heatmap_flux_timeline([h1, h2])
        assert flux[0] == pytest.approx(1.0, abs=1e-6)


class TestDetectBlinkRate:

    def test_no_blinks_in_constant_open_eye(self):
        from vid_feature_extraction.temporal_signals import detect_blink_rate  # noqa
        ear = [0.35] * 80  # consistently open
        rate = detect_blink_rate(ear, ear, fps=8.0, ear_threshold=0.20)
        assert rate == pytest.approx(0.0)

    def test_single_blink_correct_rate(self):
        from vid_feature_extraction.temporal_signals import detect_blink_rate  # noqa
        # 8 fps, 2-second clip, 1 blink → 0.5 blinks/s
        ear = [0.35] * 8 + [0.10, 0.08] + [0.35] * 6   # 16 frames
        rate = detect_blink_rate(ear, ear, fps=8.0, ear_threshold=0.20)
        assert 0.4 < rate < 0.7

    def test_short_signal_returns_zero(self):
        from vid_feature_extraction.temporal_signals import detect_blink_rate  # noqa
        assert detect_blink_rate([0.1], [0.1], fps=8.0) == pytest.approx(0.0)

    def test_zero_fps_returns_zero(self):
        from vid_feature_extraction.temporal_signals import detect_blink_rate  # noqa
        ear = [0.15] * 20
        assert detect_blink_rate(ear, ear, fps=0.0) == pytest.approx(0.0)


class TestPearsonCorrelation:

    def test_perfect_positive_correlation(self):
        from vid_feature_extraction.temporal_signals import pearson_correlation  # noqa
        x = [float(i) for i in range(10)]
        y = [float(i) * 2 + 1 for i in range(10)]
        r = pearson_correlation(x, y)
        assert r == pytest.approx(1.0, abs=1e-9)

    def test_perfect_negative_correlation(self):
        from vid_feature_extraction.temporal_signals import pearson_correlation  # noqa
        x = [float(i) for i in range(10)]
        y = [-float(i) for i in range(10)]
        r = pearson_correlation(x, y)
        assert r == pytest.approx(-1.0, abs=1e-9)

    def test_constant_signal_returns_zero(self):
        from vid_feature_extraction.temporal_signals import pearson_correlation  # noqa
        x = [1.0] * 10
        y = [float(i) for i in range(10)]
        r = pearson_correlation(x, y)
        assert r == pytest.approx(0.0)

    def test_short_signal_returns_zero(self):
        from vid_feature_extraction.temporal_signals import pearson_correlation  # noqa
        assert pearson_correlation([0.5], [0.3]) == pytest.approx(0.0)


class TestSceneFakeProbMap:

    def test_two_scenes_distinct_probs(self):
        from vid_feature_extraction.temporal_signals import compute_scene_fake_prob_map  # noqa
        frame_ids  = list(range(10))
        fake_probs = [0.2] * 5 + [0.8] * 5
        scene_ids  = [0] * 5 + [1] * 5
        scene_map, consistency = compute_scene_fake_prob_map(frame_ids, fake_probs, scene_ids)
        assert scene_map[0] == pytest.approx(0.2)
        assert scene_map[1] == pytest.approx(0.8)
        assert consistency > 0.2  # High std → inconsistent across scenes

    def test_single_scene_consistency_zero(self):
        from vid_feature_extraction.temporal_signals import compute_scene_fake_prob_map  # noqa
        frame_ids  = list(range(10))
        fake_probs = [0.5] * 10
        scene_ids  = [0] * 10
        _, consistency = compute_scene_fake_prob_map(frame_ids, fake_probs, scene_ids)
        assert consistency == pytest.approx(0.0)

    def test_empty_returns_empty(self):
        from vid_feature_extraction.temporal_signals import compute_scene_fake_prob_map  # noqa
        scene_map, consistency = compute_scene_fake_prob_map([], [], [])
        assert scene_map == {}
        assert consistency == 0.0


class TestDeriveVerdict:

    def test_all_zero_signals_verdict_real(self):
        from vid_feature_extraction.temporal_signals import derive_verdict  # noqa
        verdict, conf, signals = derive_verdict(
            fake_prob_mean=0.1, fake_prob_variance=0.001,
            fake_prob_rolling_std=0.01, fake_prob_trend_slope=0.0,
            fake_prob_peak_count=0, embedding_drift_mean=0.01,
            embedding_drift_max=0.05, heatmap_flux_mean=0.001,
            heatmap_activity_mean=0.05, scene_consistency_score=0.01,
        )
        assert verdict == "REAL"
        assert conf < 0.25

    def test_all_high_signals_verdict_fake(self):
        from vid_feature_extraction.temporal_signals import derive_verdict  # noqa
        verdict, conf, signals = derive_verdict(
            fake_prob_mean=0.85,     fake_prob_variance=0.08,
            fake_prob_rolling_std=0.12, fake_prob_trend_slope=0.01,
            fake_prob_peak_count=5,  embedding_drift_mean=0.20,
            embedding_drift_max=0.40, heatmap_flux_mean=0.10,
            heatmap_activity_mean=0.50, scene_consistency_score=0.25,
            landmark_jaw_variance=0.05, jaw_heatmap_correlation=0.80,
            landmark_available=True,
        )
        assert verdict == "FAKE"
        assert conf > 0.55
        assert len(signals) > 0

    def test_signals_fired_list_non_empty_for_fake(self):
        from vid_feature_extraction.temporal_signals import derive_verdict  # noqa
        _, _, signals = derive_verdict(
            fake_prob_mean=0.9, fake_prob_variance=0.1,
            fake_prob_rolling_std=0.15, fake_prob_trend_slope=0.01,
            fake_prob_peak_count=4, embedding_drift_mean=0.25,
            embedding_drift_max=0.5, heatmap_flux_mean=0.1,
            heatmap_activity_mean=0.6, scene_consistency_score=0.3,
        )
        assert "fake_prob_mean" in signals

    def test_landmark_excluded_when_unavailable(self):
        from vid_feature_extraction.temporal_signals import derive_verdict  # noqa
        v1, c1, _ = derive_verdict(
            fake_prob_mean=0.4, fake_prob_variance=0.01,
            fake_prob_rolling_std=0.03, fake_prob_trend_slope=0.001,
            fake_prob_peak_count=0, embedding_drift_mean=0.03,
            embedding_drift_max=0.10, heatmap_flux_mean=0.01,
            heatmap_activity_mean=0.1, scene_consistency_score=0.05,
            landmark_jaw_variance=0.08, jaw_heatmap_correlation=0.9,
            landmark_available=False,  # ← excluded
        )
        v2, c2, _ = derive_verdict(
            fake_prob_mean=0.4, fake_prob_variance=0.01,
            fake_prob_rolling_std=0.03, fake_prob_trend_slope=0.001,
            fake_prob_peak_count=0, embedding_drift_mean=0.03,
            embedding_drift_max=0.10, heatmap_flux_mean=0.01,
            heatmap_activity_mean=0.1, scene_consistency_score=0.05,
            landmark_jaw_variance=0.08, jaw_heatmap_correlation=0.9,
            landmark_available=True,   # ← included
        )
        # Including high landmark signals → higher confidence
        assert c2 >= c1


# ===========================================================================
# TrackTemporalMetrics and BranchBResult — dataclass unit tests
# ===========================================================================

class TestTrackTemporalMetrics:

    def _make_metrics(self, verdict="REAL", confidence=0.2):
        try:
            from vid_feature_extraction.result_types_l2b import TrackTemporalMetrics  # noqa
        except ImportError:
            pytest.skip("result_types_l2b not importable")
        return TrackTemporalMetrics(
            track_id="track_001",
            n_frames=20,
            n_valid_frames=18,
            fake_prob_mean=0.3,
            fake_prob_variance=0.01,
            embedding_drift_timeline=[0.03, 0.04, 0.02],
            temporal_verdict=verdict,
            temporal_confidence=confidence,
            temporal_signals_fired=["fake_prob_mean"] if verdict == "FAKE" else [],
            temporal_risk_level="HIGH" if confidence > 0.7 else "LOW",
        )

    def test_to_dict_serialisable(self):
        import json  # noqa
        m = self._make_metrics()
        d = m.to_dict()
        json.dumps(d)

    def test_coverage_property(self):
        m = self._make_metrics()
        assert m.coverage == pytest.approx(18 / 20)

    def test_coverage_zero_frames(self):
        try:
            from vid_feature_extraction.result_types_l2b import TrackTemporalMetrics  # noqa
        except ImportError:
            pytest.skip("result_types_l2b not importable")
        m = TrackTemporalMetrics(track_id="t", n_frames=0, n_valid_frames=0)
        assert m.coverage == pytest.approx(0.0)

    def test_is_temporally_fake_true(self):
        m = self._make_metrics(verdict="FAKE", confidence=0.8)
        assert m.is_temporally_fake is True

    def test_is_temporally_fake_false(self):
        m = self._make_metrics(verdict="REAL", confidence=0.1)
        assert m.is_temporally_fake is False

    def test_to_dict_keys(self):
        m = self._make_metrics()
        d = m.to_dict()
        for key in [
            "track_id", "n_frames", "n_valid_frames", "coverage",
            "fake_prob_mean", "fake_prob_variance", "embedding_drift_mean",
            "temporal_verdict", "temporal_confidence", "temporal_risk_level",
            "temporal_signals_fired",
        ]:
            assert key in d, f"Missing key: {key}"


class TestBranchBResult:

    def _make_branch_b(self):
        try:
            from vid_feature_extraction.result_types_l2b import BranchBResult, TrackTemporalMetrics  # noqa
        except ImportError:
            pytest.skip("result_types_l2b not importable")

        m1 = TrackTemporalMetrics(
            track_id="track_001", n_frames=10, n_valid_frames=9,
            temporal_verdict="FAKE", temporal_confidence=0.8,
            temporal_signals_fired=["fake_prob_mean", "embedding_drift_mean"],
        )
        m2 = TrackTemporalMetrics(
            track_id="track_002", n_frames=10, n_valid_frames=10,
            temporal_verdict="REAL", temporal_confidence=0.15,
            temporal_signals_fired=[],
        )
        return BranchBResult(
            track_metrics={"track_001": m1, "track_002": m2},
            video_fake_prob_mean=0.5,
            overall_verdict="FAKE",
            overall_confidence=0.5,
            n_fake_tracks=1,
            n_real_tracks=1,
            n_uncertain_tracks=0,
        )

    def test_unique_track_ids_sorted(self):
        bb = self._make_branch_b()
        assert bb.unique_track_ids == ["track_001", "track_002"]

    def test_n_tracks(self):
        bb = self._make_branch_b()
        assert bb.n_tracks == 2

    def test_metrics_for_track(self):
        bb = self._make_branch_b()
        m  = bb.metrics_for_track("track_001")
        assert m is not None
        assert m.temporal_verdict == "FAKE"

    def test_metrics_for_missing_track(self):
        bb = self._make_branch_b()
        assert bb.metrics_for_track("track_999") is None

    def test_signals_fired(self):
        bb = self._make_branch_b()
        signals = bb.signals_fired_for_track("track_001")
        assert "fake_prob_mean" in signals

    def test_summary_serialisable(self):
        import json  # noqa
        bb = self._make_branch_b()
        json.dumps(bb.summary())

    def test_summary_contains_tracks(self):
        bb = self._make_branch_b()
        s  = bb.summary()
        assert "tracks" in s
        assert "track_001" in s["tracks"]
        assert "overall_verdict" in s


# ===========================================================================
# TemporalConsistencyAnalyzer — integration tests with synthetic data
# ===========================================================================

class TestTemporalConsistencyAnalyzer:

    def test_repr(self):
        try:
            from vid_feature_extraction.temporal_consistency_analyzer import (  # noqa
                TemporalConsistencyAnalyzer,
            )
        except ImportError:
            pytest.skip("temporal_consistency_analyzer not importable")
        tca = TemporalConsistencyAnalyzer(rolling_window=7, min_valid_frames=4)
        r   = repr(tca)
        assert "TemporalConsistencyAnalyzer" in r
        assert "7" in r
        assert "4" in r

    def test_analyze_returns_branch_b_result(self):
        try:
            from vid_feature_extraction.temporal_consistency_analyzer import (  # noqa
                TemporalConsistencyAnalyzer,
            )
            from vid_feature_extraction.result_types_l2b import BranchBResult  # noqa
        except ImportError:
            pytest.skip("modules not importable")

        branch_a = _make_branch_a_result(["track_001", "track_002"], n_frames=10)
        layer1   = _make_layer1_result(["track_001", "track_002"], n_frames=10)

        tca    = TemporalConsistencyAnalyzer()
        result = tca.analyze(branch_a, layer1)
        assert isinstance(result, BranchBResult)

    def test_analyze_has_all_tracks(self):
        try:
            from vid_feature_extraction.temporal_consistency_analyzer import TemporalConsistencyAnalyzer  # noqa
        except ImportError:
            pytest.skip("modules not importable")

        branch_a = _make_branch_a_result(["track_001", "track_002", "track_003"], n_frames=8)
        layer1   = _make_layer1_result(["track_001", "track_002", "track_003"], n_frames=8)
        result   = TemporalConsistencyAnalyzer().analyze(branch_a, layer1)

        assert result.n_tracks == 3
        for tid in ["track_001", "track_002", "track_003"]:
            assert result.metrics_for_track(tid) is not None

    def test_high_fake_prob_yields_fake_verdict(self):
        """A track with consistently high fake_probability should be FAKE."""
        try:
            from vid_feature_extraction.temporal_consistency_analyzer import TemporalConsistencyAnalyzer  # noqa
        except ImportError:
            pytest.skip("modules not importable")

        # Consistently high fake_prob (> 0.55 threshold)
        probs    = {"track_001": [0.90] * 20}
        branch_a = _make_branch_a_result(["track_001"], n_frames=20, fake_probs=probs)
        layer1   = _make_layer1_result(["track_001"], n_frames=20)

        result = TemporalConsistencyAnalyzer().analyze(branch_a, layer1)
        m      = result.metrics_for_track("track_001")
        assert m.fake_prob_mean > 0.55
        assert m.temporal_verdict == "FAKE"

    def test_low_fake_prob_yields_real_verdict(self):
        """A track with consistently low fake_probability should be REAL."""
        try:
            from vid_feature_extraction.temporal_consistency_analyzer import TemporalConsistencyAnalyzer  # noqa
        except ImportError:
            pytest.skip("modules not importable")

        probs    = {"track_001": [0.05] * 20}
        branch_a = _make_branch_a_result(["track_001"], n_frames=20, fake_probs=probs)
        layer1   = _make_layer1_result(["track_001"], n_frames=20)

        result = TemporalConsistencyAnalyzer().analyze(branch_a, layer1)
        m      = result.metrics_for_track("track_001")
        assert m.temporal_verdict == "REAL"

    def test_flickering_signal_fires(self):
        """High fake_prob_variance should fire the variance signal."""
        try:
            from vid_feature_extraction.temporal_consistency_analyzer import TemporalConsistencyAnalyzer  # noqa
        except ImportError:
            pytest.skip("modules not importable")

        # Alternating high/low fake probs → high variance
        probs    = {"track_001": [0.1, 0.9] * 10}
        branch_a = _make_branch_a_result(["track_001"], n_frames=20, fake_probs=probs)
        layer1   = _make_layer1_result(["track_001"], n_frames=20)

        result = TemporalConsistencyAnalyzer().analyze(branch_a, layer1)
        m      = result.metrics_for_track("track_001")
        assert m.fake_prob_variance > 0.04
        assert "fake_prob_variance" in m.temporal_signals_fired

    def test_scene_consistency_computed(self):
        """Scene-split fake_probs should produce non-zero scene consistency."""
        try:
            from vid_feature_extraction.temporal_consistency_analyzer import TemporalConsistencyAnalyzer  # noqa
        except ImportError:
            pytest.skip("modules not importable")

        # First half of video = REAL, second half = FAKE
        probs    = {"track_001": [0.1] * 10 + [0.9] * 10}
        branch_a = _make_branch_a_result(["track_001"], n_frames=20, fake_probs=probs)
        layer1   = _make_layer1_result(["track_001"], n_frames=20)

        result = TemporalConsistencyAnalyzer().analyze(branch_a, layer1)
        m      = result.metrics_for_track("track_001")
        assert m.scene_consistency_score > 0.0
        assert len(m.scene_fake_prob_map) == 2

    def test_insufficient_frames_returns_uncertain(self):
        """Track with < min_valid_frames valid results → UNCERTAIN."""
        try:
            from vid_feature_extraction.temporal_consistency_analyzer import TemporalConsistencyAnalyzer  # noqa
        except ImportError:
            pytest.skip("modules not importable")

        probs    = {"track_001": [0.8, 0.9]}  # Only 2 frames
        branch_a = _make_branch_a_result(["track_001"], n_frames=2, fake_probs=probs)
        layer1   = _make_layer1_result(["track_001"], n_frames=2)

        tca    = TemporalConsistencyAnalyzer(min_valid_frames=5)
        result = tca.analyze(branch_a, layer1)
        m      = result.metrics_for_track("track_001")
        assert m.temporal_verdict == "UNCERTAIN"

    def test_overall_verdict_majority_fake(self):
        """With 2 FAKE tracks and 1 REAL, overall verdict should be FAKE."""
        try:
            from vid_feature_extraction.temporal_consistency_analyzer import TemporalConsistencyAnalyzer  # noqa
        except ImportError:
            pytest.skip("modules not importable")

        probs = {
            "track_001": [0.9] * 20,
            "track_002": [0.85] * 20,
            "track_003": [0.05] * 20,
        }
        branch_a = _make_branch_a_result(
            ["track_001", "track_002", "track_003"], n_frames=20, fake_probs=probs
        )
        layer1 = _make_layer1_result(
            ["track_001", "track_002", "track_003"], n_frames=20
        )
        result = TemporalConsistencyAnalyzer().analyze(branch_a, layer1)
        assert result.overall_verdict == "FAKE"

    def test_embedding_drift_computed(self):
        """Branch B must compute embedding drift timeline."""
        try:
            from vid_feature_extraction.temporal_consistency_analyzer import TemporalConsistencyAnalyzer  # noqa
        except ImportError:
            pytest.skip("modules not importable")

        branch_a = _make_branch_a_result(["track_001"], n_frames=10, include_embs=True)
        layer1   = _make_layer1_result(["track_001"], n_frames=10)
        result   = TemporalConsistencyAnalyzer().analyze(branch_a, layer1)
        m        = result.metrics_for_track("track_001")
        # With 10 valid frames and 10 embeddings → 9 drift values
        assert len(m.embedding_drift_timeline) == 9

    def test_landmarks_influence_verdict(self):
        """Landmark availability flag must be set based on Layer 1 data."""
        try:
            from vid_feature_extraction.temporal_consistency_analyzer import TemporalConsistencyAnalyzer  # noqa
        except ImportError:
            pytest.skip("modules not importable")

        branch_a = _make_branch_a_result(["track_001"], n_frames=15)
        layer1   = _make_layer1_result(["track_001"], n_frames=15, jaw_ratios={"track_001": [0.05]*15})
        result   = TemporalConsistencyAnalyzer().analyze(branch_a, layer1)
        m        = result.metrics_for_track("track_001")
        assert m.landmark_available is True

    def test_single_track_analyze_track(self):
        """analyze_track() shortcut must return a TrackTemporalMetrics."""
        try:
            from vid_feature_extraction.temporal_consistency_analyzer import TemporalConsistencyAnalyzer  # noqa
            from vid_feature_extraction.result_types_l2b import TrackTemporalMetrics  # noqa
        except ImportError:
            pytest.skip("modules not importable")

        branch_a = _make_branch_a_result(["track_001"], n_frames=12)
        layer1   = _make_layer1_result(["track_001"], n_frames=12)
        tca      = TemporalConsistencyAnalyzer()
        m        = tca.analyze_track("track_001", branch_a, layer1)
        assert isinstance(m, TrackTemporalMetrics)
        assert m.track_id == "track_001"

    def test_summary_json_serialisable(self):
        """BranchBResult.summary() must be JSON-serialisable."""
        import json  # noqa
        try:
            from vid_feature_extraction.temporal_consistency_analyzer import TemporalConsistencyAnalyzer  # noqa
        except ImportError:
            pytest.skip("modules not importable")

        branch_a = _make_branch_a_result(["track_001"], n_frames=8)
        layer1   = _make_layer1_result(["track_001"], n_frames=8)
        result   = TemporalConsistencyAnalyzer().analyze(branch_a, layer1)
        json.dumps(result.summary())
