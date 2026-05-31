"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.temporal_consistency_analyzer
Layer   : Layer 2 — Temporal Feature Extraction
Branch  : Branch B — Temporal Consistency Analysis

Architecture Reference (DeepGuard_Platform_Architecture.docx — VDS Layer 2, Branch B):
  Takes Branch A output (FrameVisualResult timelines) and Layer 1 output
  (FaceLandmarkResult timelines) and computes temporal consistency signals
  that only emerge when examining face evidence across multiple frames.

Why temporal analysis matters for deepfake detection:
──────────────────────────────────────────────────────────────────────
  A single-frame classifier (IDS) can be fooled by high-quality generators.
  However, maintaining temporal consistency is computationally expensive for
  generators — they must ensure that:
    1. The latent representation is stable frame-to-frame (embedding drift)
    2. The fake probability assigned by the discriminator is consistent (flickering)
    3. The spatial manipulation regions don't shift between frames (heatmap flux)
    4. Facial motion (jaw, blink) correlates properly with the synthetic face
    5. The face looks equally fake in all scenes (scene consistency)

  Branch B exploits ALL of these failure modes simultaneously.

Inputs (consumed, not modified):
    branch_a     : BranchAResult  — per-track FrameVisualResult timelines.
    layer1       : VideoPreprocessingResult — per-track FaceLandmarkResult timelines
                   and per-frame scene_id labels.

Output:
    BranchBResult  — TrackTemporalMetrics per track + video-level verdict.

Design decisions:
  1. Pure-NumPy signal processing:
       All temporal statistics in `temporal_signals.py` are NumPy-only.
       No PyTorch is required for Branch B. This allows Branch B to run
       quickly on CPU while the GPU is used by Branch A.

  2. Graceful landmark missing:
       If Layer 1 landmark data is absent for a track (detection failed),
       landmark-based signals are excluded from verdict scoring rather than
       returning a hard failure.

  3. Vectorised embedding drift:
       Consecutive cosine distances are computed with a single NumPy
       matrix multiplication rather than a Python loop (O(n) instead of O(n²)).

  4. Configurable thresholds via constructor:
       All signal thresholds and weights are overridable at init time for
       deployment tuning without code changes.

Usage:
    from feature_extraction.temporal_consistency_analyzer import TemporalConsistencyAnalyzer
    from feature_extraction.frame_visual_analyzer import FrameVisualAnalyzer
    from preprocessing.video_layer1_pipeline import VideoLayer1Pipeline

    with VideoLayer1Pipeline() as pipe:
        layer1 = pipe.run("video.mp4", "tmp/")

    analyzer_a  = FrameVisualAnalyzer()
    branch_a    = analyzer_a.analyze(layer1)

    analyzer_b  = TemporalConsistencyAnalyzer()
    branch_b    = analyzer_b.analyze(branch_a, layer1)

    print(branch_b.overall_verdict)    # 'FAKE' / 'REAL' / 'UNCERTAIN'
    for tid in branch_b.unique_track_ids:
        m = branch_b.metrics_for_track(tid)
        print(f"{tid}: {m.temporal_verdict} (conf={m.temporal_confidence:.3f})")
        print(f"  Signals fired: {m.temporal_signals_fired}")
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .result_types_l2  import BranchAResult, FrameVisualResult
from .result_types_l2b import BranchBResult, TrackTemporalMetrics
from .temporal_signals import (
    cosine_drift_timeline,
    compute_scene_fake_prob_map,
    detect_blink_rate,
    derive_verdict,
    heatmap_flux_timeline,
    linear_trend,
    peak_count,
    pearson_correlation,
    risk_level,
    rolling_std,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TemporalConsistencyAnalyzer
# ---------------------------------------------------------------------------

class TemporalConsistencyAnalyzer:
    """
    VDS Layer 2, Branch B — Temporal Consistency Analysis.

    Analyses each face track across all frames it appears in, computing
    a suite of temporal signals and deriving a per-track and video-level
    temporal deepfake verdict.

    Args:
        rolling_window (int):
            Frame window for rolling std calculation. Default: 5.
        peak_high_threshold (float):
            fake_prob level above which a frame is a candidate peak. Default: 0.70.
        peak_low_threshold (float):
            fake_prob level below which the surrounding context is a valley. Default: 0.50.
        blink_ear_threshold (float):
            EAR value below which an eye is considered 'closed'. Default: 0.20.
        verdict_fake_threshold (float):
            Minimum confidence to declare a track FAKE. Default: 0.55.
        verdict_real_threshold (float):
            Maximum confidence to declare a track REAL. Default: 0.25.
        min_valid_frames (int):
            Minimum valid frames required for temporal analysis. Tracks with
            fewer frames get verdict='UNCERTAIN'. Default: 3.
    """

    def __init__(
        self,
        rolling_window:         int   = 5,
        peak_high_threshold:    float = 0.70,
        peak_low_threshold:     float = 0.50,
        blink_ear_threshold:    float = 0.20,
        verdict_fake_threshold: float = 0.55,
        verdict_real_threshold: float = 0.25,
        min_valid_frames:       int   = 3,
    ) -> None:
        self.rolling_window         = rolling_window
        self.peak_high_threshold    = peak_high_threshold
        self.peak_low_threshold     = peak_low_threshold
        self.blink_ear_threshold    = blink_ear_threshold
        self.verdict_fake_threshold = verdict_fake_threshold
        self.verdict_real_threshold = verdict_real_threshold
        self.min_valid_frames       = min_valid_frames

        logger.info(
            "TemporalConsistencyAnalyzer init | window=%d | min_valid=%d",
            rolling_window, min_valid_frames,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        branch_a: BranchAResult,
        layer1,                    # VideoPreprocessingResult — avoids circular import
    ) -> BranchBResult:
        """
        Run Branch B on Branch A output and Layer 1 result.

        Args:
            branch_a : BranchAResult from FrameVisualAnalyzer.
            layer1   : VideoPreprocessingResult from VideoLayer1Pipeline.

        Returns:
            BranchBResult — per-track TrackTemporalMetrics + video-level verdict.
        """
        logger.info(
            "═══════════════════════════════════════════════════════════════"
        )
        logger.info(
            "VDS Layer 2 Branch B START | tracks=%s | fps=%.1f",
            branch_a.unique_track_ids,
            layer1.metadata.target_fps,
        )
        logger.info(
            "═══════════════════════════════════════════════════════════════"
        )
        t_start  = time.perf_counter()
        warnings: List[str] = []

        # Pre-build frame→scene_id lookup from Layer 1
        frame_scene_map: Dict[int, int] = {
            fp.frame_id: fp.scene_id for fp in layer1.frames
        }

        # Process each track independently
        track_metrics: Dict[str, TrackTemporalMetrics] = {}
        fps = max(layer1.metadata.target_fps, 1.0)

        for track_id in branch_a.unique_track_ids:
            try:
                metrics = self._analyze_track(
                    track_id=track_id,
                    branch_a=branch_a,
                    layer1=layer1,
                    frame_scene_map=frame_scene_map,
                    fps=fps,
                )
                track_metrics[track_id] = metrics
            except Exception as exc:
                logger.warning(
                    "Branch B failed for track=%s: %s — returning UNCERTAIN.", track_id, exc
                )
                warnings.append(f"Branch B failure for {track_id}: {exc}")
                track_metrics[track_id] = TrackTemporalMetrics(
                    track_id=track_id,
                    temporal_verdict="UNCERTAIN",
                    temporal_confidence=0.0,
                    temporal_signals_fired=[],
                )

        # ── Video-level aggregates ─────────────────────────────────────────
        n_fake      = sum(1 for m in track_metrics.values() if m.temporal_verdict == "FAKE")
        n_real      = sum(1 for m in track_metrics.values() if m.temporal_verdict == "REAL")
        n_uncertain = sum(1 for m in track_metrics.values() if m.temporal_verdict == "UNCERTAIN")

        all_probs = [
            prob
            for tid in branch_a.unique_track_ids
            for prob in branch_a.fake_probability_timeline(tid)
        ]
        video_fake_prob_mean = float(np.mean(all_probs)) if all_probs else 0.0

        all_drifts = [
            d
            for m in track_metrics.values()
            for d in m.embedding_drift_timeline
        ]
        video_drift_mean = float(np.mean(all_drifts)) if all_drifts else 0.0

        all_confs = [m.temporal_confidence for m in track_metrics.values()]
        overall_confidence = float(np.mean(all_confs)) if all_confs else 0.0

        # Overall verdict: majority vote
        if n_fake > n_real and n_fake > n_uncertain:
            overall_verdict = "FAKE"
        elif n_real > n_fake and n_real > n_uncertain:
            overall_verdict = "REAL"
        elif n_fake > 0 and n_fake == n_real:
            # Tie between FAKE and REAL → defer to probability
            overall_verdict = "FAKE" if video_fake_prob_mean > 0.5 else "REAL"
        else:
            overall_verdict = "UNCERTAIN"

        elapsed = time.perf_counter() - t_start

        logger.info(
            "═══════════════════════════════════════════════════════════════"
        )
        logger.info(
            "VDS Layer 2 Branch B COMPLETE | elapsed=%.3fs | "
            "verdict=%s | confidence=%.3f | "
            "FAKE=%d | REAL=%d | UNCERTAIN=%d",
            elapsed, overall_verdict, overall_confidence,
            n_fake, n_real, n_uncertain,
        )
        logger.info(
            "═══════════════════════════════════════════════════════════════"
        )

        return BranchBResult(
            track_metrics=track_metrics,
            video_fake_prob_mean=video_fake_prob_mean,
            video_drift_mean=video_drift_mean,
            n_fake_tracks=n_fake,
            n_real_tracks=n_real,
            n_uncertain_tracks=n_uncertain,
            overall_verdict=overall_verdict,
            overall_confidence=overall_confidence,
            processing_warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Per-track analysis
    # ------------------------------------------------------------------

    def _analyze_track(
        self,
        track_id:        str,
        branch_a:        BranchAResult,
        layer1,
        frame_scene_map: Dict[int, int],
        fps:             float,
    ) -> TrackTemporalMetrics:
        """
        Compute all temporal metrics for a single face track.
        """
        # ── Pull Branch A data for this track ─────────────────────────────
        a_results: List[FrameVisualResult] = branch_a.results_for_track(track_id)
        valid_a   = [r for r in a_results if r.detection_success]
        n_frames  = len(a_results)
        n_valid   = len(valid_a)

        if n_valid < self.min_valid_frames:
            logger.debug(
                "track=%s: only %d valid frames (< %d) — returning UNCERTAIN.",
                track_id, n_valid, self.min_valid_frames,
            )
            return TrackTemporalMetrics(
                track_id=track_id,
                n_frames=n_frames,
                n_valid_frames=n_valid,
                temporal_verdict="UNCERTAIN",
                temporal_confidence=0.0,
                temporal_signals_fired=["insufficient_frames"],
            )

        # Timelines (ordered by frame_id — already sorted by Branch A)
        fake_probs:    List[float]        = [r.fake_probability for r in valid_a]
        ood_scores:    List[float]        = [r.ood_score for r in valid_a]
        frame_ids:     List[int]          = [r.frame_id for r in valid_a]
        timestamps_ms: List[float]        = [r.timestamp_ms for r in valid_a]
        scene_ids:     List[int]          = [frame_scene_map.get(fid, 0) for fid in frame_ids]

        embeddings:   List[np.ndarray]    = [
            r.artifact_embedding for r in valid_a
            if r.artifact_embedding is not None
        ]
        heatmaps:     List[np.ndarray]    = [
            r.manipulation_heatmap for r in valid_a
            if r.manipulation_heatmap is not None
        ]

        # ── 1. IDS fake_prob statistics ───────────────────────────────────
        arr               = np.asarray(fake_probs, dtype=np.float64)
        prob_mean         = float(arr.mean())
        prob_var          = float(arr.var())
        prob_std          = float(arr.std())
        prob_max          = float(arr.max())
        prob_min          = float(arr.min())
        prob_trend        = linear_trend(fake_probs)
        prob_rolling_std  = rolling_std(fake_probs, window=self.rolling_window)
        prob_peaks        = peak_count(
            fake_probs,
            high_threshold=self.peak_high_threshold,
            low_threshold=self.peak_low_threshold,
        )

        # ── 2. Embedding drift ────────────────────────────────────────────
        drift_timeline: List[float] = []
        if len(embeddings) >= 2:
            drift_timeline  = cosine_drift_timeline(embeddings)
        drift_arr           = np.asarray(drift_timeline) if drift_timeline else np.array([0.0])
        drift_mean          = float(drift_arr.mean())
        drift_max           = float(drift_arr.max())
        drift_std           = float(drift_arr.std())

        # ── 3. Heatmap temporal flux ──────────────────────────────────────
        flux_timeline: List[float] = []
        if len(heatmaps) >= 2:
            flux_timeline = heatmap_flux_timeline(heatmaps)
        flux_arr        = np.asarray(flux_timeline) if flux_timeline else np.array([0.0])
        flux_mean       = float(flux_arr.mean())
        flux_max        = float(flux_arr.max())

        heatmap_activity_mean = 0.0
        if heatmaps:
            stacked = np.stack([h.astype(np.float64) for h in heatmaps], axis=0)
            heatmap_activity_mean = float(stacked.mean())

        # ── 4. Landmark temporal signals ──────────────────────────────────
        lm_results = layer1.landmarks_for_track(track_id)
        valid_lm   = [lm for lm in lm_results if lm.detection_success]

        landmark_available    = len(valid_lm) >= self.min_valid_frames
        jaw_variance          = 0.0
        blink_rate            = 0.0
        eye_variance          = 0.0
        jaw_heatmap_corr      = 0.0

        if landmark_available:
            jaw_ratios  = [lm.jaw_open_ratio for lm in valid_lm]
            ear_left    = [lm.eye_blink_l    for lm in valid_lm]
            ear_right   = [lm.eye_blink_r    for lm in valid_lm]

            jaw_variance = float(np.var(jaw_ratios))
            blink_rate   = detect_blink_rate(
                ear_left, ear_right, fps,
                ear_threshold=self.blink_ear_threshold,
            )
            mean_ear = [(l + r) / 2.0 for l, r in zip(ear_left, ear_right)]
            eye_variance = float(np.var(mean_ear))

            # ── 5. Cross-modal correlation (jaw ↔ heatmap_activity) ───────
            # Align jaw_ratios and heatmap_activity_per_frame by frame_id
            # The valid_lm list may have a different length than valid_a
            # (different failure modes). Use the intersection of frame_ids.
            lm_frame_map: Dict[int, float] = {
                lm.frame_id: lm.jaw_open_ratio for lm in valid_lm
            }
            heatmap_per_frame: Dict[int, float] = {
                r.frame_id: float(r.manipulation_heatmap.mean())
                for r in valid_a
                if r.manipulation_heatmap is not None
            }
            shared_frames = sorted(
                set(lm_frame_map.keys()) & set(heatmap_per_frame.keys())
            )
            if len(shared_frames) >= 3:
                jaw_vals     = [lm_frame_map[fid]     for fid in shared_frames]
                heatmap_vals = [heatmap_per_frame[fid] for fid in shared_frames]
                jaw_heatmap_corr = pearson_correlation(jaw_vals, heatmap_vals)

        # ── 6. Scene-level statistics ─────────────────────────────────────
        scene_map, scene_consistency = compute_scene_fake_prob_map(
            frame_ids, fake_probs, scene_ids
        )

        # ── 7. Verdict derivation ─────────────────────────────────────────
        verdict, confidence, signals_fired = derive_verdict(
            fake_prob_mean=prob_mean,
            fake_prob_variance=prob_var,
            fake_prob_rolling_std=prob_rolling_std,
            fake_prob_trend_slope=prob_trend,
            fake_prob_peak_count=prob_peaks,
            embedding_drift_mean=drift_mean,
            embedding_drift_max=drift_max,
            heatmap_flux_mean=flux_mean,
            heatmap_activity_mean=heatmap_activity_mean,
            scene_consistency_score=scene_consistency,
            landmark_jaw_variance=jaw_variance,
            jaw_heatmap_correlation=jaw_heatmap_corr,
            landmark_available=landmark_available,
        )

        risk = risk_level(confidence)

        logger.info(
            "Branch B | track=%s | fake_prob_mean=%.4f | drift_mean=%.4f | "
            "flux_mean=%.4f | scene_cons=%.4f | "
            "verdict=%s (conf=%.3f, risk=%s) | signals=%s",
            track_id, prob_mean, drift_mean, flux_mean, scene_consistency,
            verdict, confidence, risk, signals_fired,
        )

        return TrackTemporalMetrics(
            track_id=track_id,
            n_frames=n_frames,
            n_valid_frames=n_valid,
            # IDS statistics
            fake_prob_mean=prob_mean,
            fake_prob_variance=prob_var,
            fake_prob_std=prob_std,
            fake_prob_max=prob_max,
            fake_prob_min=prob_min,
            fake_prob_trend_slope=prob_trend,
            fake_prob_rolling_std=prob_rolling_std,
            fake_prob_peak_count=prob_peaks,
            # Embedding drift
            embedding_drift_mean=drift_mean,
            embedding_drift_max=drift_max,
            embedding_drift_std=drift_std,
            embedding_drift_timeline=drift_timeline,
            # Heatmap flux
            heatmap_flux_mean=flux_mean,
            heatmap_flux_max=flux_max,
            heatmap_activity_mean=heatmap_activity_mean,
            # Landmarks
            landmark_jaw_variance=jaw_variance,
            landmark_blink_rate=blink_rate,
            landmark_eye_variance=eye_variance,
            landmark_available=landmark_available,
            # Cross-modal
            jaw_heatmap_correlation=jaw_heatmap_corr,
            # Scenes
            scene_fake_prob_map=scene_map,
            scene_consistency_score=scene_consistency,
            # Verdict
            temporal_verdict=verdict,
            temporal_confidence=confidence,
            temporal_signals_fired=signals_fired,
            temporal_risk_level=risk,
        )

    # ------------------------------------------------------------------
    # Convenience: quick single-track analysis
    # ------------------------------------------------------------------

    def analyze_track(
        self,
        track_id:  str,
        branch_a:  BranchAResult,
        layer1,
    ) -> TrackTemporalMetrics:
        """
        Analyse a single track without running the full video pipeline.
        Useful for streaming / incremental updates.
        """
        fps = max(layer1.metadata.target_fps, 1.0)
        frame_scene_map: Dict[int, int] = {
            fp.frame_id: fp.scene_id for fp in layer1.frames
        }
        return self._analyze_track(
            track_id=track_id,
            branch_a=branch_a,
            layer1=layer1,
            frame_scene_map=frame_scene_map,
            fps=fps,
        )

    def __repr__(self) -> str:
        return (
            f"TemporalConsistencyAnalyzer("
            f"window={self.rolling_window}, "
            f"min_valid={self.min_valid_frames}, "
            f"fake_threshold={self.verdict_fake_threshold})"
        )
