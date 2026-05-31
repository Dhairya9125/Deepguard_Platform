"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.result_types_l2b
Layer   : Layer 2 — Temporal Feature Extraction
Branch  : Branch B — Temporal Consistency Analysis

Defines all result types produced by Branch B and consumed by:
  - VDS Layer 3 (Temporal Fusion Engine)
  - VDS trust layer / reporting API

Data hierarchy:
    BranchAResult + VideoPreprocessingResult (Layer 1)
           │
           ▼ Branch B analyses per-track timelines
    TrackTemporalMetrics        ← all temporal signals for ONE track
           │
           ▼ aggregated over all tracks
    BranchBResult               ← top-level Branch B output

Temporal Signals Computed per Track
    ─────────────────────────────────────────────────────────
    Signal                   Source                  Forensic value
    ─────────────────────────────────────────────────────────
    fake_prob_mean           Branch A timelines      Overall confidence
    fake_prob_variance       Branch A timelines      Flickering = deepfake instability
    fake_prob_rolling_std    Branch A timelines      Local temporal inconsistency
    fake_prob_trend_slope    Branch A timelines      Rising = gradual swap blending
    embedding_drift          Branch A artifact_emb   Cosine distance between frames
    embedding_drift_max      Branch A artifact_emb   Peak drift = swap transition
    heatmap_temporal_flux    Branch A heatmaps       Pixel-wise heatmap change rate
    landmark_jaw_variance    Layer 1 LandmarkTracer  Abnormal mouth movement
    landmark_blink_rate      Layer 1 LandmarkTracer  Unnatural blink cadence
    landmark_visual_corr     Branch A + Layer 1      EAR ↔ heatmap_activity correlation
    scene_consistency_score  Branch A + Layer 1      Per-scene mean vs. global baseline
    temporal_verdict         All signals             REAL / FAKE / UNCERTAIN
    temporal_confidence      All signals             [0, 1] confidence in verdict
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Per-track temporal metrics
# ---------------------------------------------------------------------------

@dataclass
class TrackTemporalMetrics:
    """
    All temporal consistency signals computed for a single face track.

    This is the atomic output of Branch B — one per track_id.

    ------- IDS-signal temporal statistics (from Branch A) -------

    Attributes:
        track_id                : Stable track identifier.
        n_frames                : Total frames in the track (including failures).
        n_valid_frames          : Frames with detection_success=True.

        fake_prob_mean          : Mean fake_probability across valid frames.
        fake_prob_variance      : Variance of fake_probability — high = flickering.
        fake_prob_std           : Standard deviation of fake_probability.
        fake_prob_max           : Maximum observed fake_probability.
        fake_prob_min           : Minimum observed fake_probability.
        fake_prob_trend_slope   : Linear regression slope of the fake_prob timeline.
                                  Positive = probability rising over time (gradual swap).
                                  Near-zero = stable (real face or consistent fake).
        fake_prob_rolling_std   : Mean of rolling window (w=5) std of fake_prob.
                                  High = local fluctuations (deepfake instability).
        fake_prob_peak_count    : Number of frames where fake_prob > 0.7
                                  and neighbours are < 0.5 (isolated spikes).

    ------- Embedding drift (from Branch A artifact_embedding) -------

        embedding_drift_mean    : Mean frame-to-frame cosine distance between
                                  consecutive artifact_embedding (2560,) vectors.
                                  Real faces: low drift (<0.05).
                                  Deepfakes: high drift (>0.15) — latent space jumps.
        embedding_drift_max     : Maximum cosine drift in any consecutive pair.
        embedding_drift_std     : Std of frame-to-frame cosine distances.
        embedding_drift_timeline: List of consecutive cosine distances (length n-1).

    ------- Heatmap temporal flux (from Branch A manipulation_heatmap) -------

        heatmap_flux_mean       : Mean absolute pixel-wise change between consecutive
                                  heatmaps. Real faces: near-zero.
                                  Deepfakes: higher as synthetic region boundaries shift.
        heatmap_flux_max        : Maximum frame-to-frame heatmap change.
        heatmap_activity_mean   : Mean heatmap pixel value averaged over all frames
                                  and pixels — overall manipulation signal.

    ------- Landmark temporal signals (from Layer 1 LandmarkTracer) -------

        landmark_jaw_variance   : Variance of jaw_open_ratio timeline.
                                  Deepfakes often produce unnatural jaw movement.
        landmark_blink_rate     : Mean blinks per second derived from EAR dips.
                                  Normal: 0.2–0.4 blinks/s. Anomalies = suspect.
        landmark_eye_variance   : Variance of mean EAR (left+right) — blink jitter.
        landmark_available      : True if Layer 1 landmark data was available.

    ------- Cross-modal correlation -------

        jaw_heatmap_correlation : Pearson r between jaw_open_ratio and
                                  mean heatmap activity per frame.
                                  Real speech: low-to-moderate correlation.
                                  Lip-sync deepfakes: abnormally high correlation
                                  (heatmap follows mouth exactly) or abnormally
                                  low (mouth moves, heatmap unchanged).

    ------- Scene-level statistics -------

        scene_fake_prob_map     : Dict[scene_id → mean_fake_prob in that scene].
        scene_consistency_score : Std of per-scene mean_fake_prob values.
                                  High = fake probability changes across scenes
                                  (face swap introduced mid-video).

    ------- Final Branch B verdict -------

        temporal_verdict        : 'FAKE' | 'REAL' | 'UNCERTAIN'
        temporal_confidence     : [0, 1] — confidence in the temporal verdict.
        temporal_signals_fired  : List of signal names that contributed to FAKE verdict.
        temporal_risk_level     : 'HIGH' | 'MEDIUM' | 'LOW' based on confidence.
    """
    track_id:                  str
    n_frames:                  int                     = 0
    n_valid_frames:            int                     = 0

    # --- IDS fake_prob statistics ---
    fake_prob_mean:            float                   = 0.0
    fake_prob_variance:        float                   = 0.0
    fake_prob_std:             float                   = 0.0
    fake_prob_max:             float                   = 0.0
    fake_prob_min:             float                   = 1.0
    fake_prob_trend_slope:     float                   = 0.0
    fake_prob_rolling_std:     float                   = 0.0
    fake_prob_peak_count:      int                     = 0

    # --- Embedding drift ---
    embedding_drift_mean:      float                   = 0.0
    embedding_drift_max:       float                   = 0.0
    embedding_drift_std:       float                   = 0.0
    embedding_drift_timeline:  List[float]             = field(default_factory=list)

    # --- Heatmap temporal flux ---
    heatmap_flux_mean:         float                   = 0.0
    heatmap_flux_max:          float                   = 0.0
    heatmap_activity_mean:     float                   = 0.0

    # --- Landmark signals ---
    landmark_jaw_variance:     float                   = 0.0
    landmark_blink_rate:       float                   = 0.0
    landmark_eye_variance:     float                   = 0.0
    landmark_available:        bool                    = False

    # --- Cross-modal correlation ---
    jaw_heatmap_correlation:   float                   = 0.0

    # --- Scene-level ---
    scene_fake_prob_map:       Dict[int, float]        = field(default_factory=dict)
    scene_consistency_score:   float                   = 0.0

    # --- Verdict ---
    temporal_verdict:          str                     = "UNCERTAIN"
    temporal_confidence:       float                   = 0.0
    temporal_signals_fired:    List[str]               = field(default_factory=list)
    temporal_risk_level:       str                     = "LOW"

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def is_temporally_fake(self) -> bool:
        return self.temporal_verdict == "FAKE"

    @property
    def coverage(self) -> float:
        """Fraction of frames successfully analyzed."""
        if self.n_frames == 0:
            return 0.0
        return self.n_valid_frames / self.n_frames

    def to_dict(self) -> Dict:
        """Serialise to a JSON-safe plain dict (no numpy arrays)."""
        return {
            "track_id":                 self.track_id,
            "n_frames":                 self.n_frames,
            "n_valid_frames":           self.n_valid_frames,
            "coverage":                 round(self.coverage, 4),
            # IDS statistics
            "fake_prob_mean":           round(self.fake_prob_mean, 6),
            "fake_prob_variance":       round(self.fake_prob_variance, 6),
            "fake_prob_std":            round(self.fake_prob_std, 6),
            "fake_prob_max":            round(self.fake_prob_max, 6),
            "fake_prob_min":            round(self.fake_prob_min, 6),
            "fake_prob_trend_slope":    round(self.fake_prob_trend_slope, 6),
            "fake_prob_rolling_std":    round(self.fake_prob_rolling_std, 6),
            "fake_prob_peak_count":     self.fake_prob_peak_count,
            # Embedding drift
            "embedding_drift_mean":     round(self.embedding_drift_mean, 6),
            "embedding_drift_max":      round(self.embedding_drift_max, 6),
            "embedding_drift_std":      round(self.embedding_drift_std, 6),
            "embedding_drift_n_pairs":  len(self.embedding_drift_timeline),
            # Heatmap
            "heatmap_flux_mean":        round(self.heatmap_flux_mean, 6),
            "heatmap_flux_max":         round(self.heatmap_flux_max, 6),
            "heatmap_activity_mean":    round(self.heatmap_activity_mean, 6),
            # Landmarks
            "landmark_available":       self.landmark_available,
            "landmark_jaw_variance":    round(self.landmark_jaw_variance, 6),
            "landmark_blink_rate":      round(self.landmark_blink_rate, 4),
            "landmark_eye_variance":    round(self.landmark_eye_variance, 6),
            # Cross-modal
            "jaw_heatmap_correlation":  round(self.jaw_heatmap_correlation, 6),
            # Scene
            "scene_consistency_score":  round(self.scene_consistency_score, 6),
            "scene_fake_prob_map":      {str(k): round(v, 4)
                                         for k, v in self.scene_fake_prob_map.items()},
            # Verdict
            "temporal_verdict":         self.temporal_verdict,
            "temporal_confidence":      round(self.temporal_confidence, 4),
            "temporal_risk_level":      self.temporal_risk_level,
            "temporal_signals_fired":   self.temporal_signals_fired,
        }


# ---------------------------------------------------------------------------
# Branch B aggregate result
# ---------------------------------------------------------------------------

@dataclass
class BranchBResult:
    """
    Top-level output of VDS Layer 2, Branch B — Temporal Consistency Analysis.

    Contains one `TrackTemporalMetrics` per face track, plus video-level
    aggregates used by VDS Layer 3 Temporal Fusion Engine.

    Attributes:
        track_metrics       : Dict[track_id → TrackTemporalMetrics].
        video_fake_prob_mean: Mean fake_prob across ALL tracks and ALL frames.
                              Video-level global baseline.
        video_drift_mean    : Mean embedding drift across all tracks.
        n_fake_tracks       : Tracks with temporal_verdict == 'FAKE'.
        n_real_tracks       : Tracks with temporal_verdict == 'REAL'.
        n_uncertain_tracks  : Tracks with temporal_verdict == 'UNCERTAIN'.
        overall_verdict     : Majority-vote video-level verdict.
        overall_confidence  : Mean of per-track temporal_confidence values.
        processing_warnings : Non-fatal warnings from Branch B.
    """
    track_metrics:          Dict[str, TrackTemporalMetrics]
    video_fake_prob_mean:   float                   = 0.0
    video_drift_mean:       float                   = 0.0
    n_fake_tracks:          int                     = 0
    n_real_tracks:          int                     = 0
    n_uncertain_tracks:     int                     = 0
    overall_verdict:        str                     = "UNCERTAIN"
    overall_confidence:     float                   = 0.0
    processing_warnings:    List[str]               = field(default_factory=list)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def unique_track_ids(self) -> List[str]:
        return sorted(self.track_metrics.keys())

    @property
    def n_tracks(self) -> int:
        return len(self.track_metrics)

    def metrics_for_track(self, track_id: str) -> Optional[TrackTemporalMetrics]:
        return self.track_metrics.get(track_id)

    def signals_fired_for_track(self, track_id: str) -> List[str]:
        m = self.metrics_for_track(track_id)
        return m.temporal_signals_fired if m else []

    def summary(self) -> Dict:
        """JSON-serialisable video-level summary."""
        return {
            "overall_verdict":       self.overall_verdict,
            "overall_confidence":    round(self.overall_confidence, 4),
            "n_tracks":              self.n_tracks,
            "n_fake_tracks":         self.n_fake_tracks,
            "n_real_tracks":         self.n_real_tracks,
            "n_uncertain_tracks":    self.n_uncertain_tracks,
            "video_fake_prob_mean":  round(self.video_fake_prob_mean, 4),
            "video_drift_mean":      round(self.video_drift_mean, 6),
            "n_warnings":            len(self.processing_warnings),
            "warnings":              self.processing_warnings,
            "tracks":                {
                tid: m.to_dict()
                for tid, m in self.track_metrics.items()
            },
        }
