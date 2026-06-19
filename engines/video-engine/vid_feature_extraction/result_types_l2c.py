"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.result_types_l2c
Layer   : Layer 2 — Temporal Feature Extraction
Branch  : Branch C — Temporal Motion Modeling

Defines the output data contracts and structures produced by Branch C.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TrackMotionMetrics:
    """
    All temporal motion modeling signals and model scores computed for a single face track.
    """
    track_id: str
    n_frames: int = 0
    n_valid_frames: int = 0

    # --- Deep Learning Model Scores ---
    timesformer_score: float = 0.0  # [0, 1] fake probability from TimeSformer
    videomae_score: float = 0.0     # [0, 1] fake probability from VideoMAE
    slowfast_score: float = 0.0     # [0, 1] fake probability from SlowFast

    # --- Geometric & Motion Signals (NumPy-based) ---
    jitter_entropy: float = 0.0            # Shannon entropy of landmark displacements
    landmark_velocity_variance: float = 0.0 # Variance of face landmark movement velocity
    blink_asymmetry: float = 0.0           # Left vs. right eye EAR asymmetry
    blink_duration_anomaly: float = 0.0    # Deviation from normal blink durations
    frame_inconsistency_score: float = 0.0  # Mean inter-frame MSE of face crops

    # --- Verdict & Calibration ---
    motion_verdict: str = "UNCERTAIN"       # 'REAL' | 'FAKE' | 'UNCERTAIN'
    motion_confidence: float = 0.0          # Normalized confidence score in [0, 1]
    motion_signals_fired: List[str] = field(default_factory=list)
    motion_risk_level: str = "LOW"          # 'LOW' | 'MEDIUM' | 'HIGH'

    @property
    def is_motion_fake(self) -> bool:
        """Helper to quickly check if the track is flagged as FAKE."""
        return self.motion_verdict == "FAKE"

    @property
    def coverage(self) -> float:
        """Fraction of frames successfully analyzed in this track."""
        if self.n_frames == 0:
            return 0.0
        return self.n_valid_frames / self.n_frames

    def to_dict(self) -> Dict:
        """Serialize track metrics to a JSON-safe plain dictionary."""
        return {
            "track_id": self.track_id,
            "n_frames": self.n_frames,
            "n_valid_frames": self.n_valid_frames,
            "coverage": round(self.coverage, 4),
            # Model scores
            "timesformer_score": round(self.timesformer_score, 6),
            "videomae_score": round(self.videomae_score, 6),
            "slowfast_score": round(self.slowfast_score, 6),
            # Geometric/Motion metrics
            "jitter_entropy": round(self.jitter_entropy, 6),
            "landmark_velocity_variance": round(self.landmark_velocity_variance, 6),
            "blink_asymmetry": round(self.blink_asymmetry, 6),
            "blink_duration_anomaly": round(self.blink_duration_anomaly, 6),
            "frame_inconsistency_score": round(self.frame_inconsistency_score, 6),
            # Verdict
            "motion_verdict": self.motion_verdict,
            "motion_confidence": round(self.motion_confidence, 4),
            "motion_signals_fired": self.motion_signals_fired,
            "motion_risk_level": self.motion_risk_level,
        }


@dataclass
class BranchCResult:
    """
    Top-level output of VDS Layer 2, Branch C — Temporal Motion Modeling.
    """
    track_metrics: Dict[str, TrackMotionMetrics] = field(default_factory=dict)
    overall_verdict: str = "UNCERTAIN"
    overall_confidence: float = 0.0
    n_fake_tracks: int = 0
    n_real_tracks: int = 0
    n_uncertain_tracks: int = 0
    
    # Global aggregates
    video_timesformer_mean: float = 0.0
    video_videomae_mean: float = 0.0
    video_slowfast_mean: float = 0.0
    video_inconsistency_mean: float = 0.0
    
    processing_warnings: List[str] = field(default_factory=list)

    @property
    def unique_track_ids(self) -> List[str]:
        return sorted(self.track_metrics.keys())

    @property
    def n_tracks(self) -> int:
        return len(self.track_metrics)

    def metrics_for_track(self, track_id: str) -> Optional[TrackMotionMetrics]:
        return self.track_metrics.get(track_id)

    def summary(self) -> Dict:
        """JSON-serializable video-level summary."""
        return {
            "overall_verdict": self.overall_verdict,
            "overall_confidence": round(self.overall_confidence, 4),
            "n_tracks": self.n_tracks,
            "n_fake_tracks": self.n_fake_tracks,
            "n_real_tracks": self.n_real_tracks,
            "n_uncertain_tracks": self.n_uncertain_tracks,
            "video_timesformer_mean": round(self.video_timesformer_mean, 4),
            "video_videomae_mean": round(self.video_videomae_mean, 4),
            "video_slowfast_mean": round(self.video_slowfast_mean, 4),
            "video_inconsistency_mean": round(self.video_inconsistency_mean, 4),
            "n_warnings": len(self.processing_warnings),
            "warnings": self.processing_warnings,
            "tracks": {
                tid: m.to_dict()
                for tid, m in self.track_metrics.items()
            },
        }
