"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.result_types_l2f
Layer   : Layer 2 — Temporal Feature Extraction
Branch  : Branch F — Identity Continuity Tracking

Defines output dataclasses for Branch F (Identity Continuity Tracking).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TrackContinuityMetrics:
    """
    Identity continuity and physical trajectory metrics computed for a single face track.
    """
    track_id: str
    n_frames: int = 0
    n_valid_frames: int = 0

    # --- Identity representation drift (Facenet models) ---
    facenet_drift_max: float = 0.0        # Max cosine distance to anchor embedding
    facenet_drift_var: float = 0.0        # Cosine distance variance over the track

    # --- Rigid geometry ratios instability (landmarks) ---
    geometry_instability_cv: float = 0.0  # Combined coefficient of variation of facial geometric proportions

    # --- Trajectory consistency (bboxes) ---
    max_trajectory_jump_px: float = 0.0   # Max bounding box center displacement between consecutive frames (pixels)
    trajectory_jump_count: int = 0        # Number of frames exceeding physical displacement limit

    # --- Verdict & Calibration ---
    continuity_verdict: str = "UNCERTAIN"  # 'REAL' | 'FAKE' | 'UNCERTAIN'
    continuity_confidence: float = 0.0     # Normalized confidence score in [0, 1]
    continuity_signals_fired: List[str] = field(default_factory=list)
    continuity_risk_level: str = "LOW"     # 'LOW' | 'MEDIUM' | 'HIGH'

    @property
    def is_continuity_fake(self) -> bool:
        """Helper to quickly check if the track is flagged as FAKE."""
        return self.continuity_verdict == "FAKE"

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
            # Identity representation metrics
            "facenet_drift_max": round(self.facenet_drift_max, 6),
            "facenet_drift_var": round(self.facenet_drift_var, 6),
            # Geometric ratios
            "geometry_instability_cv": round(self.geometry_instability_cv, 6),
            # Trajectory
            "max_trajectory_jump_px": round(self.max_trajectory_jump_px, 4),
            "trajectory_jump_count": self.trajectory_jump_count,
            # Verdict
            "continuity_verdict": self.continuity_verdict,
            "continuity_confidence": round(self.continuity_confidence, 4),
            "continuity_signals_fired": self.continuity_signals_fired,
            "continuity_risk_level": self.continuity_risk_level,
        }


@dataclass
class BranchFResult:
    """
    Top-level output of VDS Layer 2, Branch F — Identity Continuity Tracking.
    """
    track_metrics: Dict[str, TrackContinuityMetrics] = field(default_factory=dict)
    overall_verdict: str = "UNCERTAIN"
    overall_confidence: float = 0.0
    n_fake_tracks: int = 0
    n_real_tracks: int = 0
    n_uncertain_tracks: int = 0

    # Global video aggregates
    video_drift_mean: float = 0.0
    video_geometry_instability_mean: float = 0.0
    video_max_trajectory_jump_mean: float = 0.0

    processing_warnings: List[str] = field(default_factory=list)

    @property
    def unique_track_ids(self) -> List[str]:
        return sorted(self.track_metrics.keys())

    @property
    def n_tracks(self) -> int:
        return len(self.track_metrics)

    def metrics_for_track(self, track_id: str) -> Optional[TrackContinuityMetrics]:
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
            "video_drift_mean": round(self.video_drift_mean, 4),
            "video_geometry_instability_mean": round(self.video_geometry_instability_mean, 4),
            "video_max_trajectory_jump_mean": round(self.video_max_trajectory_jump_mean, 4),
            "n_warnings": len(self.processing_warnings),
            "warnings": self.processing_warnings,
            "tracks": {
                tid: m.to_dict()
                for tid, m in self.track_metrics.items()
            },
        }
