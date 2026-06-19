"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.result_types_l2e
Layer   : Layer 2 — Temporal Feature Extraction
Branch  : Branch E — rPPG Biological Signal Analysis

Defines output dataclasses for Branch E (rPPG Biological Signal Analysis).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TrackRPPGMetrics:
    """
    Physiological rPPG metrics and deep learning blood-pulse scores computed for a single face track.
    """
    track_id: str
    n_frames: int = 0
    n_valid_frames: int = 0

    # --- Deep Learning rPPG Model Scores ---
    physnet_score: float = 0.0      # [0, 1] fake probability from PhysNet
    deepphys_score: float = 0.0     # [0, 1] fake probability from DeepPhys

    # --- NumPy Physiological Signals ---
    pulse_rate_mean: float = 0.0          # Estimated mean heart rate in beats-per-minute (BPM)
    pulse_rate_std: float = 0.0           # Temporal variance of heart rate (consistency)
    pulse_snr: float = 0.0                # Signal-to-noise ratio of the estimated heart rate peak
    spatial_correlation_mean: float = 0.0  # Mean correlation between forehead, left cheek, and right cheek

    # --- Verdict & Calibration ---
    rppg_verdict: str = "UNCERTAIN"       # 'REAL' | 'FAKE' | 'UNCERTAIN'
    rppg_confidence: float = 0.0          # Normalized confidence score in [0, 1]
    rppg_signals_fired: List[str] = field(default_factory=list)
    rppg_risk_level: str = "LOW"          # 'LOW' | 'MEDIUM' | 'HIGH'

    @property
    def is_rppg_fake(self) -> bool:
        """Helper to quickly check if the track is flagged as FAKE."""
        return self.rppg_verdict == "FAKE"

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
            # Model metrics
            "physnet_score": round(self.physnet_score, 6),
            "deepphys_score": round(self.deepphys_score, 6),
            # Physiological signals
            "pulse_rate_mean": round(self.pulse_rate_mean, 4),
            "pulse_rate_std": round(self.pulse_rate_std, 6),
            "pulse_snr": round(self.pulse_snr, 6),
            "spatial_correlation_mean": round(self.spatial_correlation_mean, 6),
            # Verdict
            "rppg_verdict": self.rppg_verdict,
            "rppg_confidence": round(self.rppg_confidence, 4),
            "rppg_signals_fired": self.rppg_signals_fired,
            "rppg_risk_level": self.rppg_risk_level,
        }


@dataclass
class BranchEResult:
    """
    Top-level output of VDS Layer 2, Branch E — rPPG Biological Signal Analysis.
    """
    track_metrics: Dict[str, TrackRPPGMetrics] = field(default_factory=dict)
    overall_verdict: str = "UNCERTAIN"
    overall_confidence: float = 0.0
    n_fake_tracks: int = 0
    n_real_tracks: int = 0
    n_uncertain_tracks: int = 0

    # Global video aggregates
    video_physnet_mean: float = 0.0
    video_deepphys_mean: float = 0.0
    video_pulse_rate_mean: float = 0.0
    video_correlation_mean: float = 0.0

    processing_warnings: List[str] = field(default_factory=list)

    @property
    def unique_track_ids(self) -> List[str]:
        return sorted(self.track_metrics.keys())

    @property
    def n_tracks(self) -> int:
        return len(self.track_metrics)

    def metrics_for_track(self, track_id: str) -> Optional[TrackRPPGMetrics]:
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
            "video_physnet_mean": round(self.video_physnet_mean, 4),
            "video_deepphys_mean": round(self.video_deepphys_mean, 4),
            "video_pulse_rate_mean": round(self.video_pulse_rate_mean, 4),
            "video_correlation_mean": round(self.video_correlation_mean, 4),
            "n_warnings": len(self.processing_warnings),
            "warnings": self.processing_warnings,
            "tracks": {
                tid: m.to_dict()
                for tid, m in self.track_metrics.items()
            },
        }
