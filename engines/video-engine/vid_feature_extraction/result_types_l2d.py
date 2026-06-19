"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.result_types_l2d
Layer   : Layer 2 — Temporal Feature Extraction
Branch  : Branch D — Cross-Modal Synchronization

Defines output dataclasses for Branch D (Cross-Modal Synchronization).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TrackSyncMetrics:
    """
    Audio-visual synchronization and cross-modal metrics computed for a single face track.
    """
    track_id: str
    n_frames: int = 0
    n_valid_frames: int = 0

    # --- SyncNet Metrics ---
    syncnet_confidence: float = 0.0  # Lip-sync confidence level
    syncnet_offset: int = 0          # Optimal alignment shift in frames (delay)

    # --- AV-HuBERT Metrics ---
    av_hubert_score: float = 0.0     # [0, 1] synchronization/fake probability from AV-HuBERT

    # --- NumPy Forensic Signals ---
    phoneme_viseme_inconsistency: float = 0.0 # [0, 1] mismatch between audio phonemes & landmarks
    audio_video_delay: float = 0.0            # Best delay offset in seconds
    emotion_mismatch_score: float = 0.0       # [0, 1] facial expressions vs prosody mismatch

    # --- Verdict & Calibration ---
    sync_verdict: str = "UNCERTAIN"       # 'REAL' | 'FAKE' | 'UNCERTAIN'
    sync_confidence: float = 0.0          # Normalized confidence score in [0, 1]
    sync_signals_fired: List[str] = field(default_factory=list)
    sync_risk_level: str = "LOW"          # 'LOW' | 'MEDIUM' | 'HIGH'

    @property
    def is_sync_fake(self) -> bool:
        """Helper to quickly check if the track is flagged as FAKE."""
        return self.sync_verdict == "FAKE"

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
            "syncnet_confidence": round(self.syncnet_confidence, 6),
            "syncnet_offset": self.syncnet_offset,
            "av_hubert_score": round(self.av_hubert_score, 6),
            # Forensic signals
            "phoneme_viseme_inconsistency": round(self.phoneme_viseme_inconsistency, 6),
            "audio_video_delay": round(self.audio_video_delay, 6),
            "emotion_mismatch_score": round(self.emotion_mismatch_score, 6),
            # Verdict
            "sync_verdict": self.sync_verdict,
            "sync_confidence": round(self.sync_confidence, 4),
            "sync_signals_fired": self.sync_signals_fired,
            "sync_risk_level": self.sync_risk_level,
        }


@dataclass
class BranchDResult:
    """
    Top-level output of VDS Layer 2, Branch D — Cross-Modal Synchronization.
    """
    track_metrics: Dict[str, TrackSyncMetrics] = field(default_factory=dict)
    overall_verdict: str = "UNCERTAIN"
    overall_confidence: float = 0.0
    n_fake_tracks: int = 0
    n_real_tracks: int = 0
    n_uncertain_tracks: int = 0

    # Global video aggregates
    video_syncnet_conf_mean: float = 0.0
    video_av_hubert_mean: float = 0.0
    video_delay_mean: float = 0.0
    video_emotion_mismatch_mean: float = 0.0

    processing_warnings: List[str] = field(default_factory=list)

    @property
    def unique_track_ids(self) -> List[str]:
        return sorted(self.track_metrics.keys())

    @property
    def n_tracks(self) -> int:
        return len(self.track_metrics)

    def metrics_for_track(self, track_id: str) -> Optional[TrackSyncMetrics]:
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
            "video_syncnet_conf_mean": round(self.video_syncnet_conf_mean, 4),
            "video_av_hubert_mean": round(self.video_av_hubert_mean, 4),
            "video_delay_mean": round(self.video_delay_mean, 4),
            "video_emotion_mismatch_mean": round(self.video_emotion_mismatch_mean, 4),
            "n_warnings": len(self.processing_warnings),
            "warnings": self.processing_warnings,
            "tracks": {
                tid: m.to_dict()
                for tid, m in self.track_metrics.items()
            },
        }
