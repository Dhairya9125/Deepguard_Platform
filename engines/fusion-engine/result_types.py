"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.fusion-engine.result_types
Layer   : Layer 3 — Cross-Modal Fusion Engine

Defines the output data contracts and structures produced by Layer 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FusionLocalizationResult:
    """
    Holds the temporal forgery localization timelines and modality attribution scores.

    Attributes:
        temporal_timeline       : List of frame-level forgery probabilities (scalar in [0, 1]).
                                  0.0 = completely real, 1.0 = completely fake.
        modality_attribution    : Dict of culprit/contribution scores per modality.
                                  Keys: 'audio', 'visual', 'temporal', 'biological'.
                                  Values: float in [0, 1].
    """
    temporal_timeline: List[float] = field(default_factory=list)
    modality_attribution: Dict[str, float] = field(
        default_factory=lambda: {
            "audio": 0.0,
            "visual": 0.0,
            "temporal": 0.0,
            "biological": 0.0,
        }
    )

    def to_dict(self) -> Dict:
        """Serialize localization result to a JSON-safe plain dictionary."""
        return {
            "temporal_timeline": [round(float(p), 4) for p in self.temporal_timeline],
            "modality_attribution": {
                k: round(float(v), 4) for k, v in self.modality_attribution.items()
            },
        }


@dataclass
class FusionTrackResult:
    """
    Per-track fusion metrics containing localized segments, culprits, and track-level confidence.

    Attributes:
        track_id            : Stable track ID of the analyzed face.
        verdict             : Verdict for this track: 'REAL' | 'FAKE' | 'UNCERTAIN'.
        confidence          : Verdict confidence in [0, 1].
        localization        : FusionLocalizationResult containing timeline and modality ratings.
        modality_culprits   : Ordered contribution rating of which modality triggered the fake alert.
    """
    track_id: str
    verdict: str = "UNCERTAIN"
    confidence: float = 0.0
    localization: FusionLocalizationResult = field(default_factory=FusionLocalizationResult)
    modality_culprits: Dict[str, float] = field(default_factory=dict)

    @property
    def is_fake(self) -> bool:
        """Helper to check if this track is classified as FAKE."""
        return self.verdict == "FAKE"

    def to_dict(self) -> Dict:
        """Serialize track fusion result to a JSON-safe plain dictionary."""
        return {
            "track_id": self.track_id,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 4),
            "localization": self.localization.to_dict(),
            "modality_culprits": {
                k: round(float(v), 4) for k, v in self.modality_culprits.items()
            },
        }


@dataclass
class FusionResult:
    """
    Global video aggregates including the overall verdict, confidence, and track breakdowns.

    Attributes:
        overall_verdict     : Unified overall verdict for the entire video: 'REAL' | 'FAKE' | 'UNCERTAIN'.
        overall_confidence  : Aggregated global confidence in [0, 1].
        track_results       : Dict mapping track_id to its corresponding FusionTrackResult.
        processing_warnings : List of non-fatal warnings emitted during fusion (e.g. missing branches).
        metadata            : Dict of video-level context data (e.g. frame count, fps, processing time).
    """
    overall_verdict: str = "UNCERTAIN"
    overall_confidence: float = 0.0
    track_results: Dict[str, FusionTrackResult] = field(default_factory=dict)
    processing_warnings: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    @property
    def unique_track_ids(self) -> List[str]:
        """Return the sorted list of track IDs."""
        return sorted(self.track_results.keys())

    @property
    def n_tracks(self) -> int:
        """Number of tracks processed."""
        return len(self.track_results)

    def summary(self) -> Dict:
        """JSON-serializable summary of the global fusion results."""
        return {
            "overall_verdict": self.overall_verdict,
            "overall_confidence": round(self.overall_confidence, 4),
            "n_tracks": self.n_tracks,
            "tracks": {
                tid: r.to_dict() for tid, r in self.track_results.items()
            },
            "processing_warnings": self.processing_warnings,
            "metadata": self.metadata,
        }
