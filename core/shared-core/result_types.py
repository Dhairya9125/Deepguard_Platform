"""
DeepGuard Platform — Shared Result Types
Module  : core.shared-core.result_types

Defines cross-engine data types used to represent detection results
from image, video, and audio deepfake detection engines.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class DeepfakeVerdict(str, Enum):
    """Standardised three-way verdict for all detection engines."""
    REAL      = "REAL"
    FAKE      = "FAKE"
    UNCERTAIN = "UNCERTAIN"

    @classmethod
    def from_probability(cls, fake_probability: float, high: float = 0.75, low: float = 0.45) -> "DeepfakeVerdict":
        """Derive a verdict from a raw fake probability score."""
        if fake_probability >= high:
            return cls.FAKE
        elif fake_probability >= low:
            return cls.UNCERTAIN
        return cls.REAL


class MediaModality(str, Enum):
    """Media type being analysed."""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


@dataclass
class EngineResult:
    """
    Raw output from a single detection branch or sub-engine.
    All branch results are merged into an AnalysisResult by the fusion engine.
    """
    branch_name: str
    score: float                         # fake probability [0.0, 1.0]
    confidence: float                    # model confidence [0.0, 1.0]
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None          # non-None if branch failed


@dataclass
class AnalysisResult:
    """
    Final merged result produced by a detection pipeline.
    This is the canonical output type returned by BasePipeline.predict().
    """
    modality: MediaModality
    verdict: DeepfakeVerdict
    fake_probability: float              # [0.0, 1.0]
    authenticity_score: float            # [0.0, 100.0]  = (1 - fake_probability) * 100
    confidence: float                    # [0.0, 100.0]
    branch_results: List[EngineResult] = field(default_factory=list)
    forensic_report: str = ""
    processing_time_s: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary suitable for JSON storage."""
        return {
            "modality": self.modality.value,
            "verdict": self.verdict.value,
            "fake_probability": round(self.fake_probability, 6),
            "authenticity_score": round(self.authenticity_score, 2),
            "confidence": round(self.confidence, 2),
            "forensic_report": self.forensic_report,
            "processing_time_s": round(self.processing_time_s, 3),
            "created_at": self.created_at.isoformat(),
            "branch_results": [
                {
                    "branch": r.branch_name,
                    "score": round(r.score, 6),
                    "confidence": round(r.confidence, 6),
                    "metadata": r.metadata,
                    "error": r.error,
                }
                for r in self.branch_results
            ],
            **self.metadata,
        }
