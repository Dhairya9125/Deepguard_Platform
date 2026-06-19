"""
ADS Domain Entities — core data objects.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class AudioSample:
    """
    Represents a loaded audio clip ready for analysis.
    """
    waveform: np.ndarray      # shape: (samples,) — mono float32
    sample_rate: int
    duration_s: float
    file_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BranchResult:
    """Result from a single detection branch."""
    branch_name: str
    fake_score: float         # [0.0, 1.0]
    confidence: float         # [0.0, 1.0]
    features: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class AudioDetectionResult:
    """
    Final detection result from the ADS pipeline.
    """
    verdict: str                              # 'REAL', 'FAKE', 'UNCERTAIN'
    fake_probability: float                   # [0.0, 1.0]
    authenticity_score: float                 # [0.0, 100.0]
    confidence: float                         # [0.0, 100.0]
    branch_results: List[BranchResult] = field(default_factory=list)
    forensic_report: str = ""
    processing_time_s: float = 0.0
    file_name: Optional[str] = None
    duration_s: float = 0.0
    sample_rate: int = 16000
    manipulation_segments: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "fake_probability": round(self.fake_probability, 6),
            "authenticity_score": round(self.authenticity_score, 2),
            "confidence": round(self.confidence, 2),
            "forensic_report": self.forensic_report,
            "processing_time_s": round(self.processing_time_s, 3),
            "file_name": self.file_name,
            "duration_s": round(self.duration_s, 3),
            "sample_rate": self.sample_rate,
            "manipulation_segments": self.manipulation_segments,
            "branch_results": [
                {
                    "branch": r.branch_name,
                    "fake_score": round(r.fake_score, 6),
                    "confidence": round(r.confidence, 6),
                    "error": r.error,
                }
                for r in self.branch_results
            ],
        }
