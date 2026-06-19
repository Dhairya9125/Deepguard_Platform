"""
ADS — Temporal Splice Branch
Detects audio splicing — abrupt discontinuities at edit boundaries.
"""
import logging
from typing import Any, Dict, List
import numpy as np
from ..base_branch import BaseBranch
from ...domain.entities import BranchResult

logger = logging.getLogger(__name__)


class TemporalSpliceBranch(BaseBranch):
    name = "temporal_splice"

    def __init__(self, frame_size: int = 512, hop: int = 128) -> None:
        self.frame_size = frame_size
        self.hop = hop

    def predict(self, waveform: np.ndarray, features: Dict[str, Any]) -> BranchResult:
        try:
            # Compute frame-level RMS energy
            frames = [
                waveform[i : i + self.frame_size]
                for i in range(0, len(waveform) - self.frame_size, self.hop)
            ]
            rms_series = np.array([np.sqrt(np.mean(f**2)) for f in frames])

            if len(rms_series) < 4:
                return BranchResult(self.name, 0.5, 0.0)

            # Detect discontinuities via first-order differences
            diffs = np.abs(np.diff(rms_series))
            mean_diff = float(np.mean(diffs))
            max_diff  = float(np.max(diffs))
            std_diff  = float(np.std(diffs))

            # High max_diff relative to mean → potential splice
            splice_score = float(np.clip(max_diff / (mean_diff + 1e-8) / 20.0, 0.0, 1.0))
            confidence = min(0.80, 0.4 + abs(splice_score - 0.5))

            return BranchResult(
                branch_name=self.name,
                fake_score=splice_score,
                confidence=confidence,
                features={
                    "mean_energy_diff": round(mean_diff, 6),
                    "max_energy_diff":  round(max_diff, 6),
                    "std_energy_diff":  round(std_diff, 6),
                },
            )
        except Exception as exc:
            logger.error("TemporalSpliceBranch error: %s", exc)
            return BranchResult(self.name, 0.5, 0.0, error=str(exc))
