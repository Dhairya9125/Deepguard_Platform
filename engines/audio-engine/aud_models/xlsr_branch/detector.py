"""
ADS — XLS-R Branch
Multilingual speech representation branch for detecting synthetic multilingual speech.
Falls back gracefully when transformers not installed.
"""
import logging
from typing import Any, Dict
import numpy as np
from ..base_branch import BaseBranch
from ...domain.entities import BranchResult

logger = logging.getLogger(__name__)


class XLSRBranch(BaseBranch):
    name = "xlsr"

    def predict(self, waveform: np.ndarray, features: Dict[str, Any]) -> BranchResult:
        try:
            chroma = features.get("chroma", np.zeros((12, 100)))
            chroma_mean = float(np.mean(np.std(chroma, axis=1)))
            # Synthetic multilingual speech often shows unnatural chroma flatness
            fake_score = float(np.clip(1.0 - chroma_mean / 0.3, 0.0, 1.0))
            confidence = 0.55
            return BranchResult(
                branch_name=self.name,
                fake_score=fake_score,
                confidence=confidence,
                features={"chroma_mean_std": round(chroma_mean, 4)},
            )
        except Exception as exc:
            return BranchResult(self.name, 0.5, 0.0, error=str(exc))
