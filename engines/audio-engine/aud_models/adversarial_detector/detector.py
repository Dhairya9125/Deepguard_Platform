"""
ADS — Adversarial Detector Branch
Detects adversarial perturbations added to fool speaker verification systems.
"""
import logging
from typing import Any, Dict
import numpy as np
from ..base_branch import BaseBranch
from ...domain.entities import BranchResult

logger = logging.getLogger(__name__)


class AdversarialDetectorBranch(BaseBranch):
    name = "adversarial_detector"

    def predict(self, waveform: np.ndarray, features: Dict[str, Any]) -> BranchResult:
        try:
            # Adversarial perturbations often manifest as unnatural high-freq noise
            zcr = features.get("zcr", np.zeros((1, 100)))
            rms = features.get("rms", np.zeros((1, 100)))
            mean_zcr = float(np.mean(zcr))
            mean_rms = float(np.mean(rms))

            # Unnaturally high ZCR relative to RMS → adversarial noise indicator
            zcr_to_rms = mean_zcr / (mean_rms + 1e-8)
            fake_score = float(np.clip((zcr_to_rms - 5.0) / 20.0, 0.0, 1.0))
            confidence = 0.50
            return BranchResult(
                branch_name=self.name,
                fake_score=fake_score,
                confidence=confidence,
                features={
                    "mean_zcr": round(mean_zcr, 6),
                    "mean_rms": round(mean_rms, 6),
                    "zcr_to_rms": round(zcr_to_rms, 4),
                },
            )
        except Exception as exc:
            return BranchResult(self.name, 0.5, 0.0, error=str(exc))
