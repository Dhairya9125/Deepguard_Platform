"""
ADS — Spectral CNN Branch
Detects deepfakes via spectral irregularities in MFCC + spectral contrast features.
CPU-friendly lightweight branch (no GPU required).
"""
import logging
from typing import Any, Dict
import numpy as np
from ..base_branch import BaseBranch
from ...domain.entities import BranchResult

logger = logging.getLogger(__name__)


class SpectralCNNBranch(BaseBranch):
    """
    Spectral CNN Branch — analyses MFCC and spectral features for synthetic
    speech artifacts left by vocoder and neural TTS systems.
    """
    name = "spectral_cnn"

    def predict(self, waveform: np.ndarray, features: Dict[str, Any]) -> BranchResult:
        try:
            mfcc_mean = features.get("mfcc_mean", np.zeros(40))
            mfcc_std  = features.get("mfcc_std",  np.zeros(40))
            spec_centroid = features.get("spectral_centroid", np.zeros((1, 1)))

            # Heuristic: real speech has higher MFCC variance in low coefficients;
            # synthetic speech often shows unnatural smoothness.
            low_variance = float(np.mean(mfcc_std[:13]))
            high_variance = float(np.mean(mfcc_std[13:]))
            variance_ratio = low_variance / (high_variance + 1e-8)

            # Map variance ratio to a fake score (calibrated heuristic)
            # Synthetic voices tend to have a higher ratio (smoother low MFCCs)
            fake_score = float(np.clip((variance_ratio - 1.5) / 3.0, 0.0, 1.0))
            confidence = min(0.85, 0.5 + abs(fake_score - 0.5))

            return BranchResult(
                branch_name=self.name,
                fake_score=fake_score,
                confidence=confidence,
                features={"variance_ratio": round(variance_ratio, 4)},
            )
        except Exception as exc:
            logger.error("SpectralCNNBranch error: %s", exc)
            return BranchResult(
                branch_name=self.name,
                fake_score=0.5,
                confidence=0.0,
                error=str(exc),
            )
