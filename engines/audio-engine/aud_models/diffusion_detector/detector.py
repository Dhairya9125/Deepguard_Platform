"""
ADS — Diffusion Detector Branch
Detects artifacts characteristic of diffusion-model-based audio synthesis
(e.g., ElevenLabs, Bark, AudioLDM, Stable Audio).
"""
import logging
from typing import Any, Dict
import numpy as np
from ..base_branch import BaseBranch
from ...domain.entities import BranchResult

logger = logging.getLogger(__name__)


class DiffusionDetectorBranch(BaseBranch):
    name = "diffusion_detector"

    def predict(self, waveform: np.ndarray, features: Dict[str, Any]) -> BranchResult:
        try:
            # Diffusion models leave characteristic high-frequency artifacts
            spec_contrast = features.get("spectral_contrast", np.zeros((7, 100)))
            # High contrast in upper bands is a diffusion artifact indicator
            upper_bands = spec_contrast[-3:, :]  # top 3 spectral bands
            mean_upper_contrast = float(np.mean(upper_bands))

            fake_score = float(np.clip(mean_upper_contrast / 30.0, 0.0, 1.0))
            confidence = 0.60
            return BranchResult(
                branch_name=self.name,
                fake_score=fake_score,
                confidence=confidence,
                features={"mean_upper_contrast": round(mean_upper_contrast, 4)},
            )
        except Exception as exc:
            return BranchResult(self.name, 0.5, 0.0, error=str(exc))
