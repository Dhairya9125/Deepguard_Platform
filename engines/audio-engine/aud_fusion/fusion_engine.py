"""
ADS — Weighted Ensemble Fusion Engine
Combines scores from all 7 detection branches into a final verdict.
"""
import logging
from typing import List
from ..aud_domain.entities import BranchResult, AudioDetectionResult
from ..aud_config.settings import ADSConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class ADSFusionEngine:
    """
    Weighted ensemble: each branch contributes its fake_score
    scaled by its confidence and configured weight.
    """

    def __init__(self, config: ADSConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._weights = {
            "spectral_cnn":        config.spectral_weight,
            "temporal_splice":     config.temporal_weight,
            "voice_biometrics":    config.biometric_weight,
            "wavlm":               config.wavlm_weight,
            "xlsr":                config.xlsr_weight,
            "diffusion_detector":  config.diffusion_weight,
            "adversarial_detector": config.adversarial_weight,
        }

    def fuse(self, branch_results: List[BranchResult]) -> float:
        """
        Compute the final fake probability as a weighted average.
        Branches with errors are excluded from the weighting.
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for result in branch_results:
            if result.error:
                logger.warning("Branch '%s' failed — excluded from aud_fusion.", result.branch_name)
                continue
            weight = self._weights.get(result.branch_name, 0.1)
            effective_weight = weight * result.confidence
            weighted_sum += result.fake_score * effective_weight
            total_weight += effective_weight

        if total_weight < 1e-8:
            logger.warning("All branches failed. Returning uncertain score.")
            return 0.5

        return float(weighted_sum / total_weight)
