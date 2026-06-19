"""
ADS — Base Detection Branch
All 7 ADS branches inherit from BaseBranch.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np
from ..aud_domain.entities import BranchResult


class BaseBranch(ABC):
    """Abstract base for all ADS detection branches."""

    name: str = "base"

    @abstractmethod
    def predict(self, waveform: np.ndarray, features: Dict[str, Any]) -> BranchResult:
        """
        Run the branch detector on a waveform + pre-extracted features.

        Args:
            waveform: mono float32 numpy array at 16 kHz.
            features: dict from SpectralFeatureExtractor.extract().

        Returns:
            BranchResult with fake_score and confidence.
        """
        ...
