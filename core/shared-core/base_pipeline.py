"""
DeepGuard Platform — Abstract Base Pipeline
Module  : core.shared-core.base_pipeline

All detection engines (IDS, VDS, ADS) must implement BasePipeline.
This contract ensures every engine exposes a consistent predict() interface
that the API service layer can call uniformly.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from .result_types import AnalysisResult


class BasePipeline(ABC):
    """
    Abstract base class for all DeepGuard detection pipelines.

    Subclasses:
        IDSPipeline   — Image Detection Subsystem  (engines/image-engine)
        VDSPipeline   — Video Detection Subsystem  (engines/video-engine)
        ADSPipeline   — Audio Detection Subsystem  (engines/audio-engine)
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._is_loaded = False

    # ------------------------------------------------------------------ #
    # Abstract interface — subclasses must implement                       #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def load(self) -> None:
        """
        Load models and resources into memory.
        Called once at startup; subsequent calls are no-ops.
        """
        ...

    @abstractmethod
    def predict(self, media_path: Union[str, Path]) -> AnalysisResult:
        """
        Run the full detection pipeline on the given media file.

        Args:
            media_path: Absolute path to the media file to analyse.

        Returns:
            AnalysisResult with verdict, scores, and forensic report.

        Raises:
            UnsupportedMediaError: if the media format is not supported.
            AnalysisError: if the pipeline fails during processing.
        """
        ...

    # ------------------------------------------------------------------ #
    # Concrete helpers                                                     #
    # ------------------------------------------------------------------ #

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def ensure_loaded(self) -> None:
        """Load the pipeline if not already loaded (lazy init)."""
        if not self._is_loaded:
            self.load()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device!r}, loaded={self._is_loaded})"
