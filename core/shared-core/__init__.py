"""
DeepGuard Platform — Shared Core Library
Module  : core.shared-core

Provides base types, enumerations, abstract pipeline interfaces,
and exception hierarchy shared across all detection engines
(image-engine, video-engine, audio-engine, fusion-engine).
"""

from .result_types import DeepfakeVerdict, MediaModality, AnalysisResult, EngineResult
from .base_pipeline import BasePipeline
from .exceptions import (
    DeepGuardError,
    EngineNotReadyError,
    UnsupportedMediaError,
    AnalysisError,
    ModelLoadError,
)

__all__ = [
    "DeepfakeVerdict",
    "MediaModality",
    "AnalysisResult",
    "EngineResult",
    "BasePipeline",
    "DeepGuardError",
    "EngineNotReadyError",
    "UnsupportedMediaError",
    "AnalysisError",
    "ModelLoadError",
]
