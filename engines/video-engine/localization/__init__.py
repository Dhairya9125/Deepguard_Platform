"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Package : engines.video-engine.localization
Layer   : Layer 4 — Video Localization

Exports the VideoLocalizer engine and VideoLocalizationResult data contract.
"""

from .video_localizer import VideoLocalizer
from .result_types import VideoLocalizationResult

__all__ = [
    "VideoLocalizer",
    "VideoLocalizationResult",
]
