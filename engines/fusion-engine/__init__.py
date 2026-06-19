"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Package : engines.fusion-engine
Layer   : Layer 3 — Cross-Modal Fusion Engine

Exports the core orchestrator and data contracts for Layer 3.
"""

from fusion_core.multimodal_fusion_engine import CrossModalFusionEngine
from result_types import (
    FusionLocalizationResult,
    FusionResult,
    FusionTrackResult,
)

__all__ = [
    "CrossModalFusionEngine",
    "FusionLocalizationResult",
    "FusionResult",
    "FusionTrackResult",
]
