"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Package : engines.video-engine.explainability
Layer   : Layer 5 — Explainability

Exports the ExplainabilityEngine and ExplainabilityResult.
"""

from .explainability_engine import ExplainabilityEngine
from .result_types import ExplainabilityResult
from .visualizer import VDSVisualizer
from .report_generator import ForensicReportGenerator

__all__ = [
    "ExplainabilityEngine",
    "ExplainabilityResult",
    "VDSVisualizer",
    "ForensicReportGenerator",
]
