"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.explainability.result_types
Layer   : Layer 5 — Explainability

Defines the output data contract for the Explainability Engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ExplainabilityResult:
    """
    Holds the paths to the generated visualizations and the comprehensive forensic summary.

    Attributes:
        heatmap_paths            : List of paths to generated frame heatmaps (suspicious frames).
        lip_sync_graph_path      : Path to the generated AV mismatch timeline plot.
        rppg_graph_path          : Path to the biological signal (pulse) plot.
        temporal_graph_path      : Path to the motion/consistency anomaly plot.
        report_path              : Path to the generated forensic summary report (Markdown/JSON).
        metadata                 : Extra information about the generation process.
    """
    heatmap_paths: List[Path] = field(default_factory=list)
    lip_sync_graph_path: Optional[Path] = None
    rppg_graph_path: Optional[Path] = None
    temporal_graph_path: Optional[Path] = None
    report_path: Optional[Path] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize result to a JSON-safe dictionary."""
        return {
            "heatmap_paths": [str(p) for p in self.heatmap_paths],
            "lip_sync_graph_path": str(self.lip_sync_graph_path) if self.lip_sync_graph_path else None,
            "rppg_graph_path": str(self.rppg_graph_path) if self.rppg_graph_path else None,
            "temporal_graph_path": str(self.temporal_graph_path) if self.temporal_graph_path else None,
            "report_path": str(self.report_path) if self.report_path else None,
            "metadata": self.metadata,
        }
