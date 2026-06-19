"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.explainability.explainability_engine
Layer   : Layer 5 — Explainability

The main orchestrator for generating visualizations and forensic reports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .result_types import ExplainabilityResult
from .visualizer import VDSVisualizer
from .report_generator import ForensicReportGenerator

logger = logging.getLogger(__name__)


class ExplainabilityEngine:
    """
    VDS Layer 5 — Explainability Engine.
    Orchestrates the generation of all visualizations and reports.
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualizer = VDSVisualizer(self.output_dir)
        self.report_generator = ForensicReportGenerator(self.output_dir)
        logger.info(f"ExplainabilityEngine initialized (output_dir: {self.output_dir})")

    def generate_explanation(
        self,
        preprocessing_result: Any,
        branch_e_result: Optional[Any],
        fusion_result: Any,
        localization_result: Any,
    ) -> ExplainabilityResult:
        """
        Consumes preceding layer results to produce visualizations and a report.

        Args:
            preprocessing_result: Layer 1 output (frames).
            branch_e_result     : Layer 2 Branch E output (rPPG).
            fusion_result       : Layer 3 output (verdict, timeline).
            localization_result : Layer 4 output (manipulated frames/regions, audio sync).

        Returns:
            ExplainabilityResult containing paths to generated artifacts.
        """
        fps = 8.0
        if localization_result:
            fps = getattr(localization_result, "metadata", {}).get("fps", 8.0)

        graph_paths: Dict[str, Path] = {}

        # 1. Temporal Inconsistency Graph
        # We take the max forgery probability across all tracks per frame
        max_fusion_timeline: List[float] = []
        track_results = getattr(fusion_result, "track_results", {})
        for track_res in track_results.values():
            loc = getattr(track_res, "localization", None)
            if loc:
                timeline = getattr(loc, "temporal_timeline", [])
                if not max_fusion_timeline:
                    max_fusion_timeline = list(timeline)
                else:
                    for i, val in enumerate(timeline):
                        if i < len(max_fusion_timeline):
                            max_fusion_timeline[i] = max(max_fusion_timeline[i], val)
                        else:
                            max_fusion_timeline.append(val)

        if max_fusion_timeline:
            path = self.visualizer.plot_temporal_inconsistencies(max_fusion_timeline, fps)
            if path:
                graph_paths["temporal"] = path

        # 2. Lip Sync Timeline Graph
        mismatch_timeline = getattr(localization_result, "cross_modal_mismatch_timeline", [])
        audio_timestamps = getattr(localization_result, "audio_timestamps", [])
        if mismatch_timeline:
            path = self.visualizer.plot_lip_sync_timeline(mismatch_timeline, audio_timestamps, fps)
            if path:
                graph_paths["lip_sync"] = path

        # 3. rPPG Biological Signal Graph
        if branch_e_result:
            # Just take the first track's BVP signal for visualization
            e_track_metrics = getattr(branch_e_result, "track_metrics", {})
            if e_track_metrics:
                first_track = list(e_track_metrics.values())[0]
                bvp_signal = getattr(first_track, "bvp_signal", [])
                hr = getattr(first_track, "estimated_heart_rate", 0.0)
                if bvp_signal:
                    path = self.visualizer.plot_rppg_signals(bvp_signal, hr)
                    if path:
                        graph_paths["rppg"] = path

        # 4. Frame Heatmaps
        heatmap_paths: List[Path] = []
        if preprocessing_result and localization_result:
            manipulated_regions = getattr(localization_result, "manipulated_regions", {})
            manipulated_frames = getattr(localization_result, "manipulated_frames", [])
            heatmap_paths = self.visualizer.render_frame_heatmaps(
                preprocessing_result, manipulated_regions, manipulated_frames
            )
            if heatmap_paths:
                graph_paths["heatmaps"] = heatmap_paths[0]  # Just to signal it exists for the report

        # 5. Generate Report
        report_path = self.report_generator.generate_report(
            fusion_result=fusion_result,
            localization_result=localization_result,
            graph_paths=graph_paths,
        )

        # 6. Return Result
        return ExplainabilityResult(
            heatmap_paths=heatmap_paths,
            lip_sync_graph_path=graph_paths.get("lip_sync"),
            rppg_graph_path=graph_paths.get("rppg"),
            temporal_graph_path=graph_paths.get("temporal"),
            report_path=report_path,
            metadata={"fps": fps, "graphs_generated": list(graph_paths.keys())},
        )
