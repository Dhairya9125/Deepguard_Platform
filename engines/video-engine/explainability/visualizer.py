"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.explainability.visualizer
Layer   : Layer 5 — Explainability

Generates matplotlib graphs and cv2 heatmap frame overlays.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class VDSVisualizer:
    """Generates visual proof artifacts for VDS explainability."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Use a non-interactive backend for server-side / headless rendering
        plt.switch_backend("Agg")

    def _get_path(self, filename: str) -> Path:
        return self.output_dir / filename

    def plot_lip_sync_timeline(
        self,
        timeline: List[float],
        audio_timestamps: List[Tuple[float, float]],
        fps: float,
    ) -> Optional[Path]:
        """Plots the Cross-Modal AV mismatch timeline."""
        if not timeline:
            return None

        out_path = self._get_path("lip_sync_timeline.png")
        x_axis = [i / fps for i in range(len(timeline))]

        plt.figure(figsize=(10, 4))
        plt.plot(x_axis, timeline, color="blue", label="AV Mismatch Prob")
        plt.axhline(y=0.5, color="red", linestyle="--", label="Mismatch Threshold")

        # Highlight audio anomaly intervals
        for (start_s, end_s) in audio_timestamps:
            plt.axvspan(start_s, end_s, color="red", alpha=0.3, label="Detected Anomaly")

        # Deduplicate legend labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.title("Cross-Modal Synchronization Mismatch")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Mismatch Probability")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

        return out_path

    def plot_rppg_signals(
        self,
        bvp_signal: List[float],
        heart_rate: float,
    ) -> Optional[Path]:
        """Plots the Blood Volume Pulse (BVP) signal from Branch E."""
        if not bvp_signal:
            return None

        out_path = self._get_path("rppg_signal.png")
        x_axis = list(range(len(bvp_signal)))

        plt.figure(figsize=(10, 4))
        plt.plot(x_axis, bvp_signal, color="green", label="Extracted BVP Signal")
        plt.title(f"rPPG Biological Signal Analysis (Est. HR: {heart_rate:.1f} BPM)")
        plt.xlabel("Frame Window")
        plt.ylabel("Normalized BVP Amplitude")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

        return out_path

    def plot_temporal_inconsistencies(
        self,
        fusion_timeline: List[float],
        fps: float,
    ) -> Optional[Path]:
        """Plots the fusion temporal inconsistency forgery probability."""
        if not fusion_timeline:
            return None

        out_path = self._get_path("temporal_inconsistency.png")
        x_axis = [i / fps for i in range(len(fusion_timeline))]

        plt.figure(figsize=(10, 4))
        plt.plot(x_axis, fusion_timeline, color="purple", label="Forgery Probability")
        plt.axhline(y=0.5, color="red", linestyle="--", label="Threshold")

        plt.fill_between(
            x_axis, fusion_timeline, 0.5,
            where=(np.array(fusion_timeline) > 0.5),
            color="red", alpha=0.3, interpolate=True
        )

        plt.title("Temporal Artifact & Motion Inconsistency")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Forgery Probability")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

        return out_path

    def render_frame_heatmaps(
        self,
        preprocessing_result: Any,
        manipulated_regions: Dict[int, List[Tuple[int, int, int, int]]],
        manipulated_frames: List[int],
    ) -> List[Path]:
        """
        Draws highlighted bounding boxes over the original frames for suspicious regions.
        Saves as PNGs in the output directory.
        """
        saved_paths: List[Path] = []
        if not manipulated_frames or not manipulated_regions:
            return saved_paths

        frames_list = getattr(preprocessing_result, "frames", [])
        frame_lookup = {getattr(f, "frame_id", -1): f for f in frames_list}

        for frame_id in manipulated_frames:
            if frame_id not in manipulated_regions:
                continue

            fp = frame_lookup.get(frame_id)
            if fp is None:
                continue

            # Get image data
            img = None
            if getattr(fp, "rgb_array", None) is not None:
                # Convert RGB to BGR for cv2
                img = cv2.cvtColor(fp.rgb_array, cv2.COLOR_RGB2BGR)
            elif getattr(fp, "frame_path", None) is not None:
                img = cv2.imread(str(fp.frame_path))

            if img is None:
                continue

            # Draw semi-transparent red overlay on manipulated regions
            overlay = img.copy()
            for (gx, gy, gw, gh) in manipulated_regions[frame_id]:
                cv2.rectangle(overlay, (gx, gy), (gx + gw, gy + gh), (0, 0, 255), -1)

            # Blend overlay with original image
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

            # Draw bounding box border
            for (gx, gy, gw, gh) in manipulated_regions[frame_id]:
                cv2.rectangle(img, (gx, gy), (gx + gw, gy + gh), (0, 0, 255), 2)
                cv2.putText(
                    img, "ARTIFACT", (gx, max(gy - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )

            out_path = self._get_path(f"heatmap_frame_{frame_id:04d}.png")
            cv2.imwrite(str(out_path), img)
            saved_paths.append(out_path)

        return saved_paths
