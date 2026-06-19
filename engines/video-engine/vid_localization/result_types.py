"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.localization.result_types
Layer   : Layer 4 — Video Localization

Defines the output data contracts for Layer 4 Video Localization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any


@dataclass
class VideoLocalizationResult:
    """
    Holds the temporal and spatial localization coordinates of a video deepfake analysis.

    Attributes:
        manipulated_frames              : List of frame indices containing visual or global manipulation.
        manipulated_regions             : Dict mapping frame index to a list of bounding boxes
                                          [x, y, w, h] in global frame coordinates.
        audio_timestamps                : List of (start_sec, end_sec) intervals where audio manipulation
                                          or severe lip-sync mismatches were detected.
        cross_modal_mismatch_timeline   : Frame-aligned sync mismatch values in [0, 1].
        metadata                        : Dict containing video properties (fps, duration, etc.).
    """
    manipulated_frames: List[int] = field(default_factory=list)
    manipulated_regions: Dict[int, List[Tuple[int, int, int, int]]] = field(default_factory=dict)
    audio_timestamps: List[Tuple[float, float]] = field(default_factory=list)
    cross_modal_mismatch_timeline: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize localization result to a JSON-safe plain dictionary."""
        return {
            "manipulated_frames": self.manipulated_frames,
            "manipulated_regions": {
                str(fid): [list(bbox) for bbox in bboxes]
                for fid, bboxes in self.manipulated_regions.items()
            },
            "audio_timestamps": [
                [round(start, 3), round(end, 3)] for start, end in self.audio_timestamps
            ],
            "cross_modal_mismatch_timeline": [
                round(float(v), 4) for v in self.cross_modal_mismatch_timeline
            ],
            "metadata": self.metadata,
        }
