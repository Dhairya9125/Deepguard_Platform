"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.result_types_l2
Layer   : Layer 2 — Temporal Feature Extraction
Branch  : Branch A — Frame-Level Visual Analysis

Defines the result types produced by Branch A and consumed by:
  - Branch B (Temporal Consistency Analysis)
  - VDS Layer 3 (Temporal Fusion Engine)
  - VDS trust layer / reporting layer

Data hierarchy:
    VideoPreprocessingResult (Layer 1 output)
           │
           ▼ Branch A processes each TrackedFace.aligned_crop
    FrameVisualResult          ← one per tracked face per frame
           │
           ▼ aggregated across all frames per track
    BranchAResult              ← top-level Branch A output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Per-face, per-frame result
# ---------------------------------------------------------------------------

@dataclass
class FrameVisualResult:
    """
    IDS output for a single tracked face in a single video frame.

    This is the atomic unit of Branch A output. It contains the complete
    IDS verdict — identical to what the image-engine would produce for a
    standalone still image — but tagged with the video-domain identifiers
    (track_id, frame_id, timestamp_ms) so it can be assembled into timelines
    for temporal analysis in Branch B.

    Attributes:
        track_id            : Stable track ID from VDS Layer 1 FaceTracker.
        frame_id            : Zero-based frame index from FramePacket.
        timestamp_ms        : Timestamp of the parent frame in milliseconds.

        fake_probability    : IDSFusionEngine output — scalar in [0, 1].
                              0.0 = definitely real, 1.0 = definitely fake.
        ood_score           : Out-of-Distribution score — AutoEncoder MSE
                              reconstruction loss. High = unusual input.

        branch_embeddings   : Dict of the 5 IDS branch feature vectors.
                              Keys: 'spatial', 'frequency', 'discrepancy',
                                    'noise', 'fingerprint'.
                              Values: (512,) float32 numpy arrays.
        artifact_embedding  : The full fused representation from IDSFusionEngine
                              — (2560,) float32 numpy array. Used by Branch B
                              for temporal embedding distance analysis.

        manipulation_heatmap: (224, 224) float32 array in [0, 1].
                              Pixel-wise probability of manipulation.
                              Produced by IDSFusionEngine heatmap decoder.
        manipulated_mask    : (224, 224) uint8 array (0 or 255).
                              Binary mask of AI-manipulated regions after
                              LocalizationEngine thresholding.
        blend_boundaries    : (224, 224) uint8 array.
                              Morphological gradient highlighting the seams
                              between real and synthetic regions.
        bounding_boxes      : List of (x, y, w, h) tuples — bounding boxes
                              around detected manipulated regions.

        detection_success   : False if IDS processing failed on this crop
                              (e.g. degenerate image, model exception).
                              When False, all array fields are None.
        failure_reason      : Human-readable reason if detection_success=False.
    """
    track_id:             str
    frame_id:             int
    timestamp_ms:         float

    # IDS verdicts
    fake_probability:     float                         = 0.0
    ood_score:            float                         = 0.0

    # IDS embeddings
    branch_embeddings:    Dict[str, np.ndarray]         = field(default_factory=dict)
    artifact_embedding:   Optional[np.ndarray]          = field(default=None, repr=False)

    # IDS spatial outputs
    manipulation_heatmap: Optional[np.ndarray]          = field(default=None, repr=False)
    manipulated_mask:     Optional[np.ndarray]          = field(default=None, repr=False)
    blend_boundaries:     Optional[np.ndarray]          = field(default=None, repr=False)
    bounding_boxes:       List[Tuple[int, int, int, int]] = field(default_factory=list)

    # Status
    detection_success:    bool                          = True
    failure_reason:       Optional[str]                 = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def is_fake(self) -> bool:
        """True if fake_probability > 0.5 and detection succeeded."""
        return self.detection_success and self.fake_probability > 0.5

    @property
    def n_manipulated_regions(self) -> int:
        """Number of distinct manipulated bounding boxes detected."""
        return len(self.bounding_boxes)

    @property
    def has_spatial_outputs(self) -> bool:
        return self.manipulation_heatmap is not None

    def to_dict(self) -> Dict:
        """
        Serialise to a plain dict (excluding large numpy arrays).
        JSON-safe — suitable for logging and reporting.
        """
        return {
            "track_id":              self.track_id,
            "frame_id":              self.frame_id,
            "timestamp_ms":          round(self.timestamp_ms, 2),
            "fake_probability":      round(self.fake_probability, 6),
            "ood_score":             round(self.ood_score, 6),
            "is_fake":               self.is_fake,
            "n_manipulated_regions": self.n_manipulated_regions,
            "bounding_boxes":        [list(bb) for bb in self.bounding_boxes],
            "branch_embedding_keys": list(self.branch_embeddings.keys()),
            "artifact_embedding_shape": (
                list(self.artifact_embedding.shape)
                if self.artifact_embedding is not None else None
            ),
            "heatmap_shape": (
                list(self.manipulation_heatmap.shape)
                if self.manipulation_heatmap is not None else None
            ),
            "detection_success":     self.detection_success,
            "failure_reason":        self.failure_reason,
        }


# ---------------------------------------------------------------------------
# Branch A aggregate result
# ---------------------------------------------------------------------------

@dataclass
class BranchAResult:
    """
    Top-level output of VDS Layer 2, Branch A — Frame-Level Visual Analysis.

    Contains a per-track timeline of FrameVisualResult objects covering all
    frames where that track was active and the crop was successfully processed.

    This is the sole input to:
      - Branch B (Temporal Consistency Analysis) — consumes `frame_results`
      - VDS Layer 3 Temporal Fusion Engine

    Attributes:
        frame_results       : Dict[track_id → ordered List[FrameVisualResult]].
                              Ordered by frame_id (ascending).
        n_frames_processed  : Total number of (track, frame) pairs attempted.
        n_faces_total       : Total FrameVisualResult objects produced
                              (success + failure).
        n_faces_successful  : Successfully processed FrameVisualResult objects.
        processing_warnings : Non-fatal warnings emitted during Branch A.
    """
    frame_results:        Dict[str, List[FrameVisualResult]]
    n_frames_processed:   int                 = 0
    n_faces_total:        int                 = 0
    n_faces_successful:   int                 = 0
    processing_warnings:  List[str]           = field(default_factory=list)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def unique_track_ids(self) -> List[str]:
        return sorted(self.frame_results.keys())

    @property
    def n_tracks(self) -> int:
        return len(self.frame_results)

    def results_for_track(self, track_id: str) -> List[FrameVisualResult]:
        """Return the ordered FrameVisualResult timeline for a track."""
        return self.frame_results.get(track_id, [])

    def fake_probability_timeline(self, track_id: str) -> List[float]:
        """
        Return fake_probability values (successful results only) for a track,
        ordered by frame_id. Ready for Branch B temporal analysis.
        """
        return [
            r.fake_probability
            for r in self.results_for_track(track_id)
            if r.detection_success
        ]

    def artifact_embedding_timeline(self, track_id: str) -> List[np.ndarray]:
        """
        Return artifact_embedding arrays for a track (successful results only).
        Shape per entry: (2560,). Used by Branch B for temporal drift detection.
        """
        return [
            r.artifact_embedding
            for r in self.results_for_track(track_id)
            if r.detection_success and r.artifact_embedding is not None
        ]

    def mean_fake_probability(self, track_id: str) -> Optional[float]:
        """Mean fake_probability across all successful frames for a track."""
        timeline = self.fake_probability_timeline(track_id)
        if not timeline:
            return None
        return float(sum(timeline) / len(timeline))

    def summary(self) -> Dict:
        """
        Return a JSON-serialisable summary of Branch A results.
        """
        track_summaries = {}
        for tid in self.unique_track_ids:
            results = self.results_for_track(tid)
            successful = [r for r in results if r.detection_success]
            probs = [r.fake_probability for r in successful]
            track_summaries[tid] = {
                "n_frames":           len(results),
                "n_successful":       len(successful),
                "mean_fake_prob":     round(sum(probs) / len(probs), 4) if probs else None,
                "max_fake_prob":      round(max(probs), 4) if probs else None,
                "min_fake_prob":      round(min(probs), 4) if probs else None,
                "verdict":            "FAKE" if (probs and sum(probs)/len(probs) > 0.5) else "REAL",
            }

        return {
            "n_tracks":             self.n_tracks,
            "n_frames_processed":   self.n_frames_processed,
            "n_faces_total":        self.n_faces_total,
            "n_faces_successful":   self.n_faces_successful,
            "n_warnings":           len(self.processing_warnings),
            "warnings":             self.processing_warnings,
            "track_summaries":      track_summaries,
        }
