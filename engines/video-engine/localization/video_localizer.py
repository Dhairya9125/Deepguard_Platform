"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.localization.video_localizer
Layer   : Layer 4 — Video Localization

Orchestrates the spatial and temporal localization of deepfake artifacts.
Translates Layer 3 fusion results and Layer 2 feature timelines into concrete
spatial coordinates, frame indices, audio intervals, and mismatch timelines.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .result_types import VideoLocalizationResult

logger = logging.getLogger(__name__)


class VideoLocalizer:
    """
    VDS Layer 4 — Video Localization Engine.

    Responsible for:
      1. Identifying manipulated frames from the Layer 3 fusion timeline.
      2. Mapping crop-space bounding boxes back to global frame coordinates.
      3. Extracting audio manipulation timestamps from sync mismatch timelines.
      4. Building frame-aligned cross-modal mismatch timelines.
    """

    def __init__(
        self,
        visual_threshold: float = 0.5,
        audio_threshold: float = 0.5,
        min_audio_interval_s: float = 0.1,
    ) -> None:
        """
        Args:
            visual_threshold    : Threshold above which a frame is flagged as manipulated.
            audio_threshold     : Threshold above which a frame contributes to an audio
                                  manipulation interval.
            min_audio_interval_s: Minimum duration (seconds) for an audio interval to be
                                  reported. Shorter intervals are discarded as noise.
        """
        self.visual_threshold = visual_threshold
        self.audio_threshold = audio_threshold
        self.min_audio_interval_s = min_audio_interval_s

        logger.info(
            "VideoLocalizer initialized | visual_threshold=%.2f | audio_threshold=%.2f | "
            "min_audio_interval_s=%.2f",
            visual_threshold, audio_threshold, min_audio_interval_s,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def localize(
        self,
        preprocessing_result: Optional[Any] = None,
        fusion_result: Optional[Any] = None,
        branch_a_result: Optional[Any] = None,
        branch_d_result: Optional[Any] = None,
    ) -> VideoLocalizationResult:
        """
        Process Layer 1/2/3 outputs to produce the final VideoLocalizationResult.

        Args:
            preprocessing_result: Layer 1 VideoPreprocessingResult (frames, metadata, bboxes).
            fusion_result       : Layer 3 FusionResult (per-track timelines, attribution).
            branch_a_result     : Layer 2 Branch A BranchAResult (per-frame IDS heatmaps / bboxes).
            branch_d_result     : Layer 2 Branch D BranchDResult (sync metrics per track).

        Returns:
            VideoLocalizationResult
        """
        warnings: List[str] = []

        # Extract video metadata
        fps = 8.0  # default
        duration_s = 0.0
        frame_width = 0
        frame_height = 0
        n_frames = 0

        if preprocessing_result is not None:
            meta = getattr(preprocessing_result, "metadata", None)
            if meta is not None:
                fps = getattr(meta, "target_fps", 8.0)
                duration_s = getattr(meta, "duration_s", 0.0)
                frame_width = getattr(meta, "width", 0)
                frame_height = getattr(meta, "height", 0)
            n_frames = getattr(preprocessing_result, "n_frames", 0)
        else:
            warnings.append("Missing Layer 1 preprocessing result.")

        if fusion_result is None:
            warnings.append("Missing Layer 3 fusion result.")
            return VideoLocalizationResult(
                metadata={"fps": fps, "duration_s": duration_s, "warnings": warnings},
            )

        # ------------------------------------------------------------------
        # 1. Manipulated Frames — from Layer 3 fusion localization timelines
        # ------------------------------------------------------------------
        manipulated_frames: List[int] = []
        # Aggregate across all tracks — a frame is flagged if ANY track
        # exceeds the visual threshold at that frame index.
        max_timeline_len = 0
        all_timelines: Dict[str, List[float]] = {}

        track_results = getattr(fusion_result, "track_results", {})
        for track_id, track_res in track_results.items():
            loc = getattr(track_res, "localization", None)
            if loc is not None:
                timeline = getattr(loc, "temporal_timeline", [])
                if timeline:
                    all_timelines[track_id] = timeline
                    max_timeline_len = max(max_timeline_len, len(timeline))

        # Build a frame-level aggregate (max across tracks)
        frame_scores: List[float] = [0.0] * max_timeline_len
        for timeline in all_timelines.values():
            for i, score in enumerate(timeline):
                if i < len(frame_scores):
                    frame_scores[i] = max(frame_scores[i], float(score))

        for frame_idx, score in enumerate(frame_scores):
            if score >= self.visual_threshold:
                manipulated_frames.append(frame_idx)

        # ------------------------------------------------------------------
        # 2. Manipulated Regions — inverse coordinate mapping
        # ------------------------------------------------------------------
        manipulated_regions: Dict[int, List[Tuple[int, int, int, int]]] = {}

        if branch_a_result is not None and preprocessing_result is not None:
            frame_results_map = getattr(branch_a_result, "frame_results", {})
            frames_list = getattr(preprocessing_result, "frames", [])

            # Build a frame_id → FramePacket lookup
            frame_lookup: Dict[int, Any] = {}
            for fp in frames_list:
                frame_lookup[getattr(fp, "frame_id", -1)] = fp

            for track_id, frame_vis_results in frame_results_map.items():
                for fvr in frame_vis_results:
                    frame_id = getattr(fvr, "frame_id", -1)

                    # Only process frames that are flagged as manipulated
                    if frame_id not in manipulated_frames:
                        continue

                    # Get the crop-space bounding boxes from Branch A
                    crop_bboxes = getattr(fvr, "bounding_boxes", [])
                    if not crop_bboxes:
                        continue

                    # Find the face bbox in the original frame
                    face_bbox = self._find_face_bbox(
                        frame_lookup.get(frame_id), track_id
                    )
                    if face_bbox is None:
                        continue

                    # Map each crop-space bbox to global frame coordinates
                    global_bboxes = []
                    for crop_bbox in crop_bboxes:
                        global_bbox = self._map_crop_to_global(
                            face_bbox=face_bbox,
                            crop_bbox=crop_bbox,
                            crop_size=224,
                            frame_width=frame_width,
                            frame_height=frame_height,
                        )
                        global_bboxes.append(global_bbox)

                    if global_bboxes:
                        if frame_id not in manipulated_regions:
                            manipulated_regions[frame_id] = []
                        manipulated_regions[frame_id].extend(global_bboxes)
        else:
            if branch_a_result is None:
                warnings.append("Missing Branch A result — spatial regions not available.")

        # ------------------------------------------------------------------
        # 3. Cross-Modal Mismatch Timeline — from Branch D sync metrics
        # ------------------------------------------------------------------
        cross_modal_mismatch_timeline: List[float] = []

        if branch_d_result is not None:
            # Build per-frame mismatch from Branch D track metrics
            d_track_metrics = getattr(branch_d_result, "track_metrics", {})
            if d_track_metrics:
                # Take the maximum sync mismatch across all tracks
                # Each track provides scalar metrics; we broadcast them across
                # the frame timeline to form a uniform per-frame signal
                mismatch_values: List[float] = []
                for track_id, metrics in d_track_metrics.items():
                    # Combine phoneme-viseme inconsistency + emotion mismatch
                    pv = float(getattr(metrics, "phoneme_viseme_inconsistency", 0.0))
                    em = float(getattr(metrics, "emotion_mismatch_score", 0.0))
                    av = float(getattr(metrics, "av_hubert_score", 0.0))
                    # Composite mismatch: weighted average
                    composite = 0.4 * pv + 0.3 * em + 0.3 * av
                    mismatch_values.append(composite)

                if mismatch_values:
                    # Broadcast the max composite mismatch across all frames
                    peak_mismatch = max(mismatch_values)
                    timeline_len = max_timeline_len if max_timeline_len > 0 else n_frames
                    cross_modal_mismatch_timeline = [peak_mismatch] * timeline_len
        else:
            warnings.append("Missing Branch D result — cross-modal mismatch timeline not available.")

        # If we have per-frame fusion timelines and per-track audio attribution,
        # modulate the mismatch timeline with frame-level fusion scores
        if cross_modal_mismatch_timeline and frame_scores:
            for i in range(min(len(cross_modal_mismatch_timeline), len(frame_scores))):
                # Scale mismatch by the fusion-level forgery probability
                audio_attrib = 0.0
                for track_res in track_results.values():
                    loc = getattr(track_res, "localization", None)
                    if loc is not None:
                        ma = getattr(loc, "modality_attribution", {})
                        audio_attrib = max(audio_attrib, float(ma.get("audio", 0.0)))
                cross_modal_mismatch_timeline[i] = min(
                    1.0,
                    cross_modal_mismatch_timeline[i] * (0.5 + 0.5 * audio_attrib),
                )

        # ------------------------------------------------------------------
        # 4. Audio Timestamps — group contiguous frames above threshold
        # ------------------------------------------------------------------
        audio_timestamps: List[Tuple[float, float]] = []

        if cross_modal_mismatch_timeline:
            audio_timestamps = self._extract_audio_intervals(
                timeline=cross_modal_mismatch_timeline,
                fps=fps,
                threshold=self.audio_threshold,
                min_duration_s=self.min_audio_interval_s,
            )
        elif frame_scores:
            # Fallback: use frame_scores with audio attribution weighting
            audio_attribution_max = 0.0
            for track_res in track_results.values():
                loc = getattr(track_res, "localization", None)
                if loc is not None:
                    ma = getattr(loc, "modality_attribution", {})
                    audio_attribution_max = max(
                        audio_attribution_max, float(ma.get("audio", 0.0))
                    )

            if audio_attribution_max > 0.3:
                audio_weighted = [
                    s * audio_attribution_max for s in frame_scores
                ]
                audio_timestamps = self._extract_audio_intervals(
                    timeline=audio_weighted,
                    fps=fps,
                    threshold=self.audio_threshold,
                    min_duration_s=self.min_audio_interval_s,
                )

        return VideoLocalizationResult(
            manipulated_frames=manipulated_frames,
            manipulated_regions=manipulated_regions,
            audio_timestamps=audio_timestamps,
            cross_modal_mismatch_timeline=cross_modal_mismatch_timeline,
            metadata={
                "fps": fps,
                "duration_s": duration_s,
                "frame_width": frame_width,
                "frame_height": frame_height,
                "n_frames_analyzed": max_timeline_len,
                "visual_threshold": self.visual_threshold,
                "audio_threshold": self.audio_threshold,
                "warnings": warnings,
            },
        )

    # ------------------------------------------------------------------
    # Coordinate Mapping Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_face_bbox(
        frame_packet: Optional[Any], track_id: str
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Find the face bounding box for a given track_id in a FramePacket.

        Returns:
            (x1, y1, x2, y2) in global frame pixel coordinates, or None.
        """
        if frame_packet is None:
            return None

        tracked_faces = getattr(frame_packet, "tracked_faces", [])
        for face in tracked_faces:
            if getattr(face, "track_id", None) == track_id:
                return getattr(face, "bbox", None)

        return None

    @staticmethod
    def _map_crop_to_global(
        face_bbox: Tuple[int, int, int, int],
        crop_bbox: Tuple[int, int, int, int],
        crop_size: int = 224,
        frame_width: int = 0,
        frame_height: int = 0,
    ) -> Tuple[int, int, int, int]:
        """
        Inverse-transform a crop-space bounding box (x, y, w, h) in 224×224 space
        back to global frame coordinates.

        The aligned face crop is generated from face_bbox = (x1, y1, x2, y2) in the
        original frame and resized to (crop_size × crop_size). To reverse this:
          global_x = face_x1 + (crop_x / crop_size) * face_width
          global_y = face_y1 + (crop_y / crop_size) * face_height

        Args:
            face_bbox  : (x1, y1, x2, y2) face bounding box in global frame coordinates.
            crop_bbox  : (x, y, w, h) bounding box in the 224×224 crop space.
            crop_size  : Size of the crop (default 224).
            frame_width : Frame width for clamping.
            frame_height: Frame height for clamping.

        Returns:
            (x, y, w, h) in global frame coordinates.
        """
        fx1, fy1, fx2, fy2 = face_bbox
        face_w = fx2 - fx1
        face_h = fy2 - fy1

        cx, cy, cw, ch = crop_bbox

        # Scale factors from crop space to face bbox space
        scale_x = face_w / crop_size if crop_size > 0 else 1.0
        scale_y = face_h / crop_size if crop_size > 0 else 1.0

        # Map to global coordinates
        gx = int(fx1 + cx * scale_x)
        gy = int(fy1 + cy * scale_y)
        gw = max(1, int(cw * scale_x))
        gh = max(1, int(ch * scale_y))

        # Clamp to frame boundaries
        if frame_width > 0:
            gx = max(0, min(gx, frame_width - 1))
            gw = min(gw, frame_width - gx)
        if frame_height > 0:
            gy = max(0, min(gy, frame_height - 1))
            gh = min(gh, frame_height - gy)

        return (gx, gy, gw, gh)

    @staticmethod
    def _extract_audio_intervals(
        timeline: List[float],
        fps: float,
        threshold: float = 0.5,
        min_duration_s: float = 0.1,
    ) -> List[Tuple[float, float]]:
        """
        Group contiguous frames exceeding a threshold into (start_sec, end_sec) intervals.

        Args:
            timeline       : Frame-aligned mismatch scores.
            fps            : Frames per second (used to convert indices to seconds).
            threshold      : Minimum score for a frame to be included.
            min_duration_s : Minimum interval duration to keep (filters noise).

        Returns:
            List of (start_sec, end_sec) tuples.
        """
        if not timeline or fps <= 0:
            return []

        intervals: List[Tuple[float, float]] = []
        in_interval = False
        start_idx = 0

        for i, score in enumerate(timeline):
            if score >= threshold:
                if not in_interval:
                    start_idx = i
                    in_interval = True
            else:
                if in_interval:
                    start_s = start_idx / fps
                    end_s = i / fps
                    if (end_s - start_s) >= min_duration_s:
                        intervals.append((start_s, end_s))
                    in_interval = False

        # Close any trailing interval
        if in_interval:
            start_s = start_idx / fps
            end_s = len(timeline) / fps
            if (end_s - start_s) >= min_duration_s:
                intervals.append((start_s, end_s))

        return intervals
