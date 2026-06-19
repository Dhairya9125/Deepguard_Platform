"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.preprocessing.scene_detector
Layer   : Layer 1 — Temporal Preprocessing
Task    : Temporal scene segmentation via PySceneDetect

Architecture Reference (DeepGuard_Platform_Architecture.docx — VDS Layer 1):
  - Algorithm     : PySceneDetect ContentDetector (histogram-based hard-cut detection)
  - Input         : Source video file path
  - Output        : List[SceneBoundary] + in-place scene_id stamp on each FramePacket
  - Backend       : PySceneDetect v0.6+ with OpenCV VideoStream backend

Design decisions:
  1. We use ContentDetector (not ThresholdDetector or AdaptiveDetector) because:
       - It is computationally light (frame-level histogram distance, no ML).
       - It is robust for interview/talking-head videos where cuts are hard.
       - It works at the full video path level — no per-frame numpy arrays needed.

  2. PySceneDetect operates on the ORIGINAL video file, not on the sub-sampled
     extracted frames. This ensures scene boundaries are detected at native FPS
     precision. The detected boundaries are then mapped back to the extracted
     frame indices (FramePacket.frame_id space) by timestamp interpolation.

  3. `assign_scene_ids()` stamps each FramePacket.scene_id in-place using
     binary search (bisect) — O(F log S) where F = frames, S = scenes.

  4. If PySceneDetect finds zero cuts (entire video is one scene), a single
     SceneBoundary covering all frames is returned — downstream layers always
     receive at least one scene.

  5. The `min_scene_len` parameter (default 15 frames at native FPS) prevents
     very short flicker scenes from being registered as separate scenes.

Usage:
    from vid_preprocessing.scene_detector import SceneDetector

    detector = SceneDetector(threshold=27.0, min_scene_len=15)
    scenes   = detector.detect("input.mp4")
    detector.assign_scene_ids(frames)   # stamps FramePacket.scene_id in-place
"""

from __future__ import annotations

import bisect
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from .result_types import FramePacket, SceneBoundary

logger = logging.getLogger(__name__)


class SceneDetector:
    """
    PySceneDetect-based video scene segmentation module.

    This is the fourth module in VDS Layer 1.  It analyses the full source
    video to find hard scene cuts, then maps those boundaries onto the list
    of extracted FramePackets produced by VideoIngestor.

    Args:
        threshold (float):
            ContentDetector threshold — mean absolute pixel difference between
            adjacent frames (after HSV histogram comparison).
            Lower  = more sensitive (more scenes detected).
            Higher = less sensitive (fewer scenes detected).
            Default: 27.0 — PySceneDetect's recommended value for general video.
            For interview / single-camera footage, 30–40 reduces false positives.
        min_scene_len (int):
            Minimum scene length in frames (native FPS). Scenes shorter than
            this are merged into the previous scene. Default: 15 (~0.5s at 30fps).
        show_progress (bool):
            If True, PySceneDetect shows a tqdm progress bar during detection.
            Default: False (clean for pipeline use).
    """

    def __init__(
        self,
        threshold:     float = 27.0,
        min_scene_len: int   = 15,
        show_progress: bool  = False,
    ) -> None:
        self.threshold     = threshold
        self.min_scene_len = min_scene_len
        self.show_progress = show_progress

        # Cached result from last detect() call
        self._last_scenes:       Optional[List[SceneBoundary]] = None
        self._last_video_path:   Optional[Path]                = None

        logger.info(
            "SceneDetector init | threshold=%.1f | min_scene_len=%d",
            threshold, min_scene_len,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, video_path: str | Path) -> List[SceneBoundary]:
        """
        Detect scene boundaries in the source video.

        Uses PySceneDetect's ContentDetector on the full-resolution native-FPS
        video. Results are returned in frame-index / timestamp order.

        Args:
            video_path: Path to the source video file.

        Returns:
            List[SceneBoundary] — at least one scene (the entire video) is
            always returned even if no cuts are detected.

        Raises:
            FileNotFoundError : video_path does not exist.
            ImportError       : scenedetect not installed.
            RuntimeError      : PySceneDetect fails to open the video.
        """
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(
                f"[SceneDetector] Video not found: {video_path}"
            )

        logger.info("Running scene detection on: %s", video_path)

        try:
            from scenedetect import (  # noqa: PLC0415
                open_video,
                SceneManager,
            )
            from scenedetect.detectors import ContentDetector  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "scenedetect is required for scene detection.\n"
                "Install with:  pip install scenedetect[opencv]>=0.6.1"
            ) from exc

        # ── Open video ───────────────────────────────────────────────────
        try:
            video = open_video(str(video_path))
        except Exception as exc:
            raise RuntimeError(
                f"[SceneDetector] PySceneDetect failed to open video: {exc}"
            ) from exc

        # ── Detect scenes ─────────────────────────────────────────────────
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(
                threshold=self.threshold,
                min_scene_len=self.min_scene_len,
            )
        )

        scene_manager.detect_scenes(
            video=video,
            show_progress=self.show_progress,
        )

        raw_scenes = scene_manager.get_scene_list()
        logger.info(
            "PySceneDetect found %d scene(s) (threshold=%.1f).",
            len(raw_scenes), self.threshold,
        )

        # ── Convert to SceneBoundary objects ─────────────────────────────
        # raw_scenes: list of (start_timecode, end_timecode) FrameTimecode pairs
        if not raw_scenes:
            # Single scene covers the entire video
            duration_ms = self._get_video_duration_ms(video_path)
            total_frames = self._get_video_frame_count(video_path)
            scenes = [
                SceneBoundary(
                    scene_id=0,
                    start_frame=0,
                    end_frame=max(0, total_frames - 1),
                    start_ms=0.0,
                    end_ms=duration_ms,
                    frame_count=total_frames,
                )
            ]
            logger.info("No cuts detected — single scene covers entire video.")
        else:
            scenes = []
            for idx, (start_tc, end_tc) in enumerate(raw_scenes):
                start_frame = int(start_tc.get_frames())
                end_frame   = max(start_frame, int(end_tc.get_frames()) - 1)
                start_ms    = float(start_tc.get_seconds()) * 1000.0
                end_ms      = float(end_tc.get_seconds())   * 1000.0
                n_frames    = end_frame - start_frame + 1

                scenes.append(
                    SceneBoundary(
                        scene_id=idx,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        frame_count=n_frames,
                    )
                )

            for sc in scenes:
                logger.debug(
                    "  Scene %02d | frames [%d–%d] | %.0f ms – %.0f ms | %d frames",
                    sc.scene_id, sc.start_frame, sc.end_frame,
                    sc.start_ms, sc.end_ms, sc.frame_count,
                )

        # Cache result for assign_scene_ids()
        self._last_scenes     = scenes
        self._last_video_path = video_path
        return scenes

    def assign_scene_ids(
        self,
        frames: List[FramePacket],
        scenes: Optional[List[SceneBoundary]] = None,
    ) -> None:
        """
        Stamp each FramePacket.scene_id in-place by matching its timestamp_ms
        to the scene boundary intervals.

        This bridges the gap between PySceneDetect's native-FPS frame indices
        and the sub-sampled extracted-frame indices in FramePacket.frame_id.
        Matching is done by timestamp (ms) rather than frame index to avoid
        FPS sub-sampling alignment errors.

        Args:
            frames : List of FramePacket from VideoIngestor (timestamp_ms populated).
            scenes : List[SceneBoundary] from detect(). If None, uses the cached
                     result from the last detect() call.

        Raises:
            ValueError: If neither scenes nor a cached result is available.
        """
        if scenes is None:
            scenes = self._last_scenes
        if scenes is None:
            raise ValueError(
                "[SceneDetector] No scene list available. "
                "Call detect() before assign_scene_ids()."
            )

        if not frames:
            logger.info("assign_scene_ids called with empty frame list — nothing to do.")
            return

        # Build a sorted list of scene start timestamps for bisect
        scene_starts_ms = [sc.start_ms for sc in scenes]

        unassigned = 0
        for fp in frames:
            # bisect_right returns the insertion point AFTER any existing entry
            # Subtracting 1 gives the index of the scene whose start ≤ fp.timestamp_ms
            idx = bisect.bisect_right(scene_starts_ms, fp.timestamp_ms) - 1
            idx = max(0, min(idx, len(scenes) - 1))   # clamp to valid range
            fp.scene_id = scenes[idx].scene_id
            if fp.scene_id == -1:
                unassigned += 1

        assigned = len(frames) - unassigned
        logger.info(
            "assign_scene_ids complete | frames=%d | assigned=%d | n_scenes=%d",
            len(frames), assigned, len(scenes),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_video_duration_ms(video_path: Path) -> float:
        """Get video duration in ms using OpenCV (fallback for single-scene case)."""
        import cv2  # noqa: PLC0415
        cap = cv2.VideoCapture(str(video_path))
        fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
        return (n_frame / fps) * 1000.0

    @staticmethod
    def _get_video_frame_count(video_path: Path) -> int:
        """Get total native frame count using OpenCV."""
        import cv2  # noqa: PLC0415
        cap = cv2.VideoCapture(str(video_path))
        n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(0, n)

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def summary(self, scenes: Optional[List[SceneBoundary]] = None) -> str:
        """Return a human-readable summary of detected scenes."""
        sc = scenes or self._last_scenes or []
        lines = [f"SceneDetector Summary ({len(sc)} scene(s)):"]
        for s in sc:
            lines.append(
                f"  [{s.scene_id:02d}] frames {s.start_frame:6d}–{s.end_frame:6d} | "
                f"{s.start_ms/1000:.2f}s – {s.end_ms/1000:.2f}s | "
                f"{s.frame_count} frames"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SceneDetector("
            f"threshold={self.threshold}, "
            f"min_scene_len={self.min_scene_len})"
        )
