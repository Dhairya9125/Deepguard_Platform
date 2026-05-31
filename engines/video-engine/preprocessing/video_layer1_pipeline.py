"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.preprocessing.video_layer1_pipeline
Layer   : Layer 1 — Temporal Preprocessing
Task    : Orchestrator — wires VideoIngestor → FaceTracker → LandmarkTracer
          → SceneDetector into a single `.run()` call

Architecture Reference (DeepGuard_Platform_Architecture.docx — VDS Layer 1):
  The pipeline executes four sequential stages:

    Stage 1 | VideoIngestor   → frames (RGB arrays) + audio WAV + VideoMetadata
    Stage 2 | SceneDetector   → scene boundaries (run on source video, not frames)
               ↕  assign_scene_ids stamps each FramePacket.scene_id in-place
    Stage 3 | FaceTracker     → TrackedFace list per frame (stable track IDs)
    Stage 4 | LandmarkTracer  → 478-pt landmark timeline per track + EMA smoothing
    ───────────────────────────────────────────────────────────────────────────
    Output  | VideoPreprocessingResult — sole input to VDS Layer 2

  Scene detection runs before face tracking (Stage 2 before Stage 3) so that
  downstream layers can group landmark timelines per scene. The source video
  file is only read twice: once by VideoIngestor (frame pipe), once by
  PySceneDetect. All other processing operates on in-memory numpy arrays.

Performance note:
  On a modern CPU (8-core, no GPU), expect approximately:
    - 8 fps extraction  : ~0.3× real-time I/O cost
    - OpenCV DNN tracker: ~50 fps detection on 720p crops
    - MediaPipe FaceMesh: ~20 fps per track per 224×224 crop
    - Scene detection   : ~1× real-time for the PySceneDetect pass

Usage:
    from preprocessing.video_layer1_pipeline import VideoLayer1Pipeline

    pipeline = VideoLayer1Pipeline(
        target_fps=8.0,
        tracker_backend="dnn",
        save_frames=False,
    )
    result = pipeline.run(
        video_path="interview.mp4",
        workspace_dir="tmp/run_01",
    )

    print(result.summary())

    # Access per-track landmark timeline
    for track_id, lm_list in result.landmarks.items():
        print(track_id, len(lm_list), "frames with landmarks")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from .face_tracker     import FaceTracker
from .landmark_tracer  import LandmarkTracer
from .result_types     import VideoPreprocessingResult
from .scene_detector   import SceneDetector
from .video_ingestor   import VideoIngestor

logger = logging.getLogger(__name__)


class VideoLayer1Pipeline:
    """
    VDS Layer 1 Orchestrator.

    Wires the four preprocessing modules into a single `.run()` method that
    returns a `VideoPreprocessingResult` ready for Layer 2 consumption.

    Args:
        target_fps (float):
            Frame extraction rate. 0.0 = native video FPS. Default: 8.0.
        audio_sample_rate (int):
            WAV output sample rate. Default: 16 000 Hz.
        save_frames (bool):
            Flush extracted frames to JPEG under workspace_dir/frames/.
            Default: False (keep in memory only).
        jpeg_quality (int):
            JPEG compression quality when save_frames=True. Default: 95.
        tracker_backend (str):
            'dnn' (fast, OpenCV DNN) or 'retinaface' (accurate, InsightFace).
            Default: 'dnn'.
        tracker_max_age (int):
            DeepSORT max consecutive missed frames before retiring a track.
            Default: 30.
        tracker_min_hits (int):
            Minimum detections before a track is confirmed. Default: 3.
        tracker_confidence (float):
            Minimum face detection confidence. Default: 0.70.
        scene_threshold (float):
            PySceneDetect ContentDetector threshold. Default: 27.0.
        scene_min_len (int):
            Minimum scene length in native frames. Default: 15.
        landmark_ema_alpha (float):
            EMA smoothing factor for landmarks [0, 1]. Default: 0.3.
        landmark_refine (bool):
            MediaPipe refine_landmarks (adds iris, 478 vs 468 pts). Default: True.
        ffmpeg_loglevel (str):
            FFmpeg subprocess log level. Default: 'error'.
    """

    def __init__(
        self,
        target_fps:           float = 8.0,
        audio_sample_rate:    int   = 16_000,
        save_frames:          bool  = False,
        jpeg_quality:         int   = 95,
        tracker_backend:      str   = "dnn",
        tracker_max_age:      int   = 30,
        tracker_min_hits:     int   = 3,
        tracker_confidence:   float = 0.70,
        scene_threshold:      float = 27.0,
        scene_min_len:        int   = 15,
        landmark_ema_alpha:   float = 0.3,
        landmark_refine:      bool  = True,
        ffmpeg_loglevel:      str   = "error",
    ) -> None:

        self.target_fps = target_fps

        # ── Instantiate modules ─────────────────────────────────────────
        self.ingestor = VideoIngestor(
            target_fps=target_fps,
            audio_sample_rate=audio_sample_rate,
            save_frames=save_frames,
            jpeg_quality=jpeg_quality,
            ffmpeg_loglevel=ffmpeg_loglevel,
        )

        self.tracker = FaceTracker(
            backend=tracker_backend,
            confidence_threshold=tracker_confidence,
            max_age=tracker_max_age,
            min_hits=tracker_min_hits,
        )

        self.landmark_tracer = LandmarkTracer(
            refine_landmarks=landmark_refine,
            ema_alpha=landmark_ema_alpha,
        )

        self.scene_detector = SceneDetector(
            threshold=scene_threshold,
            min_scene_len=scene_min_len,
        )

        logger.info(
            "VideoLayer1Pipeline initialised | fps=%.1f | tracker=%s | "
            "scene_thresh=%.1f | ema=%.2f",
            target_fps, tracker_backend, scene_threshold, landmark_ema_alpha,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        video_path:    str | Path,
        workspace_dir: str | Path,
    ) -> VideoPreprocessingResult:
        """
        Execute the full Layer 1 pipeline on a single video file.

        Pipeline stages (in order):
            1. VideoIngestor   — frame & audio extraction
            2. SceneDetector   — scene boundary detection + scene_id assignment
            3. FaceTracker     — multi-face tracking with stable track IDs
            4. LandmarkTracer  — 478-pt landmark extraction + EMA smoothing

        Args:
            video_path    : Path to the source video file.
            workspace_dir : Output directory for audio.wav and optional frames/.
                            Created automatically if it does not exist.

        Returns:
            VideoPreprocessingResult — unified Layer 1 output.
        """
        video_path    = Path(video_path).resolve()
        workspace_dir = Path(workspace_dir).resolve()
        warnings: list[str] = []

        pipeline_start = time.perf_counter()
        logger.info(
            "═══════════════════════════════════════════════════════════════"
        )
        logger.info("VDS Layer 1 pipeline START | video=%s", video_path.name)
        logger.info(
            "═══════════════════════════════════════════════════════════════"
        )

        # ────────────────────────────────────────────────────────────────
        # Stage 1 — Frame & Audio Extraction
        # ────────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        logger.info("── Stage 1: Frame & Audio Extraction ─────────────────────")

        try:
            metadata, frames = self.ingestor.ingest(
                video_path=video_path,
                workspace_dir=workspace_dir,
            )
        except Exception as exc:
            logger.exception("Stage 1 FAILED: %s", exc)
            raise

        if not metadata.has_audio:
            warnings.append("No audio stream found — audio extraction skipped.")

        logger.info(
            "Stage 1 done in %.2fs | frames=%d | audio=%s",
            time.perf_counter() - t0, len(frames), metadata.has_audio,
        )

        if not frames:
            logger.warning("No frames extracted — returning minimal result.")
            return VideoPreprocessingResult(
                metadata=metadata,
                frames=[],
                scenes=[],
                landmarks={},
                processing_ok=False,
                warnings=warnings + ["No frames could be extracted from video."],
            )

        # ────────────────────────────────────────────────────────────────
        # Stage 2 — Scene Detection + scene_id assignment
        # ────────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        logger.info("── Stage 2: Scene Detection ──────────────────────────────")

        try:
            scenes = self.scene_detector.detect(video_path)
            self.scene_detector.assign_scene_ids(frames, scenes)
        except Exception as exc:
            logger.warning(
                "Scene detection failed (non-fatal): %s — treating as single scene.",
                exc,
            )
            warnings.append(f"Scene detection failed: {exc}. Single scene assumed.")
            # Assign all frames to scene 0
            for fp in frames:
                fp.scene_id = 0
            from .result_types import SceneBoundary  # noqa: PLC0415
            scenes = [
                SceneBoundary(
                    scene_id=0,
                    start_frame=0,
                    end_frame=len(frames) - 1,
                    start_ms=frames[0].timestamp_ms,
                    end_ms=frames[-1].timestamp_ms,
                    frame_count=len(frames),
                )
            ]

        logger.info(
            "Stage 2 done in %.2fs | scenes=%d",
            time.perf_counter() - t0, len(scenes),
        )
        logger.info(self.scene_detector.summary(scenes))

        # ────────────────────────────────────────────────────────────────
        # Stage 3 — Face Tracking
        # ────────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        logger.info("── Stage 3: Face Tracking ────────────────────────────────")

        try:
            self.tracker.reset()
            self.tracker.process_video(frames)
        except Exception as exc:
            logger.warning("Face tracking failed (non-fatal): %s", exc)
            warnings.append(f"Face tracking error: {exc}")

        # Collect unique track IDs
        unique_track_ids = sorted(
            {tf.track_id for fp in frames for tf in fp.tracked_faces}
        )
        total_face_instances = sum(len(fp.tracked_faces) for fp in frames)

        logger.info(
            "Stage 3 done in %.2fs | unique_tracks=%d | face_instances=%d",
            time.perf_counter() - t0, len(unique_track_ids), total_face_instances,
        )

        # ────────────────────────────────────────────────────────────────
        # Stage 4 — Landmark Tracing
        # ────────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        logger.info("── Stage 4: Landmark Tracing ─────────────────────────────")

        landmarks = {}
        try:
            self.landmark_tracer.reset()
            landmarks = self.landmark_tracer.process_video(frames)
        except Exception as exc:
            logger.warning("Landmark tracing failed (non-fatal): %s", exc)
            warnings.append(f"Landmark tracing error: {exc}")

        lm_success = sum(
            r.detection_success
            for v in landmarks.values()
            for r in v
        )
        lm_total = sum(len(v) for v in landmarks.values())

        logger.info(
            "Stage 4 done in %.2fs | lm_results=%d | successful=%d",
            time.perf_counter() - t0, lm_total, lm_success,
        )

        # ────────────────────────────────────────────────────────────────
        # Assemble result
        # ────────────────────────────────────────────────────────────────
        total_elapsed = time.perf_counter() - pipeline_start

        result = VideoPreprocessingResult(
            metadata=metadata,
            frames=frames,
            scenes=scenes,
            landmarks=landmarks,
            unique_track_ids=unique_track_ids,
            processing_ok=True,
            warnings=warnings,
        )

        logger.info(
            "═══════════════════════════════════════════════════════════════"
        )
        logger.info(
            "VDS Layer 1 pipeline COMPLETE | elapsed=%.2fs | %s",
            total_elapsed, video_path.name,
        )
        logger.info(
            "  frames=%d | scenes=%d | tracks=%d | lm_results=%d | warnings=%d",
            result.n_frames, result.n_scenes, result.n_unique_tracks,
            lm_total, len(warnings),
        )
        logger.info(
            "═══════════════════════════════════════════════════════════════"
        )

        return result

    # ------------------------------------------------------------------
    # Context manager support (closes MediaPipe gracefully)
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release MediaPipe FaceMesh resources."""
        self.landmark_tracer.close()
        logger.info("VideoLayer1Pipeline closed.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self) -> str:
        return (
            f"VideoLayer1Pipeline("
            f"target_fps={self.target_fps}, "
            f"tracker={self.tracker.backend!r})"
        )
