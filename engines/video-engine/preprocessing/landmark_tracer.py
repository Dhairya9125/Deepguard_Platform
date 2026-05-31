"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.preprocessing.landmark_tracer
Layer   : Layer 1 — Temporal Preprocessing
Task    : Dense 478-point 3D facial landmark extraction & temporal smoothing
          via MediaPipe FaceMesh

Architecture Reference (DeepGuard_Platform_Architecture.docx — VDS Layer 1):
  - Landmark source   : MediaPipe FaceMesh — 478 3D landmarks (x, y, z) per face
  - Input             : Aligned face crop (RGB uint8) from TrackedFace.aligned_crop
  - Temporal smoothing: Per-track Exponential Moving Average (EMA, α=0.3)
                        Removes frame-to-frame jitter without GPU overhead
  - Derived metrics   : jaw_open_ratio, eye_blink_l, eye_blink_r — forensically
                        useful for lip-sync and blink consistency analysis

Design decisions:
  1. MediaPipe FaceMesh is run in static_image_mode=True (not video mode).
     This is intentional: each crop is already aligned and temporally segmented
     by DeepSORT. We do not want MediaPipe's internal Kalman filter — we implement
     our own EMA on the raw 478 landmarks, giving us full control over the
     smoothing strength (α) per track.

  2. Landmarks are extracted in MediaPipe's normalised [0, 1] space and then
     converted to pixel coordinates of the input crop. The z component is kept
     as a relative depth measure (not converted to absolute units).

  3. The EMA filter is maintained per track_id. When a track disappears and
     re-appears, its EMA state is reset (new track_id = fresh start).

  4. Three forensically-important derived scalar metrics are computed from
     specific landmark indices defined in MediaPipe's FaceMesh topology:
       - jaw_open_ratio : vertical mouth opening / face height
       - eye_blink_l    : Left Eye Aspect Ratio (EAR) — Soukupová & Čech (2016)
       - eye_blink_r    : Right Eye Aspect Ratio (EAR)

MediaPipe FaceMesh landmark indices used for derived metrics:
  - Upper lip mid:  13   / Lower lip mid:  14  (jaw open)
  - Chin mid:      152   / Forehead approx: 10 (face height)
  - Left eye EAR:  [362, 385, 387, 263, 373, 380]
  - Right eye EAR: [33,  160, 158, 133, 153, 144]

Usage:
    from preprocessing.landmark_tracer import LandmarkTracer

    tracer = LandmarkTracer(refine_landmarks=True, ema_alpha=0.3)
    for fp in frames:
        for tf in fp.tracked_faces:
            result = tracer.trace(
                track_id=tf.track_id,
                frame_id=fp.frame_id,
                timestamp_ms=fp.timestamp_ms,
                face_crop_rgb=tf.aligned_crop,
            )
    tracer.reset()
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .result_types import FaceLandmarkResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MediaPipe FaceMesh landmark index constants
# ---------------------------------------------------------------------------

# Jaw open ratio: vertical mouth gap / face height
_UPPER_LIP_IDX  = 13
_LOWER_LIP_IDX  = 14
_CHIN_IDX       = 152
_FOREHEAD_IDX   = 10   # approximate forehead point

# Eye Aspect Ratio (EAR) indices — 6 points per eye
# EAR = (‖p2-p6‖ + ‖p3-p5‖) / (2 * ‖p1-p4‖)
_LEFT_EYE_EAR_IDX  = [362, 385, 387, 263, 373, 380]  # MediaPipe left eye
_RIGHT_EYE_EAR_IDX = [33,  160, 158, 133, 153, 144]  # MediaPipe right eye


# ---------------------------------------------------------------------------
# EMA state container
# ---------------------------------------------------------------------------

class _EMAState:
    """Per-track EMA state for 478-point landmark smoothing."""

    def __init__(self, alpha: float) -> None:
        self.alpha    = alpha
        self.ema: Optional[np.ndarray] = None  # (478, 3)

    def update(self, raw: np.ndarray) -> np.ndarray:
        """
        Apply EMA: smoothed_t = α * raw_t + (1 - α) * smoothed_{t-1}.
        On first call, smoothed = raw (cold start).
        """
        if self.ema is None:
            self.ema = raw.copy()
        else:
            self.ema = self.alpha * raw + (1.0 - self.alpha) * self.ema
        return self.ema.copy()

    def reset(self) -> None:
        self.ema = None


# ---------------------------------------------------------------------------
# LandmarkTracer
# ---------------------------------------------------------------------------

class LandmarkTracer:
    """
    MediaPipe FaceMesh landmark extractor with per-track EMA temporal smoothing.

    This is the third module in VDS Layer 1.  It processes aligned face crops
    (from FaceTracker) and produces dense 478-point 3D landmark arrays with
    temporal smoothing and derived forensic metrics.

    Args:
        refine_landmarks (bool):
            If True, enables MediaPipe attention mesh — adds iris landmarks
            (total 478 instead of 468) and improves eye/lip precision.
            Slightly slower. Default: True.
        ema_alpha (float):
            EMA smoothing factor α in [0, 1].
            0.0 = fully smooth (static), 1.0 = no smoothing (raw).
            Default: 0.3 — balances jitter removal vs. responsiveness.
        min_detection_confidence (float):
            MediaPipe minimum face detection confidence. Default: 0.5.
        min_tracking_confidence (float):
            MediaPipe minimum tracking confidence. Default: 0.5.
        max_num_faces (int):
            Maximum faces MediaPipe will attempt to find in each crop.
            Since crops are already isolated faces, 1 is optimal. Default: 1.
    """

    def __init__(
        self,
        refine_landmarks:          bool  = True,
        ema_alpha:                 float = 0.3,
        min_detection_confidence:  float = 0.5,
        min_tracking_confidence:   float = 0.5,
        max_num_faces:             int   = 1,
    ) -> None:
        self.refine_landmarks         = refine_landmarks
        self.ema_alpha                = ema_alpha
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence  = min_tracking_confidence
        self.max_num_faces            = max_num_faces

        # Lazy-initialised MediaPipe FaceMesh instance
        self._face_mesh = None

        # Per-track EMA state: track_id → _EMAState
        self._ema_states: Dict[str, _EMAState] = {}

        logger.info(
            "LandmarkTracer init | refine=%s | ema_alpha=%.2f | "
            "det_conf=%.2f | track_conf=%.2f",
            refine_landmarks, ema_alpha,
            min_detection_confidence, min_tracking_confidence,
        )

    # ------------------------------------------------------------------
    # Lazy initialiser
    # ------------------------------------------------------------------

    def _get_face_mesh(self):
        """Lazily load MediaPipe FaceMesh (first call downloads ~5 MB model)."""
        if self._face_mesh is None:
            try:
                import mediapipe as mp  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "mediapipe is required for landmark tracing.\n"
                    "Install with:  pip install mediapipe>=0.10.0"
                ) from exc

            logger.info(
                "Loading MediaPipe FaceMesh | refine_landmarks=%s …",
                self.refine_landmarks,
            )
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,          # each frame treated independently
                max_num_faces=self.max_num_faces,
                refine_landmarks=self.refine_landmarks,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            n_lm = 478 if self.refine_landmarks else 468
            logger.info(
                "MediaPipe FaceMesh ready | landmark_count=%d", n_lm
            )
        return self._face_mesh

    # ------------------------------------------------------------------
    # EMA state management
    # ------------------------------------------------------------------

    def _get_ema(self, track_id: str) -> _EMAState:
        if track_id not in self._ema_states:
            self._ema_states[track_id] = _EMAState(alpha=self.ema_alpha)
            logger.debug("EMA state created for track_id='%s'.", track_id)
        return self._ema_states[track_id]

    # ------------------------------------------------------------------
    # Landmark utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _mediapipe_to_pixel(
        mp_landmark_list,
        crop_h: int,
        crop_w: int,
    ) -> np.ndarray:
        """
        Convert MediaPipe normalised landmarks to pixel coordinates of the crop.

        Args:
            mp_landmark_list : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
            crop_h, crop_w   : Height and width of the input crop in pixels.

        Returns:
            (N, 3) float32 array — (x_px, y_px, z_rel).
            z is kept in MediaPipe's relative depth units (not converted to pixels).
        """
        n = len(mp_landmark_list.landmark)
        arr = np.zeros((n, 3), dtype=np.float32)
        for i, lm in enumerate(mp_landmark_list.landmark):
            arr[i, 0] = lm.x * crop_w   # x pixel
            arr[i, 1] = lm.y * crop_h   # y pixel
            arr[i, 2] = lm.z            # relative depth (kept normalised)
        return arr

    @staticmethod
    def _ear(landmarks: np.ndarray, indices: List[int]) -> float:
        """
        Compute Eye Aspect Ratio (EAR) from 6 landmark points.

        EAR = (‖P2-P6‖ + ‖P3-P5‖) / (2 * ‖P1-P4‖)

        EAR ≈ 0.0  → eye fully closed (blink)
        EAR ≈ 0.3  → typical open eye

        Args:
            landmarks : (N, 3) array of all face landmarks (pixel space).
            indices   : 6 indices [P1, P2, P3, P4, P5, P6] for this eye.

        Returns:
            EAR scalar float.
        """
        p1, p2, p3, p4, p5, p6 = [landmarks[i, :2] for i in indices]
        A = np.linalg.norm(p2 - p6)
        B = np.linalg.norm(p3 - p5)
        C = np.linalg.norm(p1 - p4)
        return float((A + B) / (2.0 * C + 1e-8))

    @staticmethod
    def _jaw_open_ratio(landmarks: np.ndarray) -> float:
        """
        Compute jaw open ratio: vertical mouth gap normalised by face height.

        jaw_open = ‖upper_lip - lower_lip‖ / ‖chin - forehead‖

        Returns:
            Scalar in approximately [0, 1]. Values > 0.15 indicate significant
            mouth opening (speech, emotion). Used for lip-sync deepfake checks.
        """
        upper_lip = landmarks[_UPPER_LIP_IDX, :2]
        lower_lip = landmarks[_LOWER_LIP_IDX, :2]
        chin      = landmarks[_CHIN_IDX,      :2]
        forehead  = landmarks[_FOREHEAD_IDX,  :2]

        mouth_gap  = np.linalg.norm(upper_lip - lower_lip)
        face_height = np.linalg.norm(chin - forehead) + 1e-8
        return float(mouth_gap / face_height)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trace(
        self,
        track_id:      str,
        frame_id:      int,
        timestamp_ms:  float,
        face_crop_rgb: Optional[np.ndarray],
    ) -> FaceLandmarkResult:
        """
        Extract landmarks from a single aligned face crop.

        Args:
            track_id      : Stable track ID from FaceTracker (e.g. "track_001").
            frame_id      : Frame index from FramePacket.
            timestamp_ms  : Timestamp of the frame in milliseconds.
            face_crop_rgb : Aligned RGB face crop (H, W, 3). None triggers a
                            failure result with detection_success=False.

        Returns:
            FaceLandmarkResult — always returns, even on detection failure.
            Check `.detection_success` before consuming landmark arrays.
        """
        # ── Validate input ───────────────────────────────────────────────
        if face_crop_rgb is None:
            logger.warning(
                "track_id=%s frame_id=%d: no crop provided — skipping landmarks.",
                track_id, frame_id,
            )
            return FaceLandmarkResult(
                track_id=track_id,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                detection_success=False,
            )

        if face_crop_rgb.ndim != 3 or face_crop_rgb.shape[2] != 3:
            logger.warning(
                "track_id=%s frame_id=%d: invalid crop shape %s.",
                track_id, frame_id, face_crop_rgb.shape,
            )
            return FaceLandmarkResult(
                track_id=track_id,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                detection_success=False,
            )

        crop_h, crop_w = face_crop_rgb.shape[:2]
        face_area_px   = crop_h * crop_w

        face_mesh = self._get_face_mesh()

        # ── Run MediaPipe FaceMesh ────────────────────────────────────────
        mp_result = face_mesh.process(face_crop_rgb)

        if not mp_result.multi_face_landmarks:
            logger.debug(
                "track_id=%s frame_id=%d: MediaPipe found no face in crop.",
                track_id, frame_id,
            )
            return FaceLandmarkResult(
                track_id=track_id,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                face_area_px=face_area_px,
                detection_success=False,
            )

        # Take the first (and typically only) face result
        lm_list = mp_result.multi_face_landmarks[0]

        # ── Convert to pixel coordinates ──────────────────────────────────
        raw_landmarks = self._mediapipe_to_pixel(lm_list, crop_h, crop_w)

        # ── Apply EMA smoothing ───────────────────────────────────────────
        ema_state = self._get_ema(track_id)
        smoothed  = ema_state.update(raw_landmarks)

        # ── Compute derived forensic metrics ─────────────────────────────
        try:
            jaw_open = self._jaw_open_ratio(smoothed)
        except (IndexError, ValueError):
            jaw_open = 0.0

        try:
            ear_l = self._ear(smoothed, _LEFT_EYE_EAR_IDX)
        except (IndexError, ValueError):
            ear_l = 0.0

        try:
            ear_r = self._ear(smoothed, _RIGHT_EYE_EAR_IDX)
        except (IndexError, ValueError):
            ear_r = 0.0

        logger.debug(
            "track_id=%s frame_id=%d | jaw=%.3f | ear_l=%.3f | ear_r=%.3f",
            track_id, frame_id, jaw_open, ear_l, ear_r,
        )

        return FaceLandmarkResult(
            track_id=track_id,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            landmarks_478=raw_landmarks,
            smoothed_landmarks=smoothed,
            jaw_open_ratio=jaw_open,
            eye_blink_l=ear_l,
            eye_blink_r=ear_r,
            face_area_px=face_area_px,
            detection_success=True,
        )

    def process_video(
        self,
        frames: list,  # List[FramePacket]
    ) -> Dict[str, List[FaceLandmarkResult]]:
        """
        Convenience method: run trace() for every tracked face in every frame.

        Returns:
            Dict mapping track_id → ordered List[FaceLandmarkResult].
            This dict is the `landmarks` field of VideoPreprocessingResult.
        """
        logger.info("Running LandmarkTracer on %d frames …", len(frames))

        landmarks: Dict[str, List[FaceLandmarkResult]] = {}

        for fp in frames:
            for tf in fp.tracked_faces:
                result = self.trace(
                    track_id=tf.track_id,
                    frame_id=fp.frame_id,
                    timestamp_ms=fp.timestamp_ms,
                    face_crop_rgb=tf.aligned_crop,
                )
                if tf.track_id not in landmarks:
                    landmarks[tf.track_id] = []
                landmarks[tf.track_id].append(result)

        total_results  = sum(len(v) for v in landmarks.values())
        success_count  = sum(
            r.detection_success
            for v in landmarks.values()
            for r in v
        )
        logger.info(
            "LandmarkTracer complete | tracks=%d | total_results=%d | success=%d",
            len(landmarks), total_results, success_count,
        )
        return landmarks

    def reset(self) -> None:
        """
        Reset all EMA states for processing a new video.
        Does not close the MediaPipe FaceMesh (model stays loaded).
        """
        self._ema_states.clear()
        logger.info("LandmarkTracer EMA states reset for new video.")

    def close(self) -> None:
        """
        Close the MediaPipe FaceMesh resource.
        Call when the tracer will no longer be used (e.g. end of process).
        """
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
            logger.info("MediaPipe FaceMesh closed.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self) -> str:
        return (
            f"LandmarkTracer("
            f"refine_landmarks={self.refine_landmarks}, "
            f"ema_alpha={self.ema_alpha})"
        )
