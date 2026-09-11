"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.preprocessing.face_tracker
Layer   : Layer 1 — Temporal Preprocessing
Task    : Multi-face tracking across video frames via DeepSORT

Architecture Reference (DeepGuard_Platform_Architecture.docx — VDS Layer 1):
  - Per-frame detection  : OpenCV DNN (default, fast) or RetinaFace (high-quality)
  - Identity continuity  : DeepSORT (deep-sort-realtime) assigns persistent track IDs
  - Output               : List[TrackedFace] per frame — enriched with track_id and
                           track_age for downstream temporal analysis

Design decisions:
  1. Dual-backend detection:
       'dnn'       — OpenCV's built-in Caffe/SSD face detector (~50 fps on CPU for
                     720p). Best for long videos where per-frame speed matters.
       'retinaface'— Reuses engines/image-engine FaceDetector (InsightFace/ONNX).
                     Highest accuracy, ~5–15 fps on CPU. Use for forensic precision.

  2. DeepSORT (deep-sort-realtime pip package) handles:
       - Kalman filter prediction of face positions between frames.
       - Re-association of detections to tracks using IoU + appearance cosine distance.
       - Automatic retirement of stale tracks after max_age missed frames.

  3. Track IDs are stable zero-padded strings (e.g. "track_001") for sort stability
     and JSON serialisability. Internal DeepSORT integer IDs are mapped on first use.

  4. Aligned face crops are produced for every confirmed track:
       'dnn'       — a simple padded rectangle crop + resize to target_size.
       'retinaface'— the full 5-point affine-aligned crop from FaceDetector.

Usage:
    from preprocessing.face_tracker import FaceTracker

    tracker = FaceTracker(backend="dnn", max_age=30, min_hits=3)
    for fp in frames:
        tracked = tracker.update(fp)     # → List[TrackedFace]  (in-place on fp too)
        fp.tracked_faces = tracked
    tracker.reset()                      # clear state for next video
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .result_types import FramePacket, TrackedFace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenCV DNN face detector helper
# ---------------------------------------------------------------------------

class _DNNFaceDetector:
    """
    Lightweight OpenCV DNN (Caffe SSD) face detector.

    Downloads the pretrained proto + caffemodel from OpenCV's GitHub on first use
    and caches them in ~/.deepguard/dnn_cache/.  No GPU required.

    Detection output: list of (x1, y1, x2, y2, confidence) tuples.
    """

    # Caffe model URLs — original OpenCV extra-models repo
    _PROTO_URL  = (
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/"
        "face_detector/deploy.prototxt"
    )
    _MODEL_URL  = (
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    )
    _CACHE_DIR  = Path.home() / ".deepguard" / "dnn_cache"
    _PROTO_PATH = _CACHE_DIR / "deploy.prototxt"
    _MODEL_PATH = _CACHE_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

    def __init__(self, confidence_threshold: float = 0.70) -> None:
        self.confidence_threshold = confidence_threshold
        self._net = None

    def _get_net(self) -> cv2.dnn.Net:
        if self._net is not None:
            return self._net

        # Download model files if not already cached
        if not self._MODEL_PATH.exists() or not self._PROTO_PATH.exists():
            self._download_model_files()

        logger.info("Loading OpenCV DNN face detector …")
        net = cv2.dnn.readNetFromCaffe(
            str(self._PROTO_PATH),
            str(self._MODEL_PATH),
        )
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._net = net
        logger.info("OpenCV DNN face detector ready.")
        return self._net

    def _download_model_files(self) -> None:
        """Download DNN model files to cache dir using urllib."""
        import urllib.request  # noqa: PLC0415

        self._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        pairs = [
            (self._PROTO_URL,  self._PROTO_PATH),
            (self._MODEL_URL,  self._MODEL_PATH),
        ]
        for url, dest in pairs:
            if not dest.exists():
                logger.info("Downloading %s → %s …", url, dest)
                try:
                    urllib.request.urlretrieve(url, dest)
                    logger.info("Downloaded: %s", dest)
                except Exception as exc:
                    raise RuntimeError(
                        f"[FaceTracker] Failed to download DNN model file from:\n{url}\n"
                        f"Error: {exc}\n\n"
                        "Please download manually and place at:\n"
                        f"  {dest}"
                    ) from exc

    def detect(
        self,
        rgb: np.ndarray,
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in an RGB frame.

        Returns:
            List of (x1, y1, x2, y2, confidence) — pixel coordinates clamped
            to the frame boundary.
        """
        net  = self._get_net()
        h, w = rgb.shape[:2]
        bgr  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Create blob: resize to 300x300, mean-subtract (104, 177, 123)
        blob = cv2.dnn.blobFromImage(
            bgr, scalefactor=1.0, size=(300, 300),
            mean=(104.0, 177.0, 123.0), swapRB=False, crop=False,
        )
        net.setInput(blob)
        detections = net.forward()  # shape: (1, 1, N, 7)

        results = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self.confidence_threshold:
                continue
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            # Clamp to frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                results.append((x1, y1, x2, y2, conf))

        return results


# ---------------------------------------------------------------------------
# FaceTracker
# ---------------------------------------------------------------------------

class FaceTracker:
    """
    Multi-face DeepSORT tracker with dual-backend detection.

    This is the second module in VDS Layer 1.  It processes frames sequentially
    and produces a `TrackedFace` list for each frame with stable track IDs.

    Args:
        backend (str):
            'dnn'        — OpenCV Caffe SSD (fast, ~50 fps on CPU). Default.
            'retinaface' — InsightFace RetinaFace (accurate, ~5–15 fps on CPU).
        confidence_threshold (float):
            Minimum detection confidence to feed into DeepSORT. Default: 0.70.
        max_age (int):
            Maximum number of consecutive missed frames before a track is retired.
            Higher = more tolerance for brief occlusion. Default: 30.
        min_hits (int):
            Minimum detections before a new track is confirmed (avoids track
            flickering on false positives). Default: 3.
        target_size (tuple):
            (width, height) of aligned face crops. Default: (224, 224).
        retinaface_model_pack (str):
            InsightFace model pack for the RetinaFace backend.
            'buffalo_sc' (fast) or 'buffalo_l' (accurate). Default: 'buffalo_sc'.
        embedder (str or None):
            DeepSORT appearance embedder for re-identification.
            None uses the default MobileNetV2 embedder from deep-sort-realtime.
    """

    def __init__(
        self,
        backend:              str   = "dnn",
        confidence_threshold: float = 0.70,
        max_age:              int   = 30,
        min_hits:             int   = 3,
        target_size:          Tuple[int, int] = (224, 224),
        retinaface_model_pack: str  = "buffalo_sc",
        embedder:             Optional[str] = None,
    ) -> None:

        if backend not in ("dnn", "retinaface"):
            raise ValueError(f"backend must be 'dnn' or 'retinaface', got '{backend}'")

        self.backend              = backend
        self.confidence_threshold = confidence_threshold
        self.max_age              = max_age
        self.min_hits             = min_hits
        self.target_size          = target_size
        self.retinaface_model_pack = retinaface_model_pack

        # Lazy-initialised components
        self._dnn_detector:        Optional[_DNNFaceDetector] = None
        self._retinaface_detector  = None   # FaceDetector from image-engine
        self._tracker              = None   # DeepSORT tracker instance

        # Track ID mapping: DeepSORT integer → stable "track_XXX" string
        self._track_id_map: Dict[int, str] = {}
        self._next_stable_id: int = 1

        logger.info(
            "FaceTracker init | backend=%s | confidence=%.2f | max_age=%d | min_hits=%d",
            backend, confidence_threshold, max_age, min_hits,
        )

    # ------------------------------------------------------------------
    # Lazy initialisers
    # ------------------------------------------------------------------

    def _get_tracker(self):
        """Lazily initialise the DeepSORT tracker."""
        if self._tracker is None:
            try:
                from deep_sort_realtime.deepsort_tracker import DeepSort  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "deep-sort-realtime is required for face tracking.\n"
                    "Install with:  pip install deep-sort-realtime"
                ) from exc

            self._tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.min_hits,
                nms_max_overlap=0.85,
                max_cosine_distance=0.3,
                nn_budget=None,
                override_track_class=None,
                embedder="mobilenet",      # built-in appearance embedder
                half=False,               # CPU — no fp16 quantisation
                bgr=False,                # we pass RGB frames
                today=None,
            )
            logger.info("DeepSORT tracker initialised (max_age=%d, n_init=%d).",
                        self.max_age, self.min_hits)
        return self._tracker

    def _get_dnn_detector(self) -> _DNNFaceDetector:
        if self._dnn_detector is None:
            self._dnn_detector = _DNNFaceDetector(
                confidence_threshold=self.confidence_threshold
            )
        return self._dnn_detector

    def _get_retinaface_detector(self):
        """
        Lazily import and initialise the FaceDetector from the image-engine.
        Handles both relative (same repo) and installed-package imports gracefully.
        """
        if self._retinaface_detector is None:
            try:
                # Try the repo-relative path first
                import sys, os  # noqa: PLC0415, E401
                image_engine_path = str(
                    Path(__file__).resolve().parents[3] / "image-engine"
                )
                if image_engine_path not in sys.path:
                    sys.path.insert(0, image_engine_path)
                from preprocessing.face_detector import FaceDetector  # noqa: PLC0415
            except ImportError:
                try:
                    from engines.image_engine.preprocessing.face_detector import FaceDetector  # noqa: PLC0415
                except ImportError as exc:
                    raise ImportError(
                        "Could not import FaceDetector from the image-engine.\n"
                        "Ensure the image-engine is on your Python path or install it."
                    ) from exc

            self._retinaface_detector = FaceDetector(
                confidence_threshold=self.confidence_threshold,
                target_size=self.target_size,
                align=True,
                model_pack=self.retinaface_model_pack,
            )
            logger.info(
                "RetinaFace backend ready (model_pack='%s').",
                self.retinaface_model_pack,
            )
        return self._retinaface_detector

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _detect_dnn(
        self,
        rgb: np.ndarray,
    ) -> Tuple[List, List]:
        """
        Run OpenCV DNN detection and produce:
          - detections_for_deepsort: [[x1, y1, w, h], confidence, 'face'] tuples
          - raw_detections: (x1, y1, x2, y2, confidence) tuples for crop extraction

        Returns:
            (deepsort_detections, raw_detections)
        """
        raw = self._get_dnn_detector().detect(rgb)
        deepsort_dets = []
        for (x1, y1, x2, y2, conf) in raw:
            deepsort_dets.append(
                ([x1, y1, x2 - x1, y2 - y1], conf, "face")
            )
        return deepsort_dets, raw

    def _detect_retinaface(
        self,
        rgb: np.ndarray,
    ) -> Tuple[List, List]:
        """
        Run RetinaFace detection and produce DeepSORT-compatible detections.

        Returns:
            (deepsort_detections, face_detection_results)
        """
        detector = self._get_retinaface_detector()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        face_results = detector.detect_and_align(bgr)

        deepsort_dets = []
        for face in face_results:
            x1, y1, x2, y2 = face.bbox
            deepsort_dets.append(
                ([x1, y1, x2 - x1, y2 - y1], face.score, "face")
            )
        return deepsort_dets, face_results

    # ------------------------------------------------------------------
    # Crop helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _crop_dnn(
        rgb: np.ndarray,
        bbox: Tuple[int, int, int, int],
        target_size: Tuple[int, int],
        pad_ratio: float = 0.20,
    ) -> np.ndarray:
        """
        Produce a padded rectangular crop from the raw DNN bounding box.
        Resizes to target_size for downstream MediaPipe input.
        """
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * pad_ratio)
        pad_y = int(bh * pad_ratio)
        x1c = max(0, x1 - pad_x)
        y1c = max(0, y1 - pad_y)
        x2c = min(w, x2 + pad_x)
        y2c = min(h, y2 + pad_y)
        crop = rgb[y1c:y2c, x1c:x2c]
        return cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------
    # Stable track ID management
    # ------------------------------------------------------------------

    def _get_stable_id(self, deepsort_id: int) -> str:
        """Map a DeepSORT integer track ID to a stable zero-padded string."""
        if deepsort_id not in self._track_id_map:
            self._track_id_map[deepsort_id] = f"track_{self._next_stable_id:03d}"
            self._next_stable_id += 1
        return self._track_id_map[deepsort_id]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame: FramePacket) -> List[TrackedFace]:
        """
        Process a single FramePacket through detection + DeepSORT tracking.

        Populates `frame.tracked_faces` in-place AND returns the same list.

        Args:
            frame: A FramePacket with rgb_array populated.

        Returns:
            List[TrackedFace] — confirmed tracks for this frame.
        """
        rgb = frame.rgb_array
        if rgb is None:
            logger.warning(
                "frame_id=%d has no rgb_array — skipping tracking.", frame.frame_id
            )
            return []

        tracker = self._get_tracker()

        # ── Detect faces ────────────────────────────────────────────────
        if self.backend == "dnn":
            deepsort_dets, raw_dets = self._detect_dnn(rgb)
            face_results = None
        else:  # retinaface
            deepsort_dets, face_results = self._detect_retinaface(rgb)
            raw_dets = None

        n_det = len(deepsort_dets)
        logger.debug(
            "frame_id=%d | backend=%s | raw_detections=%d",
            frame.frame_id, self.backend, n_det,
        )

        # ── Run DeepSORT update ─────────────────────────────────────────
        if n_det == 0:
            # Call tracker with empty list to advance Kalman predictions
            tracks = tracker.update_tracks([], frame=rgb)
        else:
            tracks = tracker.update_tracks(deepsort_dets, frame=rgb)

        # ── Build TrackedFace objects for confirmed tracks ──────────────
        tracked: List[TrackedFace] = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            ds_id   = track.track_id
            track_id = self._get_stable_id(int(ds_id))
            ltrb    = track.to_ltrb()   # (left, top, right, bottom)
            x1 = int(max(0, ltrb[0]))
            y1 = int(max(0, ltrb[1]))
            x2 = int(min(rgb.shape[1], ltrb[2]))
            y2 = int(min(rgb.shape[0], ltrb[3]))
            bbox = (x1, y1, x2, y2)

            # ── Produce aligned crop ─────────────────────────────────
            aligned_crop: Optional[np.ndarray] = None
            bbox_5lm:     Optional[list]       = None

            if self.backend == "retinaface" and face_results:
                # Match the track's LTRB to the nearest RetinaFace result by IoU
                matched_face = self._match_track_to_face(bbox, face_results)
                if matched_face is not None:
                    aligned_crop = matched_face.aligned_crop
                    lm = matched_face.landmarks
                    bbox_5lm = [
                        lm.right_eye, lm.left_eye, lm.nose,
                        lm.mouth_right, lm.mouth_left,
                    ]
            else:
                # DNN backend: produce padded rectangle crop
                if x2 > x1 and y2 > y1:
                    aligned_crop = self._crop_dnn(rgb, bbox, self.target_size)

            tracked.append(
                TrackedFace(
                    track_id=track_id,
                    track_age=track.age,
                    detection_score=track.det_conf if track.det_conf is not None else 0.0,
                    bbox=bbox,
                    frame_id=frame.frame_id,
                    timestamp_ms=frame.timestamp_ms,
                    aligned_crop=aligned_crop,
                    bbox_5lm=bbox_5lm,
                    detector_backend=self.backend,
                )
            )

        logger.debug(
            "frame_id=%d | confirmed_tracks=%d", frame.frame_id, len(tracked)
        )

        frame.tracked_faces = tracked
        return tracked

    @staticmethod
    def _match_track_to_face(
        track_bbox: Tuple[int, int, int, int],
        face_results: list,
        iou_threshold: float = 0.3,
    ):
        """
        Match a DeepSORT-predicted bbox to the closest RetinaFace detection by IoU.

        Returns the matched FaceDetectionResult, or None if no match exceeds threshold.
        """
        best_iou   = iou_threshold
        best_face  = None
        tx1, ty1, tx2, ty2 = track_bbox
        t_area = max(0, tx2 - tx1) * max(0, ty2 - ty1)

        for face in face_results:
            fx1, fy1, fx2, fy2 = face.bbox
            ix1 = max(tx1, fx1)
            iy1 = max(ty1, fy1)
            ix2 = min(tx2, fx2)
            iy2 = min(ty2, fy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                continue
            f_area = (fx2 - fx1) * (fy2 - fy1)
            union  = t_area + f_area - inter
            iou    = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou  = iou
                best_face = face

        return best_face

    def process_video(self, frames: List[FramePacket]) -> List[FramePacket]:
        """
        Convenience method: process every frame in a video sequentially.

        Populates `tracked_faces` on each FramePacket in-place and returns
        the mutated list.

        Args:
            frames: Ordered list of FramePacket objects from VideoIngestor.

        Returns:
            The same list with tracked_faces populated.
        """
        logger.info(
            "Running FaceTracker on %d frames | backend=%s …", len(frames), self.backend
        )
        for fp in frames:
            self.update(fp)

        # Collect unique track IDs for logging
        all_ids = set()
        for fp in frames:
            for tf in fp.tracked_faces:
                all_ids.add(tf.track_id)

        logger.info(
            "FaceTracker complete | unique_tracks=%d | ids=%s",
            len(all_ids), sorted(all_ids),
        )
        return frames

    def reset(self) -> None:
        """
        Reset tracker state for processing a new video.
        Clears all track ID mappings and DeepSORT internal state.
        """
        self._tracker          = None
        self._track_id_map     = {}
        self._next_stable_id   = 1
        logger.info("FaceTracker state reset for new video.")

    def __repr__(self) -> str:
        return (
            f"FaceTracker("
            f"backend='{self.backend}', "
            f"confidence_threshold={self.confidence_threshold}, "
            f"max_age={self.max_age}, "
            f"min_hits={self.min_hits})"
        )
