"""
DeepGuard Platform — Image Detection Subsystem (IDS)
Module  : engines.image-engine.preprocessing.face_detector
Layer   : Layer 1 — Preprocessing
Task    : Face Detection & Alignment via RetinaFace (InsightFace / ONNX backend)

Architecture Reference (DeepGuard_Platform_Architecture.docx — Section 3.2, Layer 1):
  - Face Detection  : RetinaFace (open-source, SOTA) — detect all faces with bounding boxes
  - Face Alignment  : 5-point landmark alignment — normalize pose and scale
  - Parallel paths  : full-image features AND face-cropped features produced
  - Tools           : OpenCV, RetinaFace

Implementation note:
  We use the `insightface` library (ONNX Runtime backend) which ships the
  RetinaFace model (buffalo_sc / buffalo_l packs) without any TensorFlow
  dependency — keeping the entire stack on PyTorch / ONNX.

Usage:
    from img_preprocessing.face_detector import FaceDetector

    detector = FaceDetector(confidence_threshold=0.90, target_size=(224, 224))

    # From a file path:
    results = detector.detect_and_align("path/to/image.jpg")

    # From a NumPy array (OpenCV BGR):
    import cv2
    img = cv2.imread("path/to/image.jpg")
    results = detector.detect_and_align(img)

    for face in results:
        print(face.score, face.bbox)
        # face.aligned_crop  → NumPy RGB array ready for model input
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Landmarks:
    """
    Five-point facial landmarks returned by RetinaFace (InsightFace).

    Coordinate system: (x, y) in the original (unscaled) image space.
    InsightFace returns landmarks ordered as:
        [right_eye, left_eye, nose, mouth_right, mouth_left]
    """
    right_eye:   Tuple[float, float]
    left_eye:    Tuple[float, float]
    nose:        Tuple[float, float]
    mouth_right: Tuple[float, float]
    mouth_left:  Tuple[float, float]

    def to_numpy(self) -> np.ndarray:
        """Return landmarks as a (5, 2) float32 array."""
        return np.array(
            [self.right_eye, self.left_eye, self.nose,
             self.mouth_right, self.mouth_left],
            dtype=np.float32,
        )


@dataclass
class FaceDetectionResult:
    """
    Output of the face detection + alignment pipeline for a single detected face.

    Attributes:
        face_id      : Sequential identifier assigned during detection (e.g. 'face_1')
        score        : Detection confidence in [0, 1]
        bbox         : Bounding box as (x1, y1, x2, y2) in pixel coordinates
        landmarks    : Five facial key-points (right eye, left eye, nose,
                       mouth right, mouth left)
        aligned_crop : Aligned & resized face crop as an RGB uint8 NumPy array.
                       Shape is (H, W, 3) — ready for downstream model input.
                       None if alignment was skipped.
        full_image   : Reference to the original BGR image — used for the
                       full-image feature branch alongside the face-cropped branch.
    """
    face_id:      str
    score:        float
    bbox:         Tuple[int, int, int, int]
    landmarks:    Landmarks
    aligned_crop: Optional[np.ndarray] = field(default=None, repr=False)
    full_image:   Optional[np.ndarray] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height

    def bbox_as_xywh(self) -> Tuple[int, int, int, int]:
        """Return bounding box in (x, y, width, height) format."""
        x1, y1, x2, y2 = self.bbox
        return x1, y1, x2 - x1, y2 - y1

    def to_dict(self) -> Dict:
        """Serialise result to a plain dictionary (excludes image arrays)."""
        return {
            "face_id":   self.face_id,
            "score":     round(self.score, 6),
            "bbox":      list(self.bbox),
            "bbox_xywh": list(self.bbox_as_xywh()),
            "landmarks": {
                "right_eye":   list(self.landmarks.right_eye),
                "left_eye":    list(self.landmarks.left_eye),
                "nose":        list(self.landmarks.nose),
                "mouth_right": list(self.landmarks.mouth_right),
                "mouth_left":  list(self.landmarks.mouth_left),
            },
            "crop_shape": list(self.aligned_crop.shape) if self.aligned_crop is not None else None,
        }


# ---------------------------------------------------------------------------
# FaceDetector
# ---------------------------------------------------------------------------

class FaceDetector:
    """
    RetinaFace-based face detection and alignment module (InsightFace / ONNX backend).

    This is Layer 1 (Preprocessing) of the Image Detection Subsystem (IDS).
    It implements:
      1. Face detection with bounding boxes and confidence scores.
      2. 5-point landmark detection (right eye, left eye, nose, mouth corners).
      3. Face alignment — affine transform that places both eyes on a horizontal
         axis, centred, at a normalised scale. This is the canonical SOTA
         pre-processing step for downstream deepfake detection models.
      4. Parallel output of the aligned face crop (face-level branch) AND
         a reference to the full image (full-image / background branch).

    Args:
        confidence_threshold (float): Minimum detection confidence to retain.
            Faces below this threshold are silently discarded. Default: 0.90.
        target_size (tuple): (width, height) of the aligned face crop output.
            Default: (224, 224) — compatible with ViT and EfficientNet inputs.
        align (bool): Whether to perform affine landmark alignment. Default: True.
        det_size (tuple): Internal detection resolution fed to RetinaFace.
            Larger values detect smaller faces but are slower. Default: (640, 640).
        model_pack (str): InsightFace model pack to use.
            'buffalo_sc' = small/fast (MobileNet backbone, recommended).
            'buffalo_l'  = large/accurate (ResNet50 backbone).
    """

    # Reference 5-point landmarks for the 112×112 ArcFace canonical alignment,
    # used to compute the affine transform target. We scale these to target_size.
    _REFERENCE_LANDMARKS_112 = np.array(
        [
            [38.2946, 51.6963],   # right eye
            [73.5318, 51.5014],   # left eye
            [56.0252, 71.7366],   # nose
            [41.5493, 92.3655],   # mouth right
            [70.7299, 92.2041],   # mouth left
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        confidence_threshold: float = 0.90,
        target_size: Tuple[int, int] = (224, 224),
        align: bool = True,
        det_size: Tuple[int, int] = (640, 640),
        model_pack: str = "buffalo_sc",
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.target_size = target_size
        self.align = align
        self.det_size = det_size
        self.model_pack = model_pack

        # Lazily initialised — expensive ONNX model load deferred to first call
        self._app = None

        logger.info(
            "FaceDetector initialised | confidence_threshold=%.2f | "
            "target_size=%s | align=%s | det_size=%s | model_pack=%s",
            confidence_threshold, target_size, align, det_size, model_pack,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_app(self):
        """
        Lazily load the InsightFace FaceAnalysis app (downloads weights on
        first run, then caches them locally under ~/.insightface/).
        A 1x1 warmup inference is run so the ONNX sessions are fully
        compiled before the first real detection call.
        """
        if self._app is None:
            try:
                from insightface.app import FaceAnalysis  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "The 'insightface' package is required.\n"
                    "Install it with:  pip install insightface onnxruntime"
                ) from exc

            logger.info(
                "Loading InsightFace FaceAnalysis (model_pack='%s') — "
                "weights will be downloaded on first run (~10 MB)...",
                self.model_pack,
            )
            app = FaceAnalysis(
                name=self.model_pack,
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=self.det_size)

            # Warmup: run a tiny blank image through the model so ONNX
            # sessions finish compiling before the first real call.
            import numpy as _np  # noqa: PLC0415
            _dummy = _np.zeros((64, 64, 3), dtype=_np.uint8)
            app.get(_dummy)
            logger.info("InsightFace FaceAnalysis ready (warmup complete).")

            self._app = app
        return self._app

    @staticmethod
    def _load_image(image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Accept a file path (str / Path) or a NumPy array (OpenCV BGR).
        Returns a BGR uint8 NumPy array.
        """
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"OpenCV failed to load image: {path}")
            logger.debug("Loaded image from disk: %s | shape=%s", path, img.shape)
            return img
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    f"Expected a 3-channel (H, W, 3) array, got shape {image.shape}"
                )
            return image
        raise TypeError(
            f"image must be a file path (str/Path) or a NumPy array, got {type(image)}"
        )

    @staticmethod
    def _clamp_bbox(
        bbox: Tuple[int, int, int, int],
        img_h: int,
        img_w: int,
    ) -> Tuple[int, int, int, int]:
        """Clamp bounding box coordinates to image boundaries."""
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))
        return x1, y1, x2, y2

    def _align_face(
        self,
        img_bgr: np.ndarray,
        landmarks: Landmarks,
    ) -> np.ndarray:
        """
        Compute a similarity transform from the 5 detected landmarks to the
        canonical ArcFace reference positions and warp the face crop.

        Uses cv2.estimateAffinePartial2D (scale + rotation + translation,
        no shear) — the standard approach for face normalisation in deepfake
        detection research.

        Returns:
            Aligned face crop as an RGB uint8 array of shape (H, W, 3).
        """
        src_pts = landmarks.to_numpy()

        # Scale reference landmarks from 112px to target_size
        ref_pts = self._REFERENCE_LANDMARKS_112.copy()
        ref_pts[:, 0] *= self.target_size[0] / 112.0
        ref_pts[:, 1] *= self.target_size[1] / 112.0

        # Estimate similarity transform (rotation + scale + translation)
        M, _ = cv2.estimateAffinePartial2D(src_pts, ref_pts, method=cv2.LMEDS)

        if M is None:
            logger.warning("Affine estimation failed — falling back to simple crop.")
            return self._simple_crop(img_bgr, landmarks)

        warped = cv2.warpAffine(
            img_bgr,
            M,
            self.target_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        # Convert BGR → RGB for downstream PyTorch model compatibility
        return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _simple_crop(
        img_bgr: np.ndarray,
        landmarks: Landmarks,
        padding: float = 0.30,
    ) -> np.ndarray:
        """
        Fallback: simple square crop around the midpoint of the two eyes.
        Returns an RGB array.
        """
        re = np.array(landmarks.right_eye)
        le = np.array(landmarks.left_eye)
        center = ((re + le) / 2).astype(int)
        eye_dist = int(np.linalg.norm(le - re))
        half = int(eye_dist * (1 + padding))
        h, w = img_bgr.shape[:2]
        x1 = max(0, center[0] - half)
        y1 = max(0, center[1] - half)
        x2 = min(w, center[0] + half)
        y2 = min(h, center[1] + half)
        crop = img_bgr[y1:y2, x1:x2]
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
    ) -> List[FaceDetectionResult]:
        """
        Run RetinaFace face detection on the given image.

        Returns a list of FaceDetectionResult objects — one per detected face
        that meets the confidence threshold. Results are sorted by descending
        confidence score.

        Args:
            image: File path (str/Path) or a BGR NumPy array.

        Returns:
            List[FaceDetectionResult]: Detected faces. May be empty.
        """
        img_bgr = self._load_image(image)
        app = self._get_app()

        logger.info(
            "Running RetinaFace detection on image shape=%s ...", img_bgr.shape
        )

        # InsightFace expects BGR — same as OpenCV default
        raw_faces = app.get(img_bgr)

        # Always log raw count before any threshold filtering (useful for debugging)
        raw_count = len(raw_faces) if raw_faces else 0
        logger.info("Raw detections: %d face(s) found by model.", raw_count)
        if raw_faces:
            scores = [round(float(f.det_score), 4) for f in raw_faces]
            logger.info("Raw confidence scores: %s  | threshold=%.2f",
                        scores, self.confidence_threshold)

        if not raw_faces:
            logger.info("No faces detected by model.")
            return []

        img_h, img_w = img_bgr.shape[:2]
        results: List[FaceDetectionResult] = []

        for idx, face in enumerate(raw_faces):
            score = float(face.det_score)

            if score < self.confidence_threshold:
                logger.info(
                    "Skipping face_%d — confidence %.4f below threshold %.2f",
                    idx + 1, score, self.confidence_threshold,
                )
                continue

            # --- Bounding box (x1, y1, x2, y2) ---
            x1, y1, x2, y2 = (int(v) for v in face.bbox)
            bbox = self._clamp_bbox((x1, y1, x2, y2), img_h, img_w)

            # --- 5-point landmarks ---
            # InsightFace kps shape: (5, 2) — order: RE, LE, Nose, MR, ML
            kps = face.kps.astype(float)
            landmarks = Landmarks(
                right_eye=   (kps[0, 0], kps[0, 1]),
                left_eye=    (kps[1, 0], kps[1, 1]),
                nose=        (kps[2, 0], kps[2, 1]),
                mouth_right= (kps[3, 0], kps[3, 1]),
                mouth_left=  (kps[4, 0], kps[4, 1]),
            )

            results.append(
                FaceDetectionResult(
                    face_id=f"face_{idx + 1}",
                    score=score,
                    bbox=bbox,
                    landmarks=landmarks,
                    full_image=img_bgr,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        logger.info("Detected %d face(s) above threshold.", len(results))
        return results

    def align_face(
        self,
        image: Union[str, Path, np.ndarray],
        face: FaceDetectionResult,
    ) -> np.ndarray:
        """
        Compute and return the aligned face crop for a previously-detected face.

        Args:
            image : The same image used in detect() — path or BGR array.
            face  : A FaceDetectionResult from detect().

        Returns:
            Aligned RGB face crop as a NumPy array of shape self.target_size + (3,).
        """
        img_bgr = self._load_image(image)
        return self._align_face(img_bgr, face.landmarks)

    def detect_and_align(
        self,
        image: Union[str, Path, np.ndarray],
    ) -> List[FaceDetectionResult]:
        """
        Full preprocessing pipeline: detect all faces AND compute aligned crops
        in a single call.

        This is the primary entry point for the IDS preprocessing layer.
        Each returned FaceDetectionResult has:
          - aligned_crop : RGB array ready for CLIP ViT / EfficientNet input
          - full_image   : original BGR array for the full-image feature branch

        Args:
            image: File path (str/Path) or a BGR NumPy array.

        Returns:
            List[FaceDetectionResult] with aligned_crop populated.
        """
        img_bgr = self._load_image(image)
        results = self.detect(img_bgr)

        if self.align:
            for face in results:
                face.aligned_crop = self._align_face(img_bgr, face.landmarks)
                logger.debug(
                    "Aligned %s | crop shape=%s",
                    face.face_id, face.aligned_crop.shape,
                )

        return results

    def visualize(
        self,
        image: Union[str, Path, np.ndarray],
        results: List[FaceDetectionResult],
        draw_landmarks: bool = True,
        draw_score: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes, landmarks, and confidence scores on a copy of the
        image for debugging and inspection.

        Args:
            image          : File path or BGR NumPy array.
            results        : From detect() or detect_and_align().
            draw_landmarks : If True, draw the 5 facial keypoints.
            draw_score     : If True, annotate confidence score near bbox.

        Returns:
            BGR NumPy array with annotations — ready for cv2.imwrite() or display.
        """
        img_bgr = self._load_image(image)
        vis = img_bgr.copy()

        BBOX_COLOR     = (0, 255, 0)      # Green
        LANDMARK_COLOR = (0, 120, 255)    # Orange
        TEXT_COLOR     = (255, 255, 255)  # White
        TEXT_BG_COLOR  = (0, 180, 0)      # Dark green

        for face in results:
            x1, y1, x2, y2 = face.bbox

            # Bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), BBOX_COLOR, thickness=2)

            # Score label
            if draw_score:
                label = f"{face.face_id} | {face.score:.2%}"
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                )
                cv2.rectangle(
                    vis,
                    (x1, y1 - th - baseline - 4),
                    (x1 + tw + 4, y1),
                    TEXT_BG_COLOR, -1,
                )
                cv2.putText(
                    vis, label,
                    (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1,
                    cv2.LINE_AA,
                )

            # Facial landmarks
            if draw_landmarks:
                lm_labels = {
                    "RE": face.landmarks.right_eye,
                    "LE": face.landmarks.left_eye,
                    "N":  face.landmarks.nose,
                    "MR": face.landmarks.mouth_right,
                    "ML": face.landmarks.mouth_left,
                }
                for lbl, (lx, ly) in lm_labels.items():
                    pt = (int(lx), int(ly))
                    cv2.circle(vis, pt, 4, LANDMARK_COLOR, -1)
                    cv2.putText(
                        vis, lbl,
                        (pt[0] + 5, pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, LANDMARK_COLOR, 1,
                        cv2.LINE_AA,
                    )

        logger.debug("Visualization generated for %d face(s).", len(results))
        return vis

    def __repr__(self) -> str:
        return (
            f"FaceDetector("
            f"confidence_threshold={self.confidence_threshold}, "
            f"target_size={self.target_size}, "
            f"align={self.align}, "
            f"model_pack='{self.model_pack}')"
        )
