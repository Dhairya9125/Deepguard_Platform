"""
DeepGuard Platform — Image Detection Subsystem (IDS)
Module  : engines.image-engine.preprocessing.face_aligner
Layer   : Layer 1 — Preprocessing
Task    : Dense Face Alignment via MediaPipe Face Mesh (478 landmarks)

Architecture Reference (DeepGuard_Platform_Architecture.docx — Section 3.2, Layer 1):
  - Face Alignment : 5-point or 68-point landmark alignment using dlib or MediaPipe
                     — normalize pose and scale
  - Tools          : OpenCV, MediaPipe

Implementation note — MediaPipe Tasks API:
  MediaPipe 0.10.31+ (the only versions available for Python 3.14) removed the
  legacy `mp.solutions` API. We use the new MediaPipe Tasks API instead:
    mediapipe.tasks.python.vision.FaceLandmarker
  The `face_landmarker.task` model (~6 MB) is downloaded automatically on
  first use and cached at engines/image-engine/models/face_landmarker.task.

Why MediaPipe AFTER RetinaFace?
  RetinaFace (InsightFace) gives 5 coarse landmarks — good for rough alignment.
  MediaPipe Face Mesh gives 478 dense sub-pixel landmarks — enabling:
    * Much more precise eye-centre estimation (averaging 6+ contour points
      or using iris-centre points 468 / 473)
    * Stable pose normalisation even for partial occlusion
    * 3D landmark coordinates for later pose estimation
    * A richer landmark set for downstream feature extraction layers

Usage:
    from img_preprocessing.face_aligner import FaceAligner, AlignmentResult

    aligner = FaceAligner(target_size=(224, 224))

    # From full image + bbox (recommended)
    result = aligner.align(img_bgr, bbox=(x1, y1, x2, y2))
    if result:
        print(result.landmarks_2d.shape)    # (478, 2) — including iris points
        print(result.aligned_crop.shape)    # (224, 224, 3) RGB

    # Context manager — auto-releases resources
    with FaceAligner() as aligner:
        results = aligner.align_batch(img_bgr, bboxes)
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe landmark index constants  (478-point topology — Tasks API)
# ---------------------------------------------------------------------------

# Eye contour points — averaged for stable eye centres
_LEFT_EYE_CONTOUR  = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE_CONTOUR = [362, 385, 387, 263, 373, 380]

# Iris centres — only available when model includes iris refinement
# (face_landmarker.task always provides these at indices 468 / 473)
_LEFT_IRIS_CENTER  = 468
_RIGHT_IRIS_CENTER = 473

# Other canonical landmarks
_NOSE_TIP     = 4
_MOUTH_LEFT   = 61
_MOUTH_RIGHT  = 291
_CHIN         = 152
_FOREHEAD     = 10

# ArcFace canonical 5-point reference at 112×112 (scaled to target_size at runtime)
_ARCFACE_REF_112 = np.array(
    [
        [38.2946, 51.6963],   # right eye
        [73.5318, 51.5014],   # left eye
        [56.0252, 71.7366],   # nose tip
        [41.5493, 92.3655],   # mouth right
        [70.7299, 92.2041],   # mouth left
    ],
    dtype=np.float32,
)

# Model download URL and local cache path
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_DIR  = Path(__file__).parent.parent / "models"
_MODEL_PATH = _MODEL_DIR / "face_landmarker.task"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    """
    Output of the MediaPipe face alignment pipeline for a single face.

    Attributes:
        aligned_crop    : RGB uint8 array (H, W, 3). Eyes horizontal, centred.
                          Ready for CLIP ViT / EfficientNet input.
        landmarks_2d    : (N, 2) float32 — pixel coords in original image space.
                          N = 478 (468 mesh + 10 iris points).
        landmarks_3d    : (N, 3) float32 — same + normalised z depth.
        eye_left        : (x, y) left eye centre in original image.
        eye_right       : (x, y) right eye centre.
        nose_tip        : (x, y) nose tip.
        mouth_left      : (x, y) left mouth corner.
        mouth_right     : (x, y) right mouth corner.
        roll_deg        : In-plane rotation angle (degrees). Near 0 = level.
        bbox_input      : (x1, y1, x2, y2) bounding box that was used.
        detection_score : MediaPipe face presence confidence score.
    """
    aligned_crop:    np.ndarray                = field(repr=False)
    landmarks_2d:    np.ndarray                = field(repr=False)
    landmarks_3d:    np.ndarray                = field(repr=False)
    eye_left:        Tuple[float, float]       = field(default=(0.0, 0.0))
    eye_right:       Tuple[float, float]       = field(default=(0.0, 0.0))
    nose_tip:        Tuple[float, float]       = field(default=(0.0, 0.0))
    mouth_left:      Tuple[float, float]       = field(default=(0.0, 0.0))
    mouth_right:     Tuple[float, float]       = field(default=(0.0, 0.0))
    roll_deg:        float                     = 0.0
    bbox_input:      Tuple[int, int, int, int] = field(default=(0, 0, 0, 0))
    detection_score: float                     = 0.0

    def to_dict(self) -> Dict:
        """Serialise to dict (excludes large image/landmark arrays)."""
        return {
            "eye_left":           list(self.eye_left),
            "eye_right":          list(self.eye_right),
            "nose_tip":           list(self.nose_tip),
            "mouth_left":         list(self.mouth_left),
            "mouth_right":        list(self.mouth_right),
            "roll_deg":           round(self.roll_deg, 4),
            "bbox_input":         list(self.bbox_input),
            "detection_score":    round(self.detection_score, 6),
            "aligned_crop_shape": list(self.aligned_crop.shape),
            "num_landmarks":      len(self.landmarks_2d),
        }

    @property
    def interocular_distance(self) -> float:
        """Pixel distance between eye centres in original image space."""
        return float(np.linalg.norm(
            np.array(self.eye_left) - np.array(self.eye_right)
        ))


# ---------------------------------------------------------------------------
# FaceAligner
# ---------------------------------------------------------------------------

class FaceAligner:
    """
    Dense face alignment using MediaPipe Face Landmarker (Tasks API, 478 pts).

    This is Step 2 of IDS Layer 1 (Preprocessing). It refines the coarse
    5-point alignment from RetinaFace into a precisely normalised face crop
    using MediaPipe's dense 478-point mesh.

    Args:
        target_size        : (W, H) output aligned crop. Default (224, 224).
        padding            : Fractional padding added to bbox before passing
                             to MediaPipe — gives the model extra context.
                             Default 0.30 (30 %).
        min_detection_conf : Face detection confidence threshold. Default 0.5.
        min_presence_conf  : Face presence confidence threshold. Default 0.5.
        model_path         : Path to the face_landmarker.task model file.
                             If None, auto-downloads to engines/image-engine/models/.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        padding: float = 0.30,
        min_detection_conf: float = 0.5,
        min_presence_conf: float = 0.5,
        model_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.target_size = target_size
        self.padding = padding
        self.min_detection_conf = min_detection_conf
        self.min_presence_conf = min_presence_conf
        self.model_path = Path(model_path) if model_path else _MODEL_PATH

        # Lazily initialised
        self._landmarker = None

        logger.info(
            "FaceAligner initialised | target_size=%s | padding=%.2f",
            target_size, padding,
        )

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def _ensure_model(self) -> Path:
        """Download face_landmarker.task if not already cached."""
        if self.model_path.exists():
            logger.debug("Model found at: %s", self.model_path)
            return self.model_path

        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Downloading face_landmarker.task (~6 MB) ...\n  from: %s\n  to  : %s",
            _MODEL_URL, self.model_path,
        )

        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                if pct % 20 == 0:
                    logger.info("  Download progress: %d%%", pct)

        urllib.request.urlretrieve(_MODEL_URL, str(self.model_path), _progress)
        logger.info("Model downloaded successfully.")
        return self.model_path

    def _get_landmarker(self):
        """Lazily load the MediaPipe FaceLandmarker (Tasks API)."""
        if self._landmarker is None:
            try:
                import mediapipe as mp  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "The 'mediapipe' package is required.\n"
                    "Install it with:  pip install mediapipe"
                ) from exc

            model_path = self._ensure_model()

            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(model_path)
                ),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=self.min_detection_conf,
                min_face_presence_confidence=self.min_presence_conf,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            logger.info("MediaPipe FaceLandmarker (Tasks API) loaded.")
        return self._landmarker

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Accept file path or BGR array. Returns BGR uint8."""
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"OpenCV failed to load: {path}")
            return img
        if isinstance(image, np.ndarray):
            return image
        raise TypeError(f"Expected path or ndarray, got {type(image)}")

    def _padded_crop(
        self,
        img_bgr: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Crop a padded face ROI. Returns (crop_bgr, (x_offset, y_offset))."""
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        pad_x = int((x2 - x1) * self.padding)
        pad_y = int((y2 - y1) * self.padding)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)
        return img_bgr[cy1:cy2, cx1:cx2].copy(), (cx1, cy1)

    def _run_mediapipe(
        self,
        crop_bgr: np.ndarray,
        offset_xy: Tuple[int, int],
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Run MediaPipe FaceLandmarker on a BGR crop.

        Returns (landmarks_2d, landmarks_3d, score) in original image coords,
        or None if no face detected.
        """
        import mediapipe as mp  # noqa: PLC0415

        landmarker = self._get_landmarker()
        ch, cw = crop_bgr.shape[:2]
        ox, oy = offset_xy

        # Tasks API requires RGB mp.Image
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)

        detection_result = landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return None

        # Take first face (num_faces=1)
        face_lm = detection_result.face_landmarks[0]

        # Convert normalised [0,1] → pixel coords in original image space
        pts_2d = np.array(
            [[lm.x * cw + ox, lm.y * ch + oy] for lm in face_lm],
            dtype=np.float32,
        )
        pts_3d = np.array(
            [[lm.x * cw + ox, lm.y * ch + oy, lm.z] for lm in face_lm],
            dtype=np.float32,
        )

        # Use face presence confidence as score, safely handling None
        first_lm = detection_result.face_landmarks[0][0]
        presence = getattr(first_lm, "presence", None)
        score = float(presence) if presence is not None else 1.0

        return pts_2d, pts_3d, score

    def _extract_key_points(
        self,
        pts_2d: np.ndarray,
    ) -> Dict[str, Tuple[float, float]]:
        """Extract 5 canonical anchor points from the full landmark mesh."""

        def _pt(idx: int) -> Tuple[float, float]:
            return (float(pts_2d[idx, 0]), float(pts_2d[idx, 1]))

        def _avg(indices: List[int]) -> Tuple[float, float]:
            sub = pts_2d[indices]
            return (float(sub[:, 0].mean()), float(sub[:, 1].mean()))

        n = len(pts_2d)

        # Use iris centres (478 landmarks) when available — sub-pixel precision
        if n >= 478:
            eye_left  = _pt(_LEFT_IRIS_CENTER)
            eye_right = _pt(_RIGHT_IRIS_CENTER)
            logger.debug("Using iris centres (N=%d landmarks).", n)
        else:
            eye_left  = _avg(_LEFT_EYE_CONTOUR)
            eye_right = _avg(_RIGHT_EYE_CONTOUR)
            logger.debug("Using eye-contour averages (N=%d landmarks).", n)

        return {
            "eye_left":    eye_left,
            "eye_right":   eye_right,
            "nose_tip":    _pt(_NOSE_TIP),
            "mouth_left":  _pt(_MOUTH_LEFT),
            "mouth_right": _pt(_MOUTH_RIGHT),
        }

    @staticmethod
    def _compute_roll(
        eye_left: Tuple[float, float],
        eye_right: Tuple[float, float],
    ) -> float:
        """In-plane roll angle (degrees). 0 = level."""
        dx = eye_right[0] - eye_left[0]
        dy = eye_right[1] - eye_left[1]
        return float(np.degrees(np.arctan2(dy, dx)))

    def _compute_transform(
        self,
        key_pts: Dict[str, Tuple[float, float]],
    ) -> Optional[np.ndarray]:
        """Compute 2D similarity transform to ArcFace canonical reference."""
        src = np.array(
            [
                key_pts["eye_left"],    # MediaPipe 'left' = viewer's left (x~38)
                key_pts["eye_right"],   # MediaPipe 'right' = viewer's right (x~73)
                key_pts["nose_tip"],
                key_pts["mouth_left"],  # viewer's left (x~41)
                key_pts["mouth_right"], # viewer's right (x~70)
            ],
            dtype=np.float32,
        )
        dst = _ARCFACE_REF_112.copy()
        dst[:, 0] *= self.target_size[0] / 112.0
        dst[:, 1] *= self.target_size[1] / 112.0
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        return M

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align(
        self,
        image: Union[str, Path, np.ndarray],
        bbox: Tuple[int, int, int, int],
    ) -> Optional[AlignmentResult]:
        """
        Full MediaPipe alignment pipeline for a single face.

        1. Crops a padded ROI around `bbox`.
        2. Runs MediaPipe FaceLandmarker → 478 dense landmarks.
        3. Extracts 5 anchor key points (iris centres, nose, mouth corners).
        4. Computes a similarity transform to ArcFace canonical reference.
        5. Warps the FULL image → aligned 224×224 RGB crop.

        Args:
            image : Full image — path or BGR array.
            bbox  : (x1, y1, x2, y2) from RetinaFace / FaceDetector.

        Returns:
            AlignmentResult, or None if MediaPipe found no face.
        """
        img_bgr = self._load_image(image)
        logger.info("Aligning face | bbox=%s | image=%s", bbox, img_bgr.shape)

        crop_bgr, offset_xy = self._padded_crop(img_bgr, bbox)
        logger.debug("Padded crop shape=%s | offset=%s", crop_bgr.shape, offset_xy)

        mp_result = self._run_mediapipe(crop_bgr, offset_xy)
        if mp_result is None:
            logger.warning(
                "MediaPipe found no face in bbox=%s — "
                "face may be too small, heavily occluded, or extreme profile.", bbox
            )
            return None

        pts_2d, pts_3d, score = mp_result
        logger.info(
            "MediaPipe: %d landmarks | score=%.4f", len(pts_2d), score
        )

        key_pts = self._extract_key_points(pts_2d)
        roll    = self._compute_roll(key_pts["eye_left"], key_pts["eye_right"])
        logger.info(
            "roll=%.2f deg | eye_left=%s | eye_right=%s",
            roll,
            tuple(round(v, 1) for v in key_pts["eye_left"]),
            tuple(round(v, 1) for v in key_pts["eye_right"]),
        )

        M = self._compute_transform(key_pts)
        if M is None:
            logger.warning("Affine estimation failed for bbox=%s.", bbox)
            return None

        aligned_bgr = cv2.warpAffine(
            img_bgr, M, self.target_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

        return AlignmentResult(
            aligned_crop=aligned_rgb,
            landmarks_2d=pts_2d,
            landmarks_3d=pts_3d,
            eye_left=key_pts["eye_left"],
            eye_right=key_pts["eye_right"],
            nose_tip=key_pts["nose_tip"],
            mouth_left=key_pts["mouth_left"],
            mouth_right=key_pts["mouth_right"],
            roll_deg=roll,
            bbox_input=bbox,
            detection_score=score,
        )

    def align_batch(
        self,
        image: Union[str, Path, np.ndarray],
        bboxes: Sequence[Tuple[int, int, int, int]],
    ) -> List[Optional[AlignmentResult]]:
        """Align multiple faces from the same image."""
        img_bgr = self._load_image(image)
        results = []
        for i, bbox in enumerate(bboxes):
            logger.info("Processing face %d/%d ...", i + 1, len(bboxes))
            results.append(self.align(img_bgr, bbox))
        return results

    def visualize(
        self,
        image: Union[str, Path, np.ndarray],
        results: List[AlignmentResult],
        draw_mesh: bool = False,
        draw_keypoints: bool = True,
        draw_roll: bool = True,
    ) -> np.ndarray:
        """
        Draw MediaPipe landmarks and key anchor points on the image.

        Args:
            draw_mesh      : If True, draw all 478 landmark dots (dense).
            draw_keypoints : If True, draw the 5 anchor points + eye line.
            draw_roll      : If True, annotate in-plane roll angle.

        Returns:
            BGR annotated NumPy array.
        """
        img_bgr  = self._load_image(image)
        vis      = img_bgr.copy()
        MESH_CLR = (180, 180, 180)
        KP_CLR   = (0, 200, 255)
        TXT_CLR  = (255, 255, 255)
        BG_CLR   = (40, 40, 40)

        for result in results:
            if result is None:
                continue

            if draw_mesh:
                for pt in result.landmarks_2d:
                    cv2.circle(vis, (int(pt[0]), int(pt[1])), 1, MESH_CLR, -1)

            if draw_keypoints:
                kps = {
                    "LE": result.eye_left,
                    "RE": result.eye_right,
                    "N":  result.nose_tip,
                    "ML": result.mouth_left,
                    "MR": result.mouth_right,
                }
                for lbl, (lx, ly) in kps.items():
                    pt = (int(lx), int(ly))
                    cv2.circle(vis, pt, 5, KP_CLR, -1)
                    cv2.putText(
                        vis, lbl, (pt[0] + 6, pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, KP_CLR, 1, cv2.LINE_AA,
                    )
                # Eye line
                le_pt = (int(result.eye_left[0]),  int(result.eye_left[1]))
                re_pt = (int(result.eye_right[0]), int(result.eye_right[1]))
                cv2.line(vis, le_pt, re_pt, KP_CLR, 1, cv2.LINE_AA)

            if draw_roll:
                x1, y1 = result.bbox_input[:2]
                label = f"roll={result.roll_deg:+.1f}deg"
                (tw, th), bl = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1
                )
                cv2.rectangle(vis, (x1, y1-th-bl-4), (x1+tw+4, y1), BG_CLR, -1)
                cv2.putText(
                    vis, label, (x1+2, y1-bl-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, TXT_CLR, 1, cv2.LINE_AA,
                )

        return vis

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            logger.info("FaceAligner closed.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return (
            f"FaceAligner(target_size={self.target_size}, "
            f"padding={self.padding})"
        )
