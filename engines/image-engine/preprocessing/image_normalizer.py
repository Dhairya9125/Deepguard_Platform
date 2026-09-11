"""
DeepGuard Platform — Image Detection Subsystem (IDS)
Module  : engines.image-engine.preprocessing.image_normalizer
Layer   : Layer 1 — Preprocessing
Task    : Image Normalisation using OpenCV

Architecture Reference:
  - Standardizing inputs for Layer 2 Feature Extraction (CLIP ViT / EfficientNet).
  - Prepares the cropped/aligned face arrays (and full images) by normalizing 
    them using ImageNet mean and standard deviation.

Usage:
    from preprocessing.image_normalizer import ImageNormalizer

    normalizer = ImageNormalizer()
    # Output from FaceAligner is RGB uint8 (H, W, 3)
    tensor_input = normalizer.normalize(aligned_crop_rgb)
    print(tensor_input.shape)  # (3, 224, 224) if to_chw=True
"""

import logging
from typing import Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

class ImageNormalizer:
    """
    Normalizes images for Deep Learning backbones using OpenCV.

    Pipeline:
      1. Ensure RGB format.
      2. Convert uint8 [0, 255] to float32 [0.0, 1.0].
      3. Subtract Mean.
      4. Divide by Standard Deviation.
      5. Optionally transpose from HWC to CHW format (PyTorch standard).

    Args:
        mean    : RGB mean values (default: ImageNet).
        std     : RGB standard deviation values (default: ImageNet).
        to_chw  : If True, transposes output to (Channels, Height, Width).
    """

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        to_chw: bool = True,
    ) -> None:
        # Convert to 4-element tuples for OpenCV Scalar compatibility
        self.mean = (float(mean[0]), float(mean[1]), float(mean[2]), 0.0)
        self.std = (float(std[0]), float(std[1]), float(std[2]), 0.0)
        self.to_chw = to_chw

        logger.debug(
            "ImageNormalizer initialised | mean=%s | std=%s | to_chw=%s",
            mean, std, to_chw
        )

    def normalize(self, image: np.ndarray, is_bgr: bool = False) -> np.ndarray:
        """
        Normalize the input image using OpenCV operations.

        Args:
            image  : The input image array (uint8 or float32).
            is_bgr : Set to True if the input is BGR. By default, assumes RGB 
                     (e.g. from FaceAligner output).

        Returns:
            Normalized float32 NumPy array.
        """
        # Ensure correct channel ordering
        if is_bgr:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img = image

        # Scale to [0, 1] if the image is uint8
        if img.dtype == np.uint8:
            img_float = img.astype(np.float32) / 255.0
        elif img.dtype == np.float32:
            img_float = img.copy()
        else:
            img_float = img.astype(np.float32)

        # Apply mean/std normalization using OpenCV functions
        # cv2.subtract and cv2.divide take tuples (Scalars) for channel-wise ops
        normalized = cv2.subtract(img_float, self.mean)
        normalized = cv2.divide(normalized, self.std)

        # Convert to Channel-First format for PyTorch/ONNX if requested
        if self.to_chw:
            # Transpose (H, W, C) -> (C, H, W)
            normalized = np.transpose(normalized, (2, 0, 1))
            # Ensure it is C-contiguous in memory after transpose
            normalized = np.ascontiguousarray(normalized)

        return normalized

    def __repr__(self) -> str:
        return f"ImageNormalizer(to_chw={self.to_chw})"
