"""
DeepGuard Platform — Image Detection Subsystem (IDS)
Module  : engines.image-engine.preprocessing.image_augmenter
Layer   : Layer 1 — Preprocessing
Task    : Image Augmentation using OpenCV / NumPy (Zero-Dependency)

Architecture Reference:
  - Augmentation for Robustness: apply JPEG compression simulation, 
    Gaussian noise, resizing — ensure detector works on real-world compressed images.
  - Social Media Simulation: Fine-tune on samples processed through simulated 
    social media pipelines.
    
    NOTE: Meta's AugLy and Albumentations failed to install due to missing 
    C++ build tools on Python 3.14 Windows. We have implemented the exact 
    same standard augmentations using natively available OpenCV and NumPy.

Usage:
    from img_preprocessing.image_augmenter import ImageAugmenter

    augmenter = ImageAugmenter(apply_prob=0.8)
    
    # Accepts RGB NumPy arrays
    aug_img = augmenter.apply_social_media_compression(original_image)
"""

import logging
import random
from typing import Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ImageAugmenter:
    """
    Robustness augmentation using OpenCV and NumPy.
    
    Simulates real-world social media degradation (JPEG compression, noise, blur)
    to prevent deepfake detectors from overfitting to pristine, high-frequency 
    GAN artifacts that do not survive upload processes.

    Args:
        apply_prob: Overall probability [0, 1] that any augmentation is applied.
                    Useful for training where you want some pristine samples.
    """

    def __init__(self, apply_prob: float = 1.0) -> None:
        self.apply_prob = apply_prob
        logger.info("ImageAugmenter initialised (Native OpenCV) | apply_prob=%.2f", apply_prob)

    def _ensure_numpy_rgb(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Convert input to uint8 NumPy array (RGB)."""
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        if isinstance(image, np.ndarray):
            if image.dtype == np.float32:
                return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
            return image
        raise TypeError(f"Expected PIL Image or NumPy array, got {type(image)}")

    def apply_social_media_compression(
        self, 
        image: Union[np.ndarray, Image.Image],
        quality_min: int = 30,
        quality_max: int = 90
    ) -> np.ndarray:
        """
        Simulates Social Media JPEG compression.
        Randomly compresses the image to a quality between quality_min and quality_max.
        """
        img_np = self._ensure_numpy_rgb(image)
        if random.random() > self.apply_prob:
            return img_np
            
        quality = random.randint(quality_min, quality_max)
        
        # OpenCV works with BGR natively, but for JPEG compression of RGB arrays, 
        # the lossy blocking artifacts will apply correctly regardless of channel order,
        # However, to be perfectly accurate for color sub-sampling, convert to BGR first.
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Encode to JPEG in memory
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, encoded_img = cv2.imencode('.jpg', img_bgr, encode_param)
        
        if not success:
            return img_np
            
        # Decode back to array
        decoded_bgr = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)

    def apply_noise(
        self, 
        image: Union[np.ndarray, Image.Image],
        var_limit: tuple = (10.0, 50.0)
    ) -> np.ndarray:
        """
        Simulates sensor noise or social media processing noise using Gaussian noise.
        """
        img_np = self._ensure_numpy_rgb(image)
        if random.random() > self.apply_prob:
            return img_np

        # Generate Gaussian noise
        std = random.uniform(var_limit[0] ** 0.5, var_limit[1] ** 0.5)
        noise = np.random.normal(0, std, img_np.shape).astype(np.float32)
        
        # Add noise and clip
        noisy_img = img_np.astype(np.float32) + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        
        return noisy_img
        
    def apply_random_pipeline(
        self, 
        image: Union[np.ndarray, Image.Image]
    ) -> np.ndarray:
        """
        Applies a random combination of augmentations (Compression, Blur, Noise, Downscale).
        Used during the training dataloader phase to simulate a complete upload pipeline.
        """
        img_np = self._ensure_numpy_rgb(image)
        if random.random() > self.apply_prob:
            return img_np
            
        h, w = img_np.shape[:2]
        result = img_np.copy()
            
        # 1. Potential slight downscale & upscale (simulating social media resize)
        if random.random() > 0.7:
            scale = random.uniform(0.7, 0.9)
            small = cv2.resize(result, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
            result = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
            
        # 2. Add some sensor/processing noise
        if random.random() > 0.5:
            # Force apply noise locally
            orig_prob = self.apply_prob
            self.apply_prob = 1.0
            result = self.apply_noise(result, var_limit=(10.0, 30.0))
            self.apply_prob = orig_prob
            
        # 3. Add slight motion/compression blur
        if random.random() > 0.7:
            k_size = random.choice([3, 5])
            result = cv2.GaussianBlur(result, (k_size, k_size), 0)
            
        # 4. Always apply some degree of JPEG compression last
        orig_prob = self.apply_prob
        self.apply_prob = 1.0
        result = self.apply_social_media_compression(result, quality_min=30, quality_max=85)
        self.apply_prob = orig_prob
        
        return result

    def __repr__(self) -> str:
        return f"ImageAugmenter(apply_prob={self.apply_prob})"
