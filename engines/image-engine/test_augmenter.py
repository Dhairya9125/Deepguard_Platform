import os
from pathlib import Path

import cv2
import numpy as np

from preprocessing.image_augmenter import ImageAugmenter

def test():
    print("Testing ImageAugmenter (Albumentations)...")
    
    # We need a real image to see the compression and noise clearly
    # We can use the aligned face output from previous test if it exists
    img_path = Path("test_outputs/aligned_mediapipe_1.jpg")
    
    if img_path.exists():
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Fallback to random image
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
    print(f"Loaded image shape: {img.shape}")
    
    augmenter = ImageAugmenter(apply_prob=1.0)  # Always apply for testing
    
    # Test 1: Compression
    compressed = augmenter.apply_social_media_compression(img, quality_min=10, quality_max=20)
    
    # Test 2: Noise
    noised = augmenter.apply_noise(img, var_limit=(50.0, 100.0))
    
    # Test 3: Full Pipeline
    pipeline_out = augmenter.apply_random_pipeline(img)
    
    # Save outputs
    out_dir = Path("test_outputs")
    out_dir.mkdir(exist_ok=True)
    
    # Convert back to BGR for saving
    cv2.imwrite(str(out_dir / "aug_compressed.jpg"), cv2.cvtColor(compressed, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "aug_noised.jpg"), cv2.cvtColor(noised, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "aug_pipeline.jpg"), cv2.cvtColor(pipeline_out, cv2.COLOR_RGB2BGR))
    
    print("Augmentations applied successfully! Outputs saved to test_outputs/aug_*.jpg")

if __name__ == "__main__":
    test()
