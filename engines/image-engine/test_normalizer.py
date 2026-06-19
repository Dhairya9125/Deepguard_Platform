import numpy as np
from img_preprocessing import ImageNormalizer

def test():
    print("Testing ImageNormalizer...")
    
    # Create a dummy image (H=224, W=224, C=3) in uint8 [0, 255]
    img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    
    # Init normalizer
    norm = ImageNormalizer(to_chw=True)
    
    # Normalize
    tensor = norm.normalize(img)
    
    print(f"Original shape: {img.shape}, dtype: {img.dtype}")
    print(f"Normalized shape: {tensor.shape}, dtype: {tensor.dtype}")
    
    # Checks
    assert tensor.shape == (3, 224, 224), "Output shape must be CHW"
    assert tensor.dtype == np.float32, "Output dtype must be float32"
    
    print("Success! Normalization pipeline works.")

if __name__ == "__main__":
    test()
