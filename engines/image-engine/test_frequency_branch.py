import torch
import cv2
import numpy as np
from pathlib import Path

from img_preprocessing.frequency_analyzer import FrequencyAnalyzer
from img_feature_extraction.frequency_branch import EfficientNetFrequencyBranch

def test():
    print("Testing Layer 2 Branch B: EfficientNetFrequencyBranch...")
    
    # 1. Initialize Frequency Analyzer
    analyzer = FrequencyAnalyzer(log_scale=True)
    
    # Create or load an image
    img_path = Path("test_outputs/aligned_mediapipe_1.jpg")
    if img_path.exists():
        img = cv2.imread(str(img_path))
    else:
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
    print("Generating stacked frequency tensor (FFT, DCT, Wavelet)...")
    freq_tensor = analyzer.get_stacked_frequency_tensor(img)
    print(f"Single item tensor shape: {freq_tensor.shape}")
    
    # Add batch dimension to simulate a DataLoader (B, 3, H, W)
    batch_tensor = freq_tensor.unsqueeze(0).repeat(2, 1, 1, 1) # Batch size 2
    print(f"Batched input shape: {batch_tensor.shape}")
    
    # 2. Initialize Branch B
    print("Initializing EfficientNet-B0 model...")
    model = EfficientNetFrequencyBranch(proj_dim=512)
    
    # 3. Forward Pass
    print("Running forward pass...")
    output = model(batch_tensor)
    
    print(f"Projected output shape: {output.shape}")
    
    # Assertions
    assert output.shape == (2, 512), "Output should be (Batch, proj_dim)"
    assert output.requires_grad == True, "Output must require grad for training"
    
    # Ensure backbone is NOT frozen (unlike Branch A)
    backbone_trainable = any(p.requires_grad for p in model.backbone.features.parameters())
    assert backbone_trainable, "EfficientNet backbone MUST be trainable for Branch B"
    
    print("Test passed! Frequency stacking and EfficientNet pipeline are fully functional.")

if __name__ == "__main__":
    test()
