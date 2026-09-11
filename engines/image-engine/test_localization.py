import torch
import cv2
import numpy as np
from pathlib import Path

from localization.localization_engine import LocalizationEngine

def test():
    print("Testing Layer 4: Localization Engine...")
    
    # Initialize Engine
    engine = LocalizationEngine(threshold=0.6)
    
    # 1. Create a simulated soft heatmap (Batch=1, C=1, H=224, W=224)
    # Background is mostly 0.1 probability
    heatmap_np = np.ones((224, 224), dtype=np.float32) * 0.1
    
    # Create two "hotspots" simulating manipulated regions (e.g., two eyes swapped)
    # Hotspot 1
    cv2.circle(heatmap_np, (70, 80), 30, 0.9, -1)
    # Hotspot 2
    cv2.circle(heatmap_np, (150, 80), 25, 0.85, -1)
    
    # Add a soft blur to simulate neural network upsampling output
    heatmap_np = cv2.GaussianBlur(heatmap_np, (15, 15), 0)
    
    # Convert to Tensor (B, 1, H, W)
    heatmap_tensor = torch.from_numpy(heatmap_np).unsqueeze(0).unsqueeze(0)
    
    # 2. Process through Layer 4
    print("Running heatmap through Localization Engine...")
    results = engine.process(heatmap_tensor)
    
    res = results[0]
    mask = res["manipulated_mask"]
    boundaries = res["blend_boundaries"]
    bboxes = res["bounding_boxes"]
    
    print(f"Extracted {len(bboxes)} bounding boxes: {bboxes}")
    
    # 3. Visualization
    # Create a blank RGB image to draw on
    vis = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Draw heatmap as red channel
    vis[:, :, 2] = (heatmap_np * 255).astype(np.uint8)
    
    # Draw boundaries as stark green
    vis[boundaries == 255] = [0, 255, 0]
    
    # Draw bounding boxes in blue
    for (x, y, w, h) in bboxes:
        cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    out_dir = Path("test_outputs")
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / "localization_output.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print("Localization visualization saved to test_outputs/localization_output.jpg")
    
    # Assertions
    assert len(bboxes) == 2, "Should detect exactly 2 hot regions"
    assert mask.shape == (224, 224)
    assert boundaries.shape == (224, 224)
    
    print("Test passed! Localization Engine successfully extracted discrete anomalies.")

if __name__ == "__main__":
    test()
