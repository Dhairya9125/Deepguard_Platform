import torch
import cv2
import numpy as np
from pathlib import Path

from feature_extraction.discrepancy_branch import DiscrepancyBranch, patch_shuffle

def test():
    print("Testing Layer 2 Branch C: DiscrepancyBranch (D3 Siamese)...")
    
    # 1. Load an image to test patch shuffling visually
    img_path = Path("test_outputs/aligned_mediapipe_1.jpg")
    if img_path.exists():
        img_np = cv2.imread(str(img_path))
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    else:
        # Fallback to a gradient pattern if no image exists so we can see the shuffle
        img_np = np.zeros((224, 224, 3), dtype=np.uint8)
        for i in range(224):
            img_np[:, i, 1] = i
            img_np[i, :, 2] = i
            
    # Convert to PyTorch Tensor (B, C, H, W)
    img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0)
    
    # Run patch shuffle utility directly to save an output image
    print("Executing patch_shuffle (4x4 grid)...")
    shuffled_tensor = patch_shuffle(img_tensor, grid_size=4)
    
    # Save the shuffled image
    out_dir = Path("test_outputs")
    out_dir.mkdir(exist_ok=True)
    
    shuffled_np = shuffled_tensor.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    cv2.imwrite(str(out_dir / "d3_patch_shuffled.jpg"), cv2.cvtColor(shuffled_np, cv2.COLOR_RGB2BGR))
    print("Shuffled image saved to test_outputs/d3_patch_shuffled.jpg")
    
    # 2. Test the Siamese Network
    print("Initializing DiscrepancyBranch (ResNet18 Siamese)...")
    model = DiscrepancyBranch(grid_size=4, proj_dim=512)
    
    # Batch input to verify dataloader compatibility
    batch_tensor = img_tensor.repeat(2, 1, 1, 1)
    
    print("Running forward pass (Original vs Shuffled)...")
    output = model(batch_tensor)
    
    print(f"Projected discrepancy output shape: {output.shape}")
    
    # Assertions
    assert output.shape == (2, 512), "Output should be (Batch, proj_dim)"
    assert output.requires_grad == True, "Output must require grad for training"
    
    # Ensure backbone is NOT frozen
    backbone_trainable = any(p.requires_grad for p in model.backbone.parameters())
    assert backbone_trainable, "ResNet backbone MUST be trainable for Branch C"
    
    print("Test passed! D3 Siamese discrepancy pipeline is fully functional.")

if __name__ == "__main__":
    test()
