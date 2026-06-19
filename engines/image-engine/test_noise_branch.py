import torch
import cv2
import numpy as np
from pathlib import Path

from img_feature_extraction.noise_branch import NoiseResidualBranch, SRMConv2d

def test():
    print("Testing Layer 2 Branch D: NoiseResidualBranch (SRM)...")
    
    # 1. Load an image to test SRM visually
    img_path = Path("test_outputs/aligned_mediapipe_1.jpg")
    if img_path.exists():
        img_np = cv2.imread(str(img_path))
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    else:
        img_np = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            
    # Convert to PyTorch Tensor [0, 1]
    img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # 2. Test SRM visual extraction
    print("Executing SRM filtering...")
    srm = SRMConv2d()
    noise_map = srm(img_tensor)
    
    # Process for visualization: noise is around 0 with small variance.
    # We multiply by 10 and add 0.5 to make the neutral grey background visible.
    vis_tensor = noise_map.squeeze(0).permute(1, 2, 0).numpy()
    vis_tensor = np.clip((vis_tensor * 10) + 0.5, 0.0, 1.0) * 255.0
    vis_tensor = vis_tensor.astype(np.uint8)
    
    out_dir = Path("test_outputs")
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / "srm_noise_residual.jpg"), cv2.cvtColor(vis_tensor, cv2.COLOR_RGB2BGR))
    print("SRM noise map saved to test_outputs/srm_noise_residual.jpg")
    
    # 3. Test the Full Network
    print("Initializing NoiseResidualBranch (ResNet18)...")
    model = NoiseResidualBranch(proj_dim=512)
    
    batch_tensor = img_tensor.repeat(2, 1, 1, 1)
    
    print("Running forward pass...")
    output = model(batch_tensor)
    
    print(f"Projected noise output shape: {output.shape}")
    
    # Assertions
    assert output.shape == (2, 512), "Output should be (Batch, proj_dim)"
    assert output.requires_grad == True, "Output must require grad for training"
    
    # Ensure backbone IS trainable
    backbone_trainable = any(p.requires_grad for p in model.backbone.parameters())
    assert backbone_trainable, "ResNet backbone MUST be trainable for Branch D"
    
    # Ensure SRM IS frozen
    srm_frozen = not next(model.srm_layer.parameters()).requires_grad
    assert srm_frozen, "SRM filters MUST be frozen"
    
    print("Test passed! SRM pipeline and noise residual learning are fully functional.")

if __name__ == "__main__":
    test()
