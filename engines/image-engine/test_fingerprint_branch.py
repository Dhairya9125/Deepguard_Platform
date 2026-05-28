import torch

from feature_extraction.fingerprint_branch import FingerprintBranch

def test():
    print("Testing Layer 2 Branch E: FingerprintBranch (Swin Transformer)...")
    
    # 1. Initialize the branch
    print("Initializing FingerprintBranch (swin_t)...")
    # Will download ~114MB if not cached
    model = FingerprintBranch(proj_dim=512)
    
    # 2. Generate dummy input tensor (B, 3, H, W)
    batch_size = 2
    img_tensor = torch.randn(batch_size, 3, 224, 224)
    print(f"Input tensor shape: {img_tensor.shape}")
    
    # 3. Forward Pass
    print("Running forward pass through Swin Attention blocks...")
    output = model(img_tensor)
    
    print(f"Projected output shape: {output.shape}")
    
    # Assertions
    assert output.shape == (batch_size, 512), "Output should be (Batch, proj_dim)"
    assert output.requires_grad == True, "Output must require grad for training"
    
    # Ensure backbone IS trainable
    backbone_trainable = any(p.requires_grad for p in model.backbone.features.parameters())
    assert backbone_trainable, "Swin Transformer backbone MUST be trainable for Branch E"
    
    print("Test passed! Swin Transformer pipeline is fully functional and trainable.")

if __name__ == "__main__":
    test()
