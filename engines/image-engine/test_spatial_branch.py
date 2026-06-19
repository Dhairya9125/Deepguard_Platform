import torch

from img_feature_extraction.spatial_branch import ClipSpatialBranch

def test():
    print("Testing Layer 2 Branch A: ClipSpatialBranch...")
    
    # Initialize the model
    # Warning: This will download ~1.7 GB of model weights on first run
    print("Initializing CLIP ViT-L/14 model (Downloading if necessary)...")
    model = ClipSpatialBranch(proj_dim=512, freeze_backbone=True)
    
    # Verify the hidden size is what we expect for ViT-L/14 (1024)
    print(f"Backbone hidden size: {model.embed_dim}")
    assert model.embed_dim == 1024, "ViT-L/14 should have 1024-d hidden size"
    
    # Create a dummy input tensor matching FaceAligner/Normalizer output shape (B, C, H, W)
    # B=2 to test batched processing
    dummy_input = torch.randn(2, 3, 224, 224)
    print(f"Dummy input shape: {dummy_input.shape}")
    
    # Forward pass
    print("Running forward pass...")
    output = model(dummy_input)
    
    print(f"Projected output shape: {output.shape}")
    
    # Assertions
    assert output.shape == (2, 512), "Output should be (Batch, proj_dim)"
    assert output.requires_grad == True, "Output must require grad for the trainable projection head"
    
    # Check freezing logic
    backbone_frozen = True
    for name, param in model.backbone.named_parameters():
        if param.requires_grad:
            backbone_frozen = False
            print(f"ERROR: Backbone parameter {name} is not frozen!")
            
    proj_trainable = True
    for name, param in model.projection.named_parameters():
        if not param.requires_grad:
            proj_trainable = False
            print(f"ERROR: Projection parameter {name} is frozen!")
            
    assert backbone_frozen, "Backbone is not fully frozen."
    assert proj_trainable, "Projection head is not fully trainable."
    
    print("Test passed! Backbone is frozen and projection head is trainable.")

if __name__ == "__main__":
    test()
