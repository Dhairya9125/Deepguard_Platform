import torch
from fusion.ids_fusion_engine import IDSFusionEngine

def test():
    print("Testing Layer 3: IDSFusionEngine...")
    
    # Initialize Engine
    print("Initializing Fusion Engine...")
    model = IDSFusionEngine(embed_dim=512, num_branches=5)
    
    # Generate dummy embeddings for a batch of 2
    batch_size = 2
    embed_dim = 512
    
    branch_embeddings = {
        "spatial": torch.randn(batch_size, embed_dim),
        "frequency": torch.randn(batch_size, embed_dim),
        "discrepancy": torch.randn(batch_size, embed_dim),
        "noise": torch.randn(batch_size, embed_dim),
        "fingerprint": torch.randn(batch_size, embed_dim),
    }
    
    print("Running forward pass...")
    outputs = model(branch_embeddings)
    
    # Check outputs
    assert "fake_probability" in outputs
    assert "manipulation_heatmap" in outputs
    assert "OOD_score" in outputs
    assert "artifact_embedding" in outputs
    
    fake_prob = outputs["fake_probability"]
    heatmap = outputs["manipulation_heatmap"]
    ood = outputs["OOD_score"]
    embed = outputs["artifact_embedding"]
    
    print(f"fake_probability shape: {fake_prob.shape}")
    print(f"manipulation_heatmap shape: {heatmap.shape}")
    print(f"OOD_score shape: {ood.shape}")
    print(f"artifact_embedding shape: {embed.shape}")
    
    assert fake_prob.shape == (batch_size, 1), "Probability must be (B, 1)"
    assert heatmap.shape == (batch_size, 1, 224, 224), "Heatmap must be (B, 1, 224, 224)"
    assert ood.shape == (batch_size, 1), "OOD score must be (B, 1)"
    assert embed.shape == (batch_size, embed_dim * 5), f"Artifact embedding must be flattened combination"
    
    print("Test passed! IDSFusionEngine correctly integrates branches and produces all 4 required outputs.")

if __name__ == "__main__":
    test()
