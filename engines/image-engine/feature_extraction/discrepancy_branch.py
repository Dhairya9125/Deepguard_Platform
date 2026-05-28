"""
DeepGuard Platform — Image Detection Subsystem (IDS)
Module  : engines.image-engine.feature_extraction.discrepancy_branch
Layer   : Layer 2 — Feature Extraction (Branch C)

Architecture Reference:
  - Branch C (Discrepancy Features): D3-inspired Siamese Network
  - Processes original image vs patch-shuffled image to learn universal 
    local blending artifacts over global semantic fingerprints.
"""

import logging
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

logger = logging.getLogger(__name__)


def patch_shuffle(x: torch.Tensor, grid_size: int = 4) -> torch.Tensor:
    """
    Shuffles an image tensor by dividing it into a grid and randomly permuting the patches.
    
    Args:
        x: Tensor of shape (B, C, H, W)
        grid_size: Number of patches per side (e.g., 4 means a 4x4 grid)
        
    Returns:
        Shuffled tensor of shape (B, C, H, W)
    """
    B, C, H, W = x.size()
    
    # Ensure dimensions are cleanly divisible by grid_size
    if H % grid_size != 0 or W % grid_size != 0:
        raise ValueError(f"Image dimensions ({H}, {W}) must be divisible by grid_size {grid_size}")
        
    patch_h = H // grid_size
    patch_w = W // grid_size
    
    # 1. Unfold into patches: (B, C, grid_size, patch_h, grid_size, patch_w)
    patches = x.view(B, C, grid_size, patch_h, grid_size, patch_w)
    
    # 2. Transpose to group grid dimensions: (B, C, grid_size, grid_size, patch_h, patch_w)
    patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
    
    # 3. Flatten the grid dimensions into a single sequence of patches: (B, C, num_patches, patch_h, patch_w)
    num_patches = grid_size * grid_size
    patches = patches.view(B, C, num_patches, patch_h, patch_w)
    
    # 4. Shuffle patches independently for each item in the batch
    shuffled = torch.zeros_like(patches)
    for b in range(B):
        perm = torch.randperm(num_patches, device=x.device)
        shuffled[b] = patches[b, :, perm, :, :]
        
    # 5. Reshape back into image format
    shuffled = shuffled.view(B, C, grid_size, grid_size, patch_h, patch_w)
    shuffled = shuffled.permute(0, 1, 2, 4, 3, 5).contiguous()
    shuffled_img = shuffled.view(B, C, H, W)
    
    return shuffled_img


class DiscrepancyBranch(nn.Module):
    """
    Branch C: Discrepancy Feature Extractor (Siamese Network).
    
    Passes both the original and patch-shuffled images through a shared ResNet backbone.
    The absolute discrepancy between the two feature representations is then projected
    into a dense embedding. This forces the network to learn local pixel-blending anomalies
    rather than memorizing semantic objects (which are destroyed in the shuffled image).
    """

    def __init__(
        self, 
        grid_size: int = 4,
        proj_dim: int = 512,
        pretrained: bool = True
    ) -> None:
        super().__init__()
        
        self.grid_size = grid_size
        self.proj_dim = proj_dim
        
        logger.info(f"Loading Siamese ResNet18 backbone | pretrained={pretrained}")
        
        # We use ResNet18 as it is computationally light for dual-branch operations
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        
        # Extract features before the final classification layer
        in_features = self.backbone.fc.in_features
        # We replace the final FC layer with Identity to just get the flattened pooling
        self.backbone.fc = nn.Identity()
        
        # The projection head takes the discrepancy vector (same size as in_features)
        self.projection = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, proj_dim),
            nn.GELU()
        )
        
        logger.info(f"DiscrepancyBranch initialised | grid_size={grid_size} | out_features={proj_dim}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (Siamese).
        
        Args:
            pixel_values: Tensor of shape (Batch, 3, H, W).
                          
        Returns:
            Projected discrepancy feature vector of shape (Batch, proj_dim).
        """
        # 1. Generate patch-shuffled images on the fly
        shuffled_values = patch_shuffle(pixel_values, grid_size=self.grid_size)
        
        # 2. Extract features from both using the shared backbone
        orig_features = self.backbone(pixel_values)
        shuff_features = self.backbone(shuffled_values)
        
        # 3. Compute absolute discrepancy
        discrepancy = torch.abs(orig_features - shuff_features)
        
        # 4. Project
        projected = self.projection(discrepancy)
        
        return projected
