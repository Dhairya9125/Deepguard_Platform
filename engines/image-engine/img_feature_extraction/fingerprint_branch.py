"""
DeepGuard Platform — Image Detection Subsystem (IDS)
Module  : engines.image-engine.feature_extraction.fingerprint_branch
Layer   : Layer 2 — Feature Extraction (Branch E)

Architecture Reference:
  - Branch E (GAN/Diffusion Fingerprints): Swin Transformer (Shifted Window)
  - Leverages hierarchical self-attention to track non-local, repeating 
    latent space artifacts (like upsampling grids or denoising signatures).
"""

import logging
import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights

logger = logging.getLogger(__name__)


class FingerprintBranch(nn.Module):
    """
    Branch E: GAN & Diffusion Fingerprint Extractor.
    
    Uses a Shifted Window (Swin) Transformer to capture non-local microscopic
    inconsistencies. While convolutional networks analyze local texture patches, 
    the Swin Transformer builds global attention maps to identify repeating 
    algorithmic traces scattered across the entire image.
    """

    def __init__(
        self, 
        proj_dim: int = 512,
        pretrained: bool = True
    ) -> None:
        super().__init__()
        
        self.proj_dim = proj_dim
        
        logger.info(f"Loading Swin-T backbone for Fingerprint Branch | pretrained={pretrained}")
        
        # Load Swin-Tiny (swin_t) backbone
        weights = Swin_T_Weights.DEFAULT if pretrained else None
        self.backbone = swin_t(weights=weights)
        
        # Replace the final classification head
        # In torchvision's swin_t, the classification head is named `head`
        in_features = self.backbone.head.in_features
        
        # Replace with our projection MLP
        self.backbone.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, proj_dim),
            nn.GELU()
        )
        
        logger.info(f"FingerprintBranch initialised | out_features={proj_dim}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            pixel_values: Tensor of shape (Batch, 3, H, W).
                          
        Returns:
            Projected fingerprint feature vector of shape (Batch, proj_dim).
        """
        # Pass through the trainable Swin Transformer
        projected = self.backbone(pixel_values)
        
        return projected
