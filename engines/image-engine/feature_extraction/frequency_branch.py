"""
DeepGuard Platform — Image Detection Subsystem (IDS)
Module  : engines.image-engine.feature_extraction.frequency_branch
Layer   : Layer 2 — Feature Extraction (Branch B)

Architecture Reference:
  - Branch B (Frequency Features): EfficientNet Backbone
  - Processes a 3-channel frequency stack (FFT, DCT, Wavelet)
  - Trainable feature extractor to recognize high-frequency spectral grids.
"""

import logging

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

logger = logging.getLogger(__name__)


class EfficientNetFrequencyBranch(nn.Module):
    """
    Branch B: Frequency Feature Extractor.
    
    Uses EfficientNet (B0) to analyze the 3-channel (FFT, DCT, Wavelet) 
    frequency spectrum stacked tensor.
    
    Unlike Branch A, this backbone is fully trainable because the generic 
    ImageNet weights must be fine-tuned to understand synthetic spectral grids
    and compression inconsistencies instead of standard spatial objects.
    """

    def __init__(
        self, 
        proj_dim: int = 512,
        pretrained: bool = True
    ) -> None:
        super().__init__()
        
        self.proj_dim = proj_dim
        
        logger.info(f"Loading EfficientNet-B0 backbone | pretrained={pretrained}")
        
        # Load the EfficientNet-B0 backbone
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        
        # We need to replace the final classifier head.
        # EfficientNet-B0 classifier is a Sequential: 
        # (0): Dropout(p=0.2, inplace=True)
        # (1): Linear(in_features=1280, out_features=1000, bias=True)
        
        in_features = self.backbone.classifier[1].in_features
        
        # Replace with our projection head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, proj_dim)
        )
        
        logger.info(f"EfficientNetFrequencyBranch initialised | out_features={proj_dim}")

    def forward(self, frequency_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            frequency_tensor: Tensor of shape (Batch, 3, 224, 224) containing
                              the stacked FFT, DCT, and Wavelet magnitude spectra.
                          
        Returns:
            Projected feature vector of shape (Batch, proj_dim).
        """
        # Pass through the trainable EfficientNet backbone and projection head
        return self.backbone(frequency_tensor)
