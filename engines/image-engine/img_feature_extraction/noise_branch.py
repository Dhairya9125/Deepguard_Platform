"""
DeepGuard Platform — Image Detection Subsystem (IDS)
Module  : engines.image-engine.feature_extraction.noise_branch
Layer   : Layer 2 — Feature Extraction (Branch D)

Architecture Reference:
  - Branch D (Noise Residual Features): SRM Filters + CNN
  - Suppresses image content to expose the high-frequency physical sensor 
    noise (e.g. PRNU traces) and compression residuals.
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights

logger = logging.getLogger(__name__)


class SRMConv2d(nn.Module):
    """
    Spatial Rich Model (SRM) Filter Layer.
    
    Applies fixed, non-trainable high-pass filters to extract the noise residuals 
    from an image while suppressing the low-frequency semantic content.
    We use the 3 fundamental SRM kernels (Horizontal, Vertical, Diagonal) to construct
    a 3-channel output that perfectly matches a standard CNN backbone input.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        # Define the 3 core SRM filters (3x3)
        # Filter 1: Horizontal edges / residuals
        filter1 = np.array([
            [ 0,  0,  0],
            [-1,  2, -1],
            [ 0,  0,  0]
        ], dtype=np.float32) / 2.0
        
        # Filter 2: Vertical edges / residuals
        filter2 = np.array([
            [ 0, -1,  0],
            [ 0,  2,  0],
            [ 0, -1,  0]
        ], dtype=np.float32) / 2.0
        
        # Filter 3: Diagonal / Cross residuals
        filter3 = np.array([
            [-1,  2, -1],
            [ 2, -4,  2],
            [-1,  2, -1]
        ], dtype=np.float32) / 4.0
        
        # Stack into (3, 1, 3, 3) format for PyTorch Conv2d weights
        weights = np.stack([filter1, filter2, filter3], axis=0)
        weights = np.expand_dims(weights, axis=1) # (3, 1, 3, 3)
        
        # Since input is normally (B, 3, H, W) and output is (B, 3, H, W),
        # we will use grouped convolution (groups=in_channels) to apply these 
        # 3 filters independently to EACH of the 3 input color channels, and then 
        # average or process them. But a standard approach is to convert input to grayscale 
        # first, or apply the 3 filters to each channel resulting in 9 channels, 
        # or just initialize a (3, 3, 3, 3) tensor by repeating the weights across input channels.
        
        # Repeating weights for all 3 input channels: (out_channels=3, in_channels=3, H=3, W=3)
        # We divide by 3 to average the color channels implicitly during convolution.
        weights = np.repeat(weights, in_channels, axis=1) / float(in_channels)
        
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1, bias=False)
        
        # Assign fixed weights and freeze them
        self.conv.weight = nn.Parameter(torch.from_numpy(weights))
        self.conv.weight.requires_grad = False
        
        logger.debug("SRMConv2d initialised with frozen 3x3 high-pass kernels.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass extracts the noise residual map.
        x: (B, C, H, W)
        """
        return self.conv(x)


class NoiseResidualBranch(nn.Module):
    """
    Branch D: Noise Residual Feature Extractor.
    
    1. Extracts invisible noise patterns using fixed SRM filters.
    2. Passes the noise map through a trainable ResNet backbone to detect 
       sensor inconsistencies and deepfake blending boundaries.
    """

    def __init__(
        self, 
        proj_dim: int = 512,
        pretrained: bool = True
    ) -> None:
        super().__init__()
        
        self.proj_dim = proj_dim
        
        # Fixed noise extraction layer
        self.srm_layer = SRMConv2d(in_channels=3)
        
        logger.info(f"Loading ResNet18 backbone for Noise Branch | pretrained={pretrained}")
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        
        # Replace the final FC layer with our projection head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, proj_dim),
            nn.GELU()
        )
        
        logger.info(f"NoiseResidualBranch initialised | out_features={proj_dim}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            pixel_values: Tensor of shape (Batch, 3, H, W).
                          
        Returns:
            Projected noise feature vector of shape (Batch, proj_dim).
        """
        # 1. Suppress image content, expose noise
        noise_map = self.srm_layer(pixel_values)
        
        # 2. Learn anomalies from the noise map
        projected = self.backbone(noise_map)
        
        return projected
