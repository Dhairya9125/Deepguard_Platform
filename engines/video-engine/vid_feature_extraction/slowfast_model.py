"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.slowfast_model
Layer   : Layer 2 — Temporal Feature Extraction
Branch  : Branch C — Temporal Motion Modeling

SlowFast 3D CNN model implemented natively in PyTorch.
"""

from __future__ import annotations

import logging
from pathlib import Path
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SlowFastModel(nn.Module):
    """
    SlowFast 3D CNN implementation.
    
    Decomposes the input video clip into:
      - Slow Pathway: Low frame rate, high spatial capacity (channels).
      - Fast Pathway: High frame rate, low spatial capacity.
    Lateral connections fuse fast features into the slow pathway.
    """
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        
        # 1. Slow Pathway (low temporal resolution, high channels)
        # Input shape: (B, C, T_slow, H, W)
        self.slow_conv = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        # 2. Fast Pathway (high temporal resolution, low channels)
        # Input shape: (B, C, T_fast, H, W)
        self.fast_conv = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        # 3. Lateral connection: Fast pathway -> Slow pathway
        # Downsamples fast features temporally by a factor of 4.
        self.lateral_conn = nn.Conv3d(
            in_channels=8,
            out_channels=16,  # 2 * fast_channels
            kernel_size=(5, 1, 1),
            stride=(4, 1, 1),
            padding=(2, 0, 0),
            bias=False
        )
        
        # 4. Stage 2 Combined
        # Input to slow stage: slow_conv (64) + lateral (16) = 80 channels
        self.slow_stage = nn.Sequential(
            nn.Conv3d(80, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        
        self.fast_stage = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        
        # Pooling & classification
        self.pool_slow = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.pool_fast = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(128 + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        
        logger.info("SlowFastModel architecture successfully constructed.")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            pixel_values: Tensor of shape (B, T, C, H, W)
                          Where T is sequence length, C=3, H=224, W=224.
                          
        Returns:
            Tensor of shape (B, num_classes)
        """
        # Permute (B, T, C, H, W) -> (B, C, T, H, W) for Conv3D
        x = pixel_values.permute(0, 2, 1, 3, 4)
        
        # Slow pathway sub-sampling: take every 4th frame (1/4 FPS)
        x_slow = x[:, :, ::4, :, :]
        x_fast = x
        
        # Stage 1 forward passes
        feat_slow_init = self.slow_conv(x_slow)
        feat_fast_init = self.fast_conv(x_fast)
        
        # Lateral connection projection
        lateral = self.lateral_conn(feat_fast_init)
        
        # Match temporal dimensions in case of rounding differences
        t_slow = feat_slow_init.shape[2]
        t_lat = lateral.shape[2]
        if t_lat != t_slow:
            if t_lat < t_slow:
                # Pad
                diff = t_slow - t_lat
                lateral = torch.cat([lateral, lateral[:, :, -1:].repeat(1, 1, diff, 1, 1)], dim=2)
            else:
                # Slice
                lateral = lateral[:, :, :t_slow, :, :]
                
        # Fuse fast features into slow pathway
        feat_slow = torch.cat([feat_slow_init, lateral], dim=1)
        
        # Stage 2 forward passes
        out_slow = self.slow_stage(feat_slow)
        out_fast = self.fast_stage(feat_fast_init)
        
        # Global Average Pooling
        pooled_slow = self.pool_slow(out_slow).squeeze(-1).squeeze(-1).squeeze(-1)
        pooled_fast = self.pool_fast(out_fast).squeeze(-1).squeeze(-1).squeeze(-1)
        
        # Concat features and classify
        combined = torch.cat([pooled_slow, pooled_fast], dim=1)
        logits = self.fc(combined)
        return logits


class SlowFastMotionModel(nn.Module):
    """
    Orchestration wrapper for the SlowFastModel.
    Handles device placement and weight loading.
    """
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu"
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.model = SlowFastModel(num_classes=2)
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
            
        self.to(self.device)

    def load_checkpoint(self, path: str | Path) -> None:
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info("Loaded SlowFast checkpoint: %s", path)
        except Exception as e:
            logger.error("Failed to load SlowFast checkpoint from %s: %s", path, e)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Input: (B, T, C, H, W).
        Returns: logits (B, 2)
        """
        pixel_values = pixel_values.to(self.device)
        return self.model(pixel_values)

    def predict_probability(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Predict fake probability in range [0, 1].
        """
        logits = self.forward(pixel_values)
        probs = torch.softmax(logits, dim=-1)
        return probs[:, 1]
