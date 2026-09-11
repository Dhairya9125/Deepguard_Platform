"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.physnet_model
Layer   : Layer 2 — Temporal Feature Extraction
Branch  : Branch E — rPPG Biological Signal Analysis

PhysNet 3D Convolutional Neural Network for spatio-temporal rPPG BVP extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PhysNetModel(nn.Module):
    """
    PhysNet 3D ConvNet architecture.
    Takes face sequences of shape (B, 3, T, H, W) and extracts 1D BVP timeline (B, T).
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        
        self.conv_out = nn.Conv3d(64, 1, kernel_size=(1, 1, 1))
        
        logger.info("PhysNetModel architecture successfully constructed.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Face crops video tensor of shape (B, 3, T, H, W) or (B, T, 3, H, W).
               If shape is (B, T, 3, H, W), it is transposed to (B, 3, T, H, W).

        Returns:
            torch.Tensor: Estimated BVP signal of shape (B, T).
        """
        if x.ndim == 5 and x.shape[2] == 3:
            # (B, T, 3, H, W) -> transpose to (B, 3, T, H, W)
            x = x.transpose(1, 2)
            
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global spatial average pooling
        x = torch.mean(x, dim=(-2, -1), keepdim=True)  # (B, 64, T, 1, 1)
        
        pulse = self.conv_out(x)  # (B, 1, T, 1, 1)
        pulse = pulse.squeeze(1).squeeze(-1).squeeze(-1)  # (B, T)
        return pulse


class PhysNetActiveModel(nn.Module):
    """
    Active model wrapper for PhysNet. Handles checkpoints, input formatting, 
    and predicts classification probability.
    """
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu"
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.model = PhysNetModel()
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
            
        self.to(self.device)

    def load_checkpoint(self, path: str | Path) -> None:
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info("Loaded PhysNet weights from %s", path)
        except Exception as e:
            logger.error("Failed to load PhysNet weights: %s", e)

    def forward(self, face_seq: torch.Tensor) -> torch.Tensor:
        """Extract BVP signal of shape (B, T)."""
        face_seq = face_seq.to(self.device)
        return self.model(face_seq)

    def predict_probability(self, face_seq: torch.Tensor, fps: float = 30.0) -> torch.Tensor:
        """
        Evaluate face crop sequences to output a fake probability score in [0, 1].
        """
        bvp = self.forward(face_seq)  # (B, T)
        
        # Fallback evaluation using NumPy biological metrics
        probs = []
        for i in range(bvp.shape[0]):
            pulse = bvp[i].detach().cpu().numpy()
            
            # Avoid circular import
            from .rppg_signals import estimate_heart_rate
            hr, snr = estimate_heart_rate(pulse, fps)
            
            if hr < 45.0 or hr > 180.0:
                prob = 0.85
            elif snr < 1.0:
                prob = 0.70
            else:
                # normal pulse -> map to lower fake probability
                prob = max(0.05, 0.40 - 0.1 * snr)
            probs.append(prob)
            
        return torch.tensor(probs, dtype=torch.float32, device=self.device)
