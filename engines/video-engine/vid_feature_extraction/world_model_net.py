"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.world_model_net
Layer   : Layer 2 — Temporal Feature Extraction
Branch  : Branch G — Scene Semantic Consistency

WorldModelNet 3D Convolutional Neural Network for full-frame scene semantic consistency classification.
"""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class WorldModelNet(nn.Module):
    """
    WorldModelNet 3D ConvNet architecture.
    Takes sequences of shape (B, 3, T, H, W) and outputs a scene realism rating.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(32, 1)
        
        logger.info("WorldModelNet architecture successfully constructed.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Full-frame video tensor of shape (B, 3, T, H, W) or (B, T, 3, H, W).
               If shape is (B, T, 3, H, W), it is transposed to (B, 3, T, H, W).

        Returns:
            torch.Tensor: Realistic probability of shape (B,).
        """
        if x.ndim == 5 and x.shape[2] == 3:
            # (B, T, 3, H, W) -> transpose to (B, 3, T, H, W)
            x = x.transpose(1, 2)
            
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        x = F.relu(self.bn2(self.conv2(x)))
        
        x = self.pool(x).squeeze(-1).squeeze(-1).squeeze(-1)  # (B, 32)
        logits = self.fc(x)  # (B, 1)
        return torch.sigmoid(logits).squeeze(1)


class WorldModelActiveClassifier(nn.Module):
    """
    Active model wrapper for WorldModelNet. Handles checkpoints, input formatting, 
    and predicts classification probability.
    """
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu"
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.model = WorldModelNet()
        self.has_weights = False
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
            
        self.to(self.device)

    def load_checkpoint(self, path: str | Path) -> None:
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.has_weights = True
            logger.info("Loaded WorldModelNet weights from %s", path)
        except Exception as e:
            logger.error("Failed to load WorldModelNet weights: %s", e)

    def forward(self, full_seq: torch.Tensor) -> torch.Tensor:
        """Rate sequence consistency score of shape (B,)."""
        full_seq = full_seq.to(self.device)
        return self.model(full_seq)

    def predict_probability(self, full_seq: torch.Tensor) -> torch.Tensor:
        """
        Evaluate full frame sequences to output a fake probability score in [0, 1].
        """
        if self.has_weights:
            realism = self.forward(full_seq)
            # Map realism: high realism -> low fake probability
            return 1.0 - realism

        # Fallback evaluation using global environment instability
        probs = []
        for i in range(full_seq.shape[0]):
            seq = full_seq[i].detach().cpu().numpy()  # (C, T, H, W) or (T, C, H, W)
            if seq.ndim == 4 and seq.shape[0] == 3:
                # Transpose to (T, H, W, C)
                seq = np.transpose(seq, (1, 2, 3, 0))
            elif seq.ndim == 4 and seq.shape[1] == 3:
                # Transpose to (T, H, W, C)
                seq = np.transpose(seq, (0, 2, 3, 1))

            frames_list = [f for f in seq]
            
            from .semantic_signals import compute_environment_instability
            instability = compute_environment_instability(frames_list)
            
            if instability > 10.0:
                prob = min(0.95, 0.40 + 0.05 * instability)
            else:
                prob = max(0.05, 0.10 + 0.02 * instability)
            probs.append(prob)
            
        return torch.tensor(probs, dtype=torch.float32, device=self.device)
