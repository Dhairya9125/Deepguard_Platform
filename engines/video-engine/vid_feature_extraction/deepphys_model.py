"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.deepphys_model
Layer   : Layer 2 — Temporal Feature Extraction
Branch  : Branch E — rPPG Biological Signal Analysis

DeepPhys 2D Convolutional Attention Network for appearance-motion rPPG extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DeepPhysModel(nn.Module):
    """
    DeepPhys model architecture.
    Appearance path processes standard face crop: (B, 3, H, W)
    Motion path processes difference frame: (B, 3, H, W)
    Outputs estimated BVP derivative: (B, 1)
    """
    def __init__(self) -> None:
        super().__init__()
        # Appearance pathway (attention generator)
        self.app_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.app_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.app_attn = nn.Conv2d(32, 1, kernel_size=1)

        # Motion pathway
        self.mot_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.mot_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 1)
        
        logger.info("DeepPhysModel constructed successfully.")

    def forward(self, app_frame: torch.Tensor, mot_frame: torch.Tensor) -> torch.Tensor:
        """
        Args:
            app_frame: (B, 3, H, W) current frame crops.
            mot_frame: (B, 3, H, W) difference frame crops.

        Returns:
            torch.Tensor: estimated derivative of shape (B, 1).
        """
        # Appearance path
        a1 = F.relu(self.app_conv1(app_frame))
        a2 = F.relu(self.app_conv2(a1))
        attn = torch.sigmoid(self.app_attn(a2))  # (B, 1, H, W)
        
        # Motion path
        m1 = F.relu(self.mot_conv1(mot_frame))
        m2 = F.relu(self.mot_conv2(m1))
        
        # Gating with spatial attention
        gated = m2 * attn
        
        pooled = self.pool(gated).squeeze(-1).squeeze(-1)  # (B, 32)
        out = self.fc(pooled)  # (B, 1)
        return out


class DeepPhysActiveModel(nn.Module):
    """
    Active model wrapper for DeepPhys. Processes sequences, extracts 1D BVP, 
    and predicts fake probability.
    """
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu"
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.model = DeepPhysModel()
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
            
        self.to(self.device)

    def load_checkpoint(self, path: str | Path) -> None:
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info("Loaded DeepPhys weights from %s", path)
        except Exception as e:
            logger.error("Failed to load DeepPhys weights: %s", e)

    def forward(self, face_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            face_seq: (B, T, 3, H, W) or (B, 3, T, H, W) face sequence.

        Returns:
            torch.Tensor: PPG signal of shape (B, T).
        """
        face_seq = face_seq.to(self.device)
        if face_seq.ndim == 5 and face_seq.shape[1] == 3:
            # Transpose (B, 3, T, H, W) -> (B, T, 3, H, W)
            face_seq = face_seq.transpose(1, 2)
            
        B, T, C, H, W = face_seq.shape
        if T < 2:
            return torch.zeros((B, T), device=self.device)

        pulses = []
        for b in range(B):
            seq = face_seq[b]  # (T, C, H, W)
            app = seq[:-1]     # (T-1, C, H, W)
            
            # Difference frames: (seq_{t+1} - seq_t) / (seq_{t+1} + seq_t + 1e-5)
            diff = (seq[1:] - seq[:-1]) / (seq[1:] + seq[:-1] + 1e-5)
            
            # Forward
            deriv = self.model(app, diff)  # (T-1, 1)
            deriv = deriv.squeeze(1)       # (T-1,)
            
            # Cumulative sum to restore signal, and pad to T length
            pulse = torch.cumsum(deriv, dim=0)
            pulse = torch.cat([pulse[0].unsqueeze(0), pulse], dim=0)  # (T,)
            pulses.append(pulse)
            
        return torch.stack(pulses, dim=0)

    def predict_probability(self, face_seq: torch.Tensor, fps: float = 30.0) -> torch.Tensor:
        """Evaluate BVP timeline and return fake probability in [0, 1]."""
        bvp = self.forward(face_seq)  # (B, T)
        
        probs = []
        for i in range(bvp.shape[0]):
            pulse = bvp[i].detach().cpu().numpy()
            
            from .rppg_signals import estimate_heart_rate
            hr, snr = estimate_heart_rate(pulse, fps)
            
            if hr < 45.0 or hr > 180.0:
                prob = 0.85
            elif snr < 1.0:
                prob = 0.70
            else:
                prob = max(0.05, 0.40 - 0.1 * snr)
            probs.append(prob)
            
        return torch.tensor(probs, dtype=torch.float32, device=self.device)
