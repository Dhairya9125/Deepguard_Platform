"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.videomae_model
Layer   : Layer 2 — Temporal Feature Extraction
Branch  : Branch C — Temporal Motion Modeling

HuggingFace VideoMAE wrapper with structural fallback support.
"""

from __future__ import annotations

import logging
from pathlib import Path
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VideoMAEFallback(nn.Module):
    """
    Lightweight fallback module that matches VideoMAEForVideoClassification
    input and output schemas for offline or low-dependency execution.
    """
    def __init__(self, num_labels: int = 400) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.conv = nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(16, num_labels)
        logger.info("VideoMAEFallback initialized.")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: Tensor of shape (B, T, C, H, W)
        """
        # Permute (B, T, C, H, W) -> (B, C, T, H, W) for Conv3D
        x = pixel_values.permute(0, 2, 1, 3, 4)
        x = self.conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits


class VideoMAEMotionModel(nn.Module):
    """
    VideoMAE Wrapper for VDS Temporal Motion Modeling.
    Loads 'MCG-NJU/videomae-base' by default.
    """
    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base",
        checkpoint_path: str | Path | None = None,
        device: str = "cpu"
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = torch.device(device)
        self.is_fallback = False

        logger.info("Initializing VideoMAE wrapper (device=%s) ...", device)

        try:
            from transformers import VideoMAEForVideoClassification
            # Use AutoConfig or direct load
            self.backbone = VideoMAEForVideoClassification.from_pretrained(model_name)
            logger.info("Successfully loaded pretrained VideoMAE: %s", model_name)
        except Exception as e:
            logger.warning(
                "Could not load HuggingFace VideoMAE (%s). Falling back to structural mock. Error: %s",
                model_name, e
            )
            self.backbone = VideoMAEFallback(num_labels=400)
            self.is_fallback = True

        # Projection head for deepfake classification
        self.num_backbone_labels = getattr(self.backbone.config, "num_labels", 400) if not self.is_fallback else 400
        self.projection_head = nn.Sequential(
            nn.Linear(self.num_backbone_labels, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        self.to(self.device)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load fine-tuned weights for the projection head / backbone."""
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.load_state_dict(state_dict)
            logger.info("Loaded VideoMAE checkpoint: %s", path)
        except Exception as e:
            logger.error("Failed to load VideoMAE checkpoint from %s: %s", path, e)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            pixel_values: Tensor of shape (B, T, C, H, W)
            
        Returns:
            Tensor of shape (B, 2) containing real/fake logits.
        """
        pixel_values = pixel_values.to(self.device)

        if self.is_fallback:
            backbone_logits = self.backbone(pixel_values)
        else:
            outputs = self.backbone(pixel_values=pixel_values)
            backbone_logits = outputs.logits

        logits = self.projection_head(backbone_logits)
        return logits

    def predict_probability(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Evaluate fake probability [0, 1] for each item in the batch.
        
        Returns:
            Tensor of shape (B,) containing fake probabilities.
        """
        logits = self.forward(pixel_values)
        probs = torch.softmax(logits, dim=-1)
        return probs[:, 1]  # Index 1 = FAKE probability
