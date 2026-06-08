"""Spectral CNN branch for detecting synthesis artifacts.

Detects GAN, vocoder, and diffusion artifacts in spectral representations.
Uses efficient CNN backbone for spectral anomaly detection.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ads.config.settings import settings

logger = logging.getLogger(__name__)


class SpectralConvBlock(nn.Module):
    """Convolutional block with batch norm and residual connection."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.GELU()

        self.shortcut = nn.Identity()
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)) + self.shortcut(x))


class SpectralCNN(nn.Module):
    """Spectral CNN branch for detecting synthesis artifacts.

    Processes stacked spectral features (MFCC, LFCC, CQCC, Chroma)
    through a convolutional backbone to detect anomalies indicative
    of synthetic speech generation.

    Architecture:
    - Multi-channel spectral input
    - EfficientNet-style conv blocks
    - Global pooling
    - Classification head
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__()
        self.config = config or {}
        self.input_channels = self.config.get("input_channels", settings.spectral_cnn.input_channels)
        self.dropout_rate = self.config.get("dropout", settings.spectral_cnn.dropout)
        self.embedding_dim = self.config.get("embedding_dim", settings.spectral_cnn.embedding_dim)

        self.stem = nn.Sequential(
            SpectralConvBlock(self.input_channels, 32, kernel_size=3, stride=2),
            SpectralConvBlock(32, 32, kernel_size=3),
        )

        self.stage1 = nn.Sequential(
            SpectralConvBlock(32, 64, kernel_size=3, stride=2),
            SpectralConvBlock(64, 64),
            SpectralConvBlock(64, 64),
        )

        self.stage2 = nn.Sequential(
            SpectralConvBlock(64, 128, kernel_size=3, stride=2),
            SpectralConvBlock(128, 128),
            SpectralConvBlock(128, 128),
        )

        self.stage3 = nn.Sequential(
            SpectralConvBlock(128, 256, kernel_size=3, stride=2),
            SpectralConvBlock(256, 256),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        spectral_features: torch.Tensor,
        return_embedding: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through spectral CNN.

        Args:
            spectral_features: (batch, channels, freq, time) tensor
            return_embedding: Whether to return embeddings

        Returns:
            Dict with logits, embedding, confidence
        """
        if spectral_features.dim() == 2:
            spectral_features = spectral_features.unsqueeze(0).unsqueeze(0)
        elif spectral_features.dim() == 3:
            spectral_features = spectral_features.unsqueeze(0)

        if spectral_features.shape[1] != self.input_channels:
            spectral_features = spectral_features.repeat(1, self.input_channels, 1, 1)
            spectral_features = spectral_features[:, : self.input_channels, :, :]

        x = self.stem(spectral_features)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x)

        embedding = self.embedding(x)
        logits = self.classifier(embedding)

        return {
            "logits": logits,
            "embedding": embedding if return_embedding else embedding.detach(),
            "confidence": torch.sigmoid(logits),
            "features": x.detach() if not return_embedding else x,
        }
