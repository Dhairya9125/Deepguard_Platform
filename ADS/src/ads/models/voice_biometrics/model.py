"""Voice Biometrics Engine using ECAPA-TDNN.

Detects speaker inconsistency, identity drift, and cloned voice artifacts.
Performs speaker embedding matching and scoring.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ads.config.settings import settings

logger = logging.getLogger(__name__)


class ECAPABlock(nn.Module):
    """ECAPA-TDNN building block with squeeze-excitation."""

    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=kernel_size // 2 * dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 1)
        self.bn2 = nn.BatchNorm1d(channels)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv1d(channels // 8, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        se = self.se(x)
        return residual + x * se


class VoiceBiometricsBranch(nn.Module):
    """Voice biometrics engine for speaker consistency analysis.

    Uses ECAPA-TDNN architecture for speaker embedding extraction
    and comparison. Detects:
    - Speaker identity drift
    - Cloned voice inconsistencies
    - Emotional/accent inconsistency
    - Multi-speaker anomalies
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__()
        self.config = config or {}
        self.embedding_dim = self.config.get("embedding_dim", settings.voice_biometrics.embedding_dim)
        self.threshold = self.config.get("threshold", settings.voice_biometrics.threshold)
        self.input_dim = 80  # FBank features

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.ecapa_blocks = nn.ModuleList([
            ECAPABlock(128, kernel_size=3, dilation=1),
            ECAPABlock(128, kernel_size=5, dilation=2),
            ECAPABlock(128, kernel_size=7, dilation=3),
        ])

        self.stat_pool = nn.AdaptiveAvgPool1d(1)

        self.embedding_layer = nn.Sequential(
            nn.Linear(128, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
        )

        self.classifier = nn.Linear(self.embedding_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        return_embedding: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through voice biometrics branch.

        Args:
            features: (batch, time, freq) acoustic features
            return_embedding: Whether to return embeddings

        Returns:
            Dict with logits, embedding, speaker_consistency_score
        """
        if features.dim() == 2:
            features = features.unsqueeze(1)
        elif features.dim() == 3 and features.shape[1] != 1:
            features = features.unsqueeze(1)

        x = self.feature_extractor(features)

        for block in self.ecapa_blocks:
            x = block(x)

        pooled = self.stat_pool(x).squeeze(-1)
        embedding = self.embedding_layer(pooled)
        embedding = F.normalize(embedding, p=2, dim=1)

        logits = self.classifier(embedding)

        return {
            "logits": logits,
            "embedding": embedding if return_embedding else embedding.detach(),
            "confidence": torch.sigmoid(logits),
            "speaker_consistency": 1.0 - torch.sigmoid(logits),
        }

    def compute_speaker_similarity(
        self, emb1: torch.Tensor, emb2: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine similarity between two speaker embeddings."""
        emb1 = F.normalize(emb1, p=2, dim=-1)
        emb2 = F.normalize(emb2, p=2, dim=-1)
        return torch.mm(emb1, emb2.T)

    def detect_speaker_drift(
        self, embeddings: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """Detect speaker identity drift across segments.

        Args:
            embeddings: List of speaker embeddings per segment

        Returns:
            Dict with drift_score, segment_scores, is_consistent
        """
        if len(embeddings) < 2:
            return {
                "drift_score": 0.0,
                "segment_scores": [1.0],
                "is_consistent": True,
            }

        stacked = torch.stack(embeddings)
        ref = stacked.mean(dim=0, keepdim=True)

        similarities = []
        for emb in embeddings:
            sim = self.compute_speaker_similarity(ref, emb.unsqueeze(0)).item()
            similarities.append(sim)

        drift_score = 1.0 - torch.tensor(similarities).mean().item()
        is_consistent = drift_score < self.threshold

        return {
            "drift_score": drift_score,
            "segment_scores": similarities,
            "is_consistent": is_consistent,
        }
