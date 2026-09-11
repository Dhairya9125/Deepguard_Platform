"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.syncnet_model
Layer   : Layer 2 — Temporal Feature Extraction
Branch  : Branch D — Cross-Modal Synchronization

SyncNet audio-visual lip-sync model implemented natively in PyTorch.
"""

from __future__ import annotations

import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SyncNetModel(nn.Module):
    """
    SyncNet model architecture.
    
    Contains two parallel convolutional pathways:
      1. Audio Net: Processes Mel-spectrogram inputs of shape (B, 1, 13, 20).
      2. Visual Net: Processes 3D mouth crop sequences of shape (B, 3, 5, 112, 112).
      
    Outputs a cosine distance similarity score representing visual-audio sync.
    """
    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # ── 1. Audio Pathway (Conv2D over Mel-Spectrogram) ──────────────────
        self.audio_net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.audio_fc = nn.Linear(128, embed_dim)

        # ── 2. Visual Pathway (Conv3D over Mouth Crops) ─────────────────────
        self.visual_net = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.visual_fc = nn.Linear(128, embed_dim)
        
        logger.info("SyncNetModel architecture successfully constructed.")

    def forward(
        self, 
        audio_spec: torch.Tensor, 
        mouth_crops: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_spec: (B, 1, 13, 20) Mel-spectrogram chunk.
            mouth_crops: (B, 3, 5, 112, 112) mouth crops timeline.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (audio_embed, visual_embed) of shape (B, embed_dim).
        """
        # Audio feature extraction
        a_feat = self.audio_net(audio_spec)
        a_feat = torch.flatten(a_feat, 1)
        a_emb = self.audio_fc(a_feat)
        a_emb = F.normalize(a_emb, p=2, dim=1)

        # Visual feature extraction
        v_feat = self.visual_net(mouth_crops)
        v_feat = torch.flatten(v_feat, 1)
        v_emb = self.visual_fc(v_feat)
        v_emb = F.normalize(v_emb, p=2, dim=1)

        return a_emb, v_emb

    def compute_distance(
        self, 
        audio_spec: torch.Tensor, 
        mouth_crops: torch.Tensor
    ) -> torch.Tensor:
        """Compute the Euclidean distance between normalized audio & visual embeddings."""
        a_emb, v_emb = self.forward(audio_spec, mouth_crops)
        # For normalized vectors, Euclidean distance is sqrt(2 - 2*cosine_similarity)
        dist = torch.sqrt(torch.sum((a_emb - v_emb) ** 2, dim=1) + 1e-9)
        return dist


class SyncNetMotionModel(nn.Module):
    """
    Wrapper for SyncNetModel that handles device placement, weight loading, and prediction scores.
    """
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu"
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.model = SyncNetModel(embed_dim=128)

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        self.to(self.device)

    def load_checkpoint(self, path: str | Path) -> None:
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info("Loaded SyncNet checkpoint weights from %s", path)
        except Exception as e:
            logger.error("Failed to load SyncNet checkpoint: %s", e)

    def forward(
        self, 
        audio_spec: torch.Tensor, 
        mouth_crops: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate cosine distance between audio and mouth crops.
        """
        audio_spec = audio_spec.to(self.device)
        mouth_crops = mouth_crops.to(self.device)
        return self.model.compute_distance(audio_spec, mouth_crops)

    def predict_sync_confidence(
        self, 
        audio_spec: torch.Tensor, 
        mouth_crops: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict synchronization confidence score.
        Lower distance = higher sync confidence.
        
        Returns:
            Tensor of shape (B,) with values mapped to a confidence scale [0, 10].
            Typically, distance < 1.0 indicates strong sync (confidence 6 to 10).
        """
        dist = self.forward(audio_spec, mouth_crops)
        # Map distance: dist=0 -> conf=10, dist=2.0 -> conf=0
        confidence = 10.0 * (1.0 - (dist / 2.0))
        return torch.clamp(confidence, 0.0, 10.0)
