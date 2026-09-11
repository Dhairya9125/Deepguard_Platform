"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.av_hubert_model
Layer   : Layer 2 — Temporal Feature Extraction
Branch  : Branch D — Cross-Modal Synchronization

AV-HuBERT multimodal audio-visual synchronization model implemented natively in PyTorch.
"""

from __future__ import annotations

import logging
from pathlib import Path
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AVHuBERTModel(nn.Module):
    """
    Multimodal Audio-Visual fusion transformer model representing AV-HuBERT behavior.
    
    Processes:
      - Visual mouth crops: (B, T, 3, 112, 112)
      - Audio waveform signal: (B, A) where A is the number of audio samples
      
    Features are extracted separately via:
      - 3D CNN Visual Front-end
      - 1D CNN Audio Front-end
    Fused via Cross-Attention layers and projected to binary classification logits (B, 2).
    """
    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.d_model = d_model

        # ── 1. Visual Front-end (3D Conv to project T×3×112×112 -> T×d_model) ──
        self.visual_frontend = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 1, 1))  # (B, 64, T, 1, 1)
        )
        self.visual_proj = nn.Linear(64, d_model)

        # ── 2. Audio Front-end (1D Conv to project A samples -> T_audio×d_model) ──
        self.audio_frontend = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=10, stride=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=10, stride=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)  # Fixed temporal dimension for cross attention
        )
        self.audio_proj = nn.Linear(64, d_model)

        # ── 3. Cross-Modal Fusion via Cross-Attention ─────────────────────────
        # Queries: Visual features (T, B, d_model)
        # Keys/Values: Audio features (T_audio, B, d_model)
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)
        self.norm1 = nn.LayerNorm(d_model)
        
        # ── 4. Classification Head ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )
        
        logger.info("AVHuBERTModel architecture successfully constructed.")

    def forward(self, mouth_crops: torch.Tensor, audio_waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mouth_crops: (B, T, 3, 112, 112)
            audio_waveform: (B, A)

        Returns:
            Tensor of shape (B, 2) containing real/fake logits.
        """
        B, T, C, H, W = mouth_crops.shape

        # 1. Visual feature extraction: permute to (B, C, T, H, W)
        v_feat = self.visual_frontend(mouth_crops.permute(0, 2, 1, 3, 4))  # (B, 64, T, 1, 1)
        v_feat = v_feat.squeeze(-1).squeeze(-1).permute(2, 0, 1)  # (T, B, 64)
        v_emb = self.visual_proj(v_feat)  # (T, B, d_model)

        # 2. Audio feature extraction: add channel dim (B, 1, A)
        if audio_waveform.ndim == 2:
            audio_waveform = audio_waveform.unsqueeze(1)
        a_feat = self.audio_frontend(audio_waveform)  # (B, 64, T_audio)
        a_feat = a_feat.permute(2, 0, 1)  # (T_audio, B, 64)
        a_emb = self.audio_proj(a_feat)  # (T_audio, B, d_model)

        # 3. Multimodal Cross-Attention fusion
        # Visual queries audio features
        attn_out, _ = self.cross_attention(query=v_emb, key=a_emb, value=a_emb)
        fused = self.norm1(v_emb + attn_out)  # Residual connection

        # 4. Pooling over temporal dimension (mean pooling over T)
        pooled = torch.mean(fused, dim=0)  # (B, d_model)

        # 5. Classifier logits
        logits = self.classifier(pooled)
        return logits


class AVHuBERTActiveModel(nn.Module):
    """
    Wrapper for AVHuBERTModel managing device placement, weight loading, and probabilities.
    """
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu"
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.model = AVHuBERTModel()

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        self.to(self.device)

    def load_checkpoint(self, path: str | Path) -> None:
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info("Loaded AV-HuBERT checkpoint weights from %s", path)
        except Exception as e:
            logger.error("Failed to load AV-HuBERT checkpoint: %s", e)

    def forward(self, mouth_crops: torch.Tensor, audio_waveform: torch.Tensor) -> torch.Tensor:
        mouth_crops = mouth_crops.to(self.device)
        audio_waveform = audio_waveform.to(self.device)
        return self.model(mouth_crops, audio_waveform)

    def predict_probability(self, mouth_crops: torch.Tensor, audio_waveform: torch.Tensor) -> torch.Tensor:
        """Predict synchronization mismatch (fake probability) in range [0, 1]."""
        logits = self.forward(mouth_crops, audio_waveform)
        probs = torch.softmax(logits, dim=-1)
        return probs[:, 1]  # Index 1 = FAKE/mismatch probability
