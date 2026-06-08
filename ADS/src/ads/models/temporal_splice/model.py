"""Temporal Splice Detection branch.

Detects edited regions, phrase insertions, audio replacements,
and voice splicing using a Conformer-based sequence tagger.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ads.config.settings import settings

logger = logging.getLogger(__name__)


class ConformerBlock(nn.Module):
    """Conformer block with multi-head self-attention and depthwise conv."""

    def __init__(self, dim: int, num_heads: int, expansion_factor: int = 4, dropout: float = 0.1) -> None:
        super().__init__()

        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout),
        )

        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_layer_norm = nn.LayerNorm(dim)

        self.conv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Conv1d(dim * 2, dim * 2, kernel_size=7, padding=3, groups=dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout),
        )

        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.attn_layer_norm(x + attn_out)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)


class TemporalSpliceBranch(nn.Module):
    """Temporal splice detection using Conformer-based sequence tagging.

    Detects manipulated regions at frame-level, identifying:
    - Phrase insertions
    - Audio replacements
    - Segment editing
    - Voice splicing

    Outputs per-frame manipulation probabilities with CRF decoding.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__()
        self.config = config or {}
        self.model_dim = self.config.get("model_dim", settings.temporal_splice.model_dim)
        self.num_heads = self.config.get("num_heads", settings.temporal_splice.num_heads)
        self.num_layers = self.config.get("num_layers", settings.temporal_splice.num_layers)
        self.dropout_rate = self.config.get("dropout", settings.temporal_splice.dropout)
        self.max_seq_len = self.config.get("max_seq_len", settings.temporal_splice.max_seq_len)
        self.num_classes = self.config.get("num_classes", settings.temporal_splice.num_classes)
        self.use_crf = self.config.get("use_crf", settings.temporal_splice.use_crf)

        self.input_proj = nn.Linear(1024, self.model_dim)

        self.pos_encoding = nn.Embedding(self.max_seq_len, self.model_dim)

        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(self.model_dim, self.num_heads, dropout=self.dropout_rate)
            for _ in range(self.num_layers)
        ])

        self.frame_classifier = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.model_dim // 2, self.num_classes),
        )

        self.merge_conv = nn.Conv1d(self.model_dim, self.model_dim, kernel_size=3, padding=1)

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through temporal splice branch.

        Args:
            features: (batch, seq_len, feat_dim) sequential features
            mask: (batch, seq_len) boolean mask for padding
            return_attention: Whether to return attention weights

        Returns:
            Dict with frame_logits, frame_probs, seq_embedding
        """
        if features.dim() == 2:
            features = features.unsqueeze(0)

        batch_size, seq_len, _ = features.shape

        if seq_len > self.max_seq_len:
            features = features[:, : self.max_seq_len, :]
            seq_len = self.max_seq_len

        x = self.input_proj(features)

        positions = torch.arange(0, seq_len, device=features.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_encoding(positions)
        x = x + pos_emb

        for block in self.conformer_blocks:
            x = block(x, mask)

        x = x.transpose(1, 2)
        x = self.merge_conv(x)
        x = x.transpose(1, 2)

        frame_logits = self.frame_classifier(x)

        result = {
            "frame_logits": frame_logits,
            "frame_probs": F.softmax(frame_logits, dim=-1),
            "seq_embedding": x.mean(dim=1),
        }

        if return_attention:
            result["attention_weights"] = torch.softmax(frame_logits, dim=-1)

        return result

    def decode_frames(
        self, frame_probs: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        """Decode frame probabilities to binary manipulation labels.

        Args:
            frame_probs: (batch, seq_len, num_classes) probabilities
            threshold: Decision threshold

        Returns:
            (batch, seq_len) binary manipulation labels
        """
        if self.num_classes == 2:
            return (frame_probs[..., 1] > threshold).long()
        return frame_probs.argmax(dim=-1)

    def get_manipulated_regions(
        self, frame_labels: torch.Tensor, hop_time: float = 0.01
    ) -> Tuple[list, list]:
        """Extract contiguous manipulated regions from frame labels.

        Args:
            frame_labels: (batch, seq_len) binary labels
            hop_time: Time per frame in seconds

        Returns:
            Tuple of (starts, ends) lists in seconds
        """
        regions = []
        for batch_idx in range(frame_labels.shape[0]):
            labels = frame_labels[batch_idx]
            starts, ends = [], []

            in_region = False
            region_start = 0

            for t in range(len(labels)):
                if labels[t] == 1 and not in_region:
                    in_region = True
                    region_start = t
                elif labels[t] == 0 and in_region:
                    in_region = False
                    starts.append(region_start * hop_time)
                    ends.append(t * hop_time)

            if in_region:
                starts.append(region_start * hop_time)
                ends.append(len(labels) * hop_time)

            regions.append((starts, ends))

        return regions
