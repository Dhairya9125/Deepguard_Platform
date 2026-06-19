"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.fusion-engine.models.fusion_transformer
Layer   : Layer 3 — Cross-Modal Fusion Engine

Natively implements the PyTorch multimodal cross-attention transformer.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MultimodalFusionTransformer(nn.Module):
    """
    Cross-Attention Multimodal Transformer for Layer 3 fusion.
    
    Inputs:
        - Audio embeddings      : (B, Ta, Da)
        - Visual embeddings     : (B, Tv, Dv)
        - Temporal embeddings   : (B, Tt, Dt)
        - Biological embeddings : (B, Tb, Db)
        
    Outputs:
        - Real/Fake classification logits : (B, 2)
        - Forgery localization timeline   : (B, Tv) frame-level probabilities
        - Modality attribution ratings    : (B, 4) probability of fake per modality
    """
    def __init__(
        self,
        audio_dim: int = 128,
        visual_dim: int = 2560,  # Matches full Branch A artifact_embedding size
        temporal_dim: int = 128,
        biological_dim: int = 128,
        fused_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
    ) -> None:
        super().__init__()
        self.fused_dim = fused_dim
        self.max_seq_len = max_seq_len

        # ── 1. Unified Projection Layers ─────────────────────────────────────
        self.proj_audio = nn.Linear(audio_dim, fused_dim)
        self.proj_visual = nn.Linear(visual_dim, fused_dim)
        self.proj_temporal = nn.Linear(temporal_dim, fused_dim)
        self.proj_biological = nn.Linear(biological_dim, fused_dim)

        # ── 2. Modality and Positional Embeddings ────────────────────────────
        # Modality embedding: 0=audio, 1=visual, 2=temporal, 3=biological
        self.modality_emb = nn.Embedding(4, fused_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, fused_dim))
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, fused_dim))

        # Initialize embeddings with small random values
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── 3. Transformer Encoder Blocks ────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fused_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ── 4. Hierarchical Fusion & Classifier Heads ────────────────────────
        # A. High-level classification (REAL vs FAKE)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

        # B. Temporal localization (forgery probability per frame/token)
        self.localizer = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # C. Modality attribution (binary fake probability per modality: audio, visual, temporal, biological)
        self.attribution = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

        logger.info("MultimodalFusionTransformer constructed successfully with fused_dim=%d", fused_dim)

    def forward(
        self,
        audio_feat: Optional[torch.Tensor] = None,
        visual_feat: Optional[torch.Tensor] = None,
        temporal_feat: Optional[torch.Tensor] = None,
        biological_feat: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
        biological_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            audio_feat: (B, Ta, Da)
            visual_feat: (B, Tv, Dv)
            temporal_feat: (B, Tt, Dt)
            biological_feat: (B, Tb, Db)
            audio_mask: (B, Ta) bool padding mask (True = padded/masked)
            visual_mask: (B, Tv) bool padding mask (True = padded/masked)
            temporal_mask: (B, Tt) bool padding mask (True = padded/masked)
            biological_mask: (B, Tb) bool padding mask (True = padded/masked)

        Returns:
            Tuple:
              - logits: (B, 2)
              - attribution: (B, 4)
              - localization: (B, Tv) if visual is present, else None
        """
        # Determine device and batch size
        device = self.cls_token.device
        B = None
        for feat in [audio_feat, visual_feat, temporal_feat, biological_feat]:
            if feat is not None:
                B = feat.shape[0]
                break
        
        if B is None:
            raise ValueError("MultimodalFusionTransformer requires at least one modality input.")

        # 1. Build initial token sequence with CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, fused_dim)
        tokens_list = [cls_tokens]
        
        # CLS is never masked (False = valid)
        cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)
        mask_list = [cls_mask]

        # Define configurations for each modality: (feature, projection, modality_idx, padding_mask)
        modalities = [
            (audio_feat, self.proj_audio, 0, audio_mask),
            (visual_feat, self.proj_visual, 1, visual_mask),
            (temporal_feat, self.proj_temporal, 2, temporal_mask),
            (biological_feat, self.proj_biological, 3, biological_mask),
        ]

        # Slices dictionary to retrieve specific outputs post-transformer
        modality_slices = {}
        current_idx = 1  # Index 0 is CLS token

        for feat, proj_layer, mod_idx, mask in modalities:
            if feat is not None and feat.shape[1] > 0:
                T_m = feat.shape[1]
                if T_m > self.max_seq_len:
                    logger.warning("Feature sequence length %d exceeds max_seq_len %d, truncating.", T_m, self.max_seq_len)
                    feat = feat[:, :self.max_seq_len, :]
                    T_m = self.max_seq_len
                    if mask is not None:
                        mask = mask[:, :self.max_seq_len]

                # Project to fused_dim
                proj_feat = proj_layer(feat)  # (B, T_m, fused_dim)

                # Add modality embedding
                mod_emb = self.modality_emb(torch.tensor([mod_idx], device=device)).unsqueeze(0)  # (1, 1, fused_dim)
                proj_feat = proj_feat + mod_emb

                # Add positional embedding
                pos_emb = self.pos_embedding[:, :T_m, :]
                proj_feat = proj_feat + pos_emb

                tokens_list.append(proj_feat)

                # Add padding mask
                if mask is not None:
                    mask_list.append(mask)
                else:
                    mask_list.append(torch.zeros((B, T_m), dtype=torch.bool, device=device))

                modality_slices[mod_idx] = (current_idx, current_idx + T_m)
                current_idx += T_m

        # Concatenate all tokens and masks
        fused_tokens = torch.cat(tokens_list, dim=1)  # (B, S, fused_dim)
        key_padding_mask = torch.cat(mask_list, dim=1)  # (B, S)

        # 2. Run cross-attention transformer encoder
        # nn.TransformerEncoder expects src_key_padding_mask
        fused_output = self.transformer_encoder(fused_tokens, src_key_padding_mask=key_padding_mask)  # (B, S, fused_dim)

        # 3. Extract CLS representation for global decisions
        cls_rep = fused_output[:, 0, :]  # (B, fused_dim)
        logits = self.classifier(cls_rep)  # (B, 2)
        attribution = self.attribution(cls_rep)  # (B, 4)

        # 4. Extract visual sequence tokens for localized predictions
        localization = None
        if 1 in modality_slices:
            v_start, v_end = modality_slices[1]
            visual_rep = fused_output[:, v_start:v_end, :]  # (B, Tv, fused_dim)
            localization = self.localizer(visual_rep).squeeze(-1)  # (B, Tv)

        return logits, attribution, localization
