"""Cross-Branch Fusion Engine.

Combines outputs from all model branches using a Transformer-based
fusion mechanism with adaptive gating. Produces final detection
decision with calibrated confidence.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ads.config.settings import settings

logger = logging.getLogger(__name__)


class FusionTransformer(nn.Module):
    """Cross-branch fusion transformer.

    Fuses embeddings from all branches using cross-attention,
    adaptive gating, and produces the final deepfake probability.

    Architecture:
    - Branch embedding projection
    - Cross-attention fusion
    - Adaptive gating network
    - Final classification head
    - Uncertainty estimation
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__()
        self.config = config or {}
        self.fusion_dim = self.config.get("fusion_dim", settings.fusion.fusion_dim)
        self.num_heads = self.config.get("num_heads", settings.fusion.num_heads)
        self.num_layers = self.config.get("num_layers", settings.fusion.num_layers)
        self.dropout_rate = self.config.get("dropout", settings.fusion.dropout)
        self.use_adaptive_fusion = self.config.get("use_adaptive_fusion", settings.fusion.use_adaptive_fusion)
        self.use_gating = self.config.get("use_gating", settings.fusion.use_gating)
        self.output_dim = self.config.get("output_dim", settings.fusion.output_dim)

        self.branch_dims = {
            "wavlm": 256,
            "xlsr": 256,
            "spectral": 256,
            "voice_biometrics": 192,
            "temporal": 256,
            "diffusion": 128,
            "adversarial": 128,
        }

        total_branch_dim = sum(self.branch_dims.values())

        self.branch_projections = nn.ModuleDict({
            name: nn.Linear(dim, self.fusion_dim)
            for name, dim in self.branch_dims.items()
        })

        self.fusion_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.fusion_dim,
                nhead=self.num_heads,
                dim_feedforward=self.fusion_dim * 4,
                dropout=self.dropout_rate,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(self.num_layers)
        ])

        self.fusion_norm = nn.LayerNorm(self.fusion_dim)

        if self.use_gating:
            self.gate_network = nn.Sequential(
                nn.Linear(self.fusion_dim * 7, self.fusion_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.fusion_dim, 7),
                nn.Softmax(dim=-1),
            )

        self.fusion_pool = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.GELU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, self.output_dim),
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        branch_embeddings: Dict[str, torch.Tensor],
        branch_confidences: Optional[Dict[str, torch.Tensor]] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Fuse branch embeddings into final prediction.

        Args:
            branch_embeddings: Dict mapping branch name to embedding tensor
            branch_confidences: Optional dict of confidence per branch
            return_attention: Whether to return attention weights

        Returns:
            Dict with logits, probability, confidence, embedding,
                  branch_weights, uncertainty
        """
        projected = {}
        valid_order = []
        for name in ["wavlm", "xlsr", "spectral", "voice_biometrics", "temporal", "diffusion", "adversarial"]:
            if name in branch_embeddings:
                emb = branch_embeddings[name]
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
                projected[name] = self.branch_projections[name](emb)
                valid_order.append(name)

        if not projected:
            batch_size = 1
            dummy = torch.zeros(1, self.fusion_dim, device=next(self.parameters()).device)
            projected["wavlm"] = dummy
            valid_order = ["wavlm"]

        batch_size = projected[valid_order[0]].shape[0]

        stacked = torch.stack([projected[name] for name in valid_order], dim=1)

        x = stacked
        for layer in self.fusion_layers:
            x = layer(x)

        x = self.fusion_norm(x)

        if self.use_gating:
            flat = x.reshape(batch_size, -1)
            gate_weights = self.gate_network(flat)
            weighted = (x * gate_weights.unsqueeze(-1)).sum(dim=1)
        else:
            weighted = x.mean(dim=1)

        fused = self.fusion_pool(weighted)
        logits = self.classifier(fused)
        uncertainty = self.uncertainty_head(fused)

        result = {
            "logits": logits,
            "probability": torch.sigmoid(logits),
            "confidence": 1.0 - torch.sigmoid(uncertainty),
            "embedding": fused,
            "uncertainty": uncertainty,
        }

        if return_attention:
            result["branch_weights"] = F.softmax(
                torch.stack([projected[name].norm(dim=-1).mean() for name in valid_order]),
                dim=-1,
            )

        return result
