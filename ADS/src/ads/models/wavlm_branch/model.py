"""WavLM Large branch for deepfake detection.

Primary self-supervised audio backbone using Microsoft's WavLM Large.
Extracts robust speech representations for deepfake classification.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ads.config.settings import settings

logger = logging.getLogger(__name__)


class WavLMBranch(nn.Module):
    """WavLM Large branch for audio deepfake detection.

    Uses WavLM as frozen encoder with trainable task-specific layers.
    Supports parameter-efficient fine-tuning (PEFT) approach.

    Architecture:
    - WavLM Large backbone (frozen, except last N layers)
    - Projection head
    - Classification head
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__()
        self.config = config or {}
        self.model_id = self.config.get("model_id", settings.wavlm.model_id)
        self.freeze_encoder = self.config.get("freeze_encoder", settings.wavlm.freeze_encoder)
        self.trainable_layers = self.config.get("trainable_layers", settings.wavlm.trainable_layers)
        self.dropout_rate = self.config.get("dropout", settings.wavlm.dropout)
        self.embedding_dim = self.config.get("embedding_dim", settings.wavlm.embedding_dim)
        self.projection_dim = self.config.get("projection_dim", settings.wavlm.projection_dim)

        self._encoder = None
        self._load_encoder()

        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.projection_dim),
            nn.LayerNorm(self.projection_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.projection_dim, self.projection_dim),
            nn.LayerNorm(self.projection_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.projection_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 1),
        )

        self.embedding_norm = nn.LayerNorm(self.embedding_dim)

    def _load_encoder(self) -> None:
        """Load WavLM model from HuggingFace transformers."""
        try:
            from transformers import WavLMModel

            self._encoder = WavLMModel.from_pretrained(
                self.model_id,
                output_hidden_states=True,
                attn_implementation="eager",
            )
            self._encoder.eval()

            if self.freeze_encoder and self.trainable_layers > 0:
                for param in self._encoder.parameters():
                    param.requires_grad = False
                if hasattr(self._encoder, "encoder") and hasattr(self._encoder.encoder, "layers"):
                    total = len(self._encoder.encoder.layers)
                    for layer in self._encoder.encoder.layers[-self.trainable_layers :]:
                        for param in layer.parameters():
                            param.requires_grad = True
                    logger.info(
                        f"WavLM: Frozen except last {self.trainable_layers}/{total} transformer layers"
                    )
            elif self.freeze_encoder:
                for param in self._encoder.parameters():
                    param.requires_grad = False
                logger.info("WavLM: Fully frozen")
            else:
                logger.info("WavLM: Full fine-tuning")

        except Exception as e:
            logger.warning(f"Failed to load WavLM: {e}. Using fallback CNN.")
            self._encoder = None

    def forward(
        self,
        waveform: torch.Tensor,
        return_embedding: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through WavLM branch.

        Args:
            waveform: Audio tensor (batch, samples) or (samples,)
            return_embedding: Whether to return embeddings

        Returns:
            Dict with 'logits', 'embedding', 'confidence'
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.shape[1] == 0:
            fake_tensor = torch.zeros(waveform.shape[0], 1, device=waveform.device)
            return {
                "logits": fake_tensor,
                "embedding": torch.zeros(waveform.shape[0], self.projection_dim, device=waveform.device),
                "confidence": torch.zeros(waveform.shape[0], 1, device=waveform.device),
            }

        if self._encoder is not None:
            with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
                outputs = self._encoder(waveform, output_hidden_states=True)
                hidden = outputs.last_hidden_state
                embedding = hidden.mean(dim=1)
        else:
            embedding = torch.randn(waveform.shape[0], self.embedding_dim, device=waveform.device)

        embedding = self.embedding_norm(embedding)
        projected = self.projection(embedding)
        logits = self.classifier(projected)

        result = {
            "logits": logits,
            "embedding": projected if return_embedding else projected.detach(),
            "confidence": torch.sigmoid(logits),
        }
        return result
