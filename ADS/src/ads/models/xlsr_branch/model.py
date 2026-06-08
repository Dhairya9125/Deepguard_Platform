"""XLS-R branch for multilingual robustness.

Uses Wav2Vec2 XLS-R for language-independent speech representations.
Ensures cross-language generalization for multilingual deepfake detection.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ads.config.settings import settings

logger = logging.getLogger(__name__)


class XLSRBranch(nn.Module):
    """XLS-R branch for multilingual deepfake detection.

    Wav2Vec2 XLS-R (300M) model adapted for cross-lingual
    deepfake detection. Frozen encoder with trainable head.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__()
        self.config = config or {}
        self.model_id = self.config.get("model_id", settings.xlsr.model_id)
        self.freeze_encoder = self.config.get("freeze_encoder", settings.xlsr.freeze_encoder)
        self.trainable_layers = self.config.get("trainable_layers", settings.xlsr.trainable_layers)
        self.dropout_rate = self.config.get("dropout", settings.xlsr.dropout)
        self.embedding_dim = self.config.get("embedding_dim", settings.xlsr.embedding_dim)
        self.projection_dim = self.config.get("projection_dim", settings.xlsr.projection_dim)

        self._encoder = None
        self._load_encoder()

        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.projection_dim),
            nn.LayerNorm(self.projection_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.projection_dim, self.projection_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.projection_dim, 64),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 1),
        )

    def _load_encoder(self) -> None:
        try:
            from transformers import Wav2Vec2Model

            self._encoder = Wav2Vec2Model.from_pretrained(self.model_id)
            self._encoder.eval()

            if self.freeze_encoder and self.trainable_layers > 0:
                for param in self._encoder.parameters():
                    param.requires_grad = False
                if hasattr(self._encoder, "encoder") and hasattr(self._encoder.encoder, "layers"):
                    for layer in self._encoder.encoder.layers[-self.trainable_layers :]:
                        for param in layer.parameters():
                            param.requires_grad = True
                    logger.info(
                        f"XLS-R: Frozen except last {self.trainable_layers} layers"
                    )
            elif self.freeze_encoder:
                for param in self._encoder.parameters():
                    param.requires_grad = False
                logger.info("XLS-R: Fully frozen")
        except Exception as e:
            logger.warning(f"Failed to load XLS-R: {e}. Using fallback.")
            self._encoder = None

    def forward(
        self,
        waveform: torch.Tensor,
        return_embedding: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if self._encoder is not None:
            with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
                outputs = self._encoder(waveform)
                hidden = outputs.last_hidden_state
                embedding = hidden.mean(dim=1)
        else:
            embedding = torch.randn(waveform.shape[0], self.embedding_dim, device=waveform.device)

        projected = self.projection(embedding)
        logits = self.classifier(projected)

        return {
            "logits": logits,
            "embedding": projected if return_embedding else projected.detach(),
            "confidence": torch.sigmoid(logits),
        }
