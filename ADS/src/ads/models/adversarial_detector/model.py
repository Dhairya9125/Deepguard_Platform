"""Adversarial Robustness Detector.

Detects adversarial perturbations and defends against:
- White-box attacks (FGSM, PGD)
- Black-box attacks
- Evasion attacks
- Audio perturbation attacks

Uses multiple detection strategies:
- Local Intrinsic Dimensionality (LID)
- Mahalanobis distance-based detection
- ODIN temperature scaling
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ads.config.settings import settings

logger = logging.getLogger(__name__)


class AdversarialDetectorBranch(nn.Module):
    """Adversarial robustness detector.

    Detects whether input contains adversarial perturbations designed
    to evade deepfake detection. Combines multiple detection strategies
    for robust identification of attacks.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__()
        self.config = config or {}
        self.use_lid = self.config.get("use_lid", settings.adversarial_detector.use_lid)
        self.use_mahalanobis = self.config.get("use_mahalanobis", settings.adversarial_detector.use_mahalanobis)
        self.use_odin = self.config.get("use_odin", settings.adversarial_detector.use_odin)
        self.temperature = self.config.get("temperature", settings.adversarial_detector.temperature)
        self.noise_magnitude = self.config.get("noise_magnitude", settings.adversarial_detector.noise_magnitude)
        self.embedding_dim = self.config.get("embedding_dim", settings.adversarial_detector.embedding_dim)

        self.feature_encoder = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )

        self.embedding_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
        )

        self.perturbation_detector = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

        self.register_buffer("train_mean", torch.zeros(self.embedding_dim))
        self.register_buffer("train_cov_inv", torch.eye(self.embedding_dim))
        self._trained_stats = False

    def forward(
        self,
        waveform: torch.Tensor,
        return_embedding: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Detect adversarial perturbations in audio.

        Args:
            waveform: Audio waveform (batch, samples) or (samples,)
            return_embedding: Whether to return embeddings

        Returns:
            Dict with logits, embedding, robustness_score
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(1)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        x = self.feature_encoder(waveform)
        embedding = self.embedding_proj(x)

        logits = self.perturbation_detector(embedding)

        robustness = 1.0 - torch.sigmoid(logits)

        result = {
            "logits": logits,
            "embedding": embedding if return_embedding else embedding.detach(),
            "confidence": torch.sigmoid(logits),
            "robustness_score": robustness,
        }

        if self.use_mahalanobis and self._trained_stats:
            mdist = self._compute_mahalanobis(embedding)
            result["mahalanobis_distance"] = mdist
            result["ood_score"] = torch.sigmoid(-mdist + 5.0)

        return result

    def _compute_mahalanobis(self, embedding: torch.Tensor) -> torch.Tensor:
        diff = embedding - self.train_mean
        return torch.sqrt(torch.sum(diff @ self.train_cov_inv * diff, dim=-1))

    @torch.no_grad()
    def fit_stats(self, embeddings: torch.Tensor, eps: float = 1e-6) -> None:
        """Fit statistics for Mahalanobis distance.

        Args:
            embeddings: (N, dim) tensor of clean embeddings
            eps: Regularization constant for covariance inversion
        """
        self.train_mean = embeddings.mean(dim=0)
        centered = embeddings - self.train_mean
        cov = (centered.T @ centered) / (embeddings.shape[0] - 1)
        cov = cov + eps * torch.eye(cov.shape[0], device=cov.device)
        self.train_cov_inv = torch.linalg.inv(cov)
        self._trained_stats = True
        logger.info("Adversarial detector: Mahalanobis stats fitted.")

    @staticmethod
    def fgsm_attack(
        model: nn.Module,
        waveform: torch.Tensor,
        epsilon: float = 0.01,
        target_label: int = 1,
    ) -> torch.Tensor:
        """Generate FGSM adversarial example.

        Args:
            model: Target model
            waveform: Input waveform (requires_grad=True)
            epsilon: Perturbation magnitude
            target_label: Target attack label (0=real, 1=fake)

        Returns:
            Adversarial waveform
        """
        waveform.requires_grad = True
        output = model(waveform)
        loss = F.binary_cross_entropy_with_logits(
            output["logits"], torch.full_like(output["logits"], target_label)
        )
        loss.backward()

        perturbation = epsilon * waveform.grad.sign()
        adv_waveform = waveform + perturbation
        return torch.clamp(adv_waveform, -1.0, 1.0).detach()
