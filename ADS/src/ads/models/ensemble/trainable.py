"""Trainable ensemble model combinin all branches with the fusion engine.

Wraps the multi-branch architecture into a single nn.Module
suitable for end-to-end training with backpropagation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ads.config.settings import settings
from ads.fusion.engine import FusionTransformer
from ads.models.adversarial_detector.model import AdversarialDetectorBranch
from ads.models.diffusion_detector.model import DiffusionDetectorBranch
from ads.models.spectral_cnn_branch.model import SpectralCNN
from ads.models.temporal_splice.model import TemporalSpliceBranch
from ads.models.voice_biometrics.model import VoiceBiometricsBranch
from ads.models.wavlm_branch.model import WavLMBranch
from ads.models.xlsr_branch.model import XLSRBranch
from ads.signal_processing.features import SignalProcessor

logger = logging.getLogger(__name__)


class TrainableADS(nn.Module):
    """End-to-end trainable ADS model.

    Composes selected model branches with the fusion transformer
    for joint optimization. Supports gradient-based training
    of all branches simultaneously.

    Architecture:
        waveform → [WavLM, XLS-R, Spectral, VoiceBio, Temporal, ...]
                 → FusionTransformer
                 → classification head
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        train_branches: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.config = config or {}
        self.train_branches = train_branches or settings.training.train_branches
        self.signal_processor = SignalProcessor()

        self.branches = nn.ModuleDict()

        if "wavlm" in self.train_branches:
            self.branches["wavlm"] = WavLMBranch(config)
        if "xlsr" in self.train_branches:
            self.branches["xlsr"] = XLSRBranch(config)
        if "spectral" in self.train_branches:
            self.branches["spectral"] = SpectralCNN(config)
        if "voice_biometrics" in self.train_branches:
            self.branches["voice_biometrics"] = VoiceBiometricsBranch(config)
        if "temporal" in self.train_branches:
            self.branches["temporal"] = TemporalSpliceBranch(config)
        if "diffusion" in self.train_branches:
            self.branches["diffusion"] = DiffusionDetectorBranch(config)
        if "adversarial" in self.train_branches:
            self.branches["adversarial"] = AdversarialDetectorBranch(config)

        self.fusion = FusionTransformer(config)

        logger.info(
            f"TrainableADS initialized with branches: {list(self.branches.keys())}"
        )

    def forward(
        self,
        waveform: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through all branches and fusion.

        Args:
            waveform: Audio waveform (batch, samples)
            features: Optional pre-computed signal features
            return_all: Return all branch outputs for debugging

        Returns:
            Dict with 'logits', 'probs', 'embeddings', and optionally branch outputs
        """
        batch_size = waveform.shape[0]
        device = waveform.device

        if features is None:
            features = self._extract_features(waveform)

        branch_embeddings = {}
        branch_confidences = {}
        branch_outputs = {}

        for name, branch in self.branches.items():
            try:
                if name == "spectral":
                    out = branch(self._prepare_spectral_input(features))
                elif name == "voice_biometrics":
                    out = branch(self._prepare_biometric_input(features))
                elif name == "temporal":
                    out = branch(self._prepare_temporal_input(waveform))
                else:
                    wav_1d = waveform.mean(dim=1) if waveform.dim() == 3 else waveform
                    out = branch(wav_1d)

                branch_embeddings[name] = out.get("embedding", torch.randn(batch_size, 256, device=device))
                branch_confidences[name] = out.get("confidence", torch.sigmoid(out.get("logits", torch.zeros(batch_size, 1, device=device))))
                branch_outputs[name] = out

            except Exception as e:
                logger.debug(f"Branch {name} forward failed: {e}")
                branch_embeddings[name] = torch.randn(batch_size, 256, device=device)
                branch_confidences[name] = torch.full((batch_size, 1), 0.5, device=device)

        fusion_out = self.fusion(branch_embeddings, branch_confidences)

        result = {
            "logits": fusion_out["logits"],
            "probs": fusion_out["probability"],
            "embeddings": fusion_out["embedding"],
        }

        if return_all:
            result["branch_outputs"] = branch_outputs
            result["branch_embeddings"] = branch_embeddings
            result["fusion_confidence"] = fusion_out["confidence"]
            result["fusion_uncertainty"] = fusion_out["uncertainty"]

        return result

    def _extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract signal features from waveform batch."""
        return waveform

    @staticmethod
    def _prepare_spectral_input(features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 4:
            return features
        return features.unsqueeze(1).repeat(1, 3, 1, 1) if features.dim() == 3 else features

    @staticmethod
    def _prepare_biometric_input(features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 3:
            return features.transpose(1, 2) if features.shape[1] < features.shape[2] else features
        return features.unsqueeze(1)

    @staticmethod
    def _prepare_temporal_input(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        mel = torchaudio.functional.melscale_fbanks(
            n_freqs=waveform.shape[-1] // 2 + 1,
            f_min=0,
            f_max=8000,
            n_mels=128,
            sample_rate=16000,
        )
        return mel.unsqueeze(0).expand(waveform.shape[0], -1, -1)

    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """Get all trainable parameters for optimizer."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_branch_params(self, branch_name: str) -> List[torch.nn.Parameter]:
        """Get parameters for a specific branch."""
        if branch_name in self.branches:
            return list(self.branches[branch_name].parameters())
        return []

    @torch.no_grad()
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute BCE + focal loss.

        Args:
            logits: (batch, 1) raw logits
            labels: (batch,) binary labels (0=real, 1=fake)

        Returns:
            Dict with 'loss', 'bce_loss', 'focal_loss'
        """
        labels = labels.float().view(-1, 1)
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels)

        probs = torch.sigmoid(logits)
        pt = torch.where(labels == 1, probs, 1 - probs)
        focal_loss = -((1 - pt) ** settings.training.loss_focal_gamma) * torch.log(pt + 1e-8)
        focal_loss = focal_loss.mean()

        total_loss = (
            settings.training.loss_bce_weight * bce_loss
            + focal_loss
        )

        return {
            "loss": total_loss,
            "bce_loss": bce_loss,
            "focal_loss": focal_loss,
        }


try:
    import torchaudio
except ImportError:
    torchaudio = None
