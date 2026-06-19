"""
ADS — WavLM Branch
Leverages WavLM (or its transformer features) for deep representation-based detection.
When torch+transformers are installed, loads WavLM-Large embeddings.
Falls back to spectral proxy features on CPU-only machines.
"""
import logging
from typing import Any, Dict
import numpy as np
from ..base_branch import BaseBranch
from ...domain.entities import BranchResult

logger = logging.getLogger(__name__)


class WavLMBranch(BaseBranch):
    name = "wavlm"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._model = None

    def _try_load_model(self):
        try:
            from transformers import WavLMModel, AutoFeatureExtractor  # type: ignore
            logger.info("Loading WavLM-Base+ (this may take a moment)...")
            self._feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
            self._model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
            self._model.eval()
            logger.info("WavLM model loaded.")
        except Exception as exc:
            logger.warning("WavLM model not available (%s). Using spectral proxy.", exc)
            self._model = None

    def predict(self, waveform: np.ndarray, features: Dict[str, Any]) -> BranchResult:
        # Use WavLM embeddings if available, else fall back to MFCC proxy
        if self._model is None:
            self._try_load_model()

        if self._model is not None:
            return self._predict_with_wavlm(waveform)
        else:
            return self._predict_proxy(features)

    def _predict_with_wavlm(self, waveform: np.ndarray) -> BranchResult:
        try:
            import torch  # type: ignore
            inputs = self._feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                outputs = self._model(**inputs)
            hidden = outputs.last_hidden_state.squeeze(0).numpy()
            # Simple anomaly score: high variance in hidden states → more natural
            variance = float(np.var(hidden))
            fake_score = float(np.clip(1.0 - variance / 0.5, 0.0, 1.0))
            return BranchResult(self.name, fake_score, 0.8, {"embedding_variance": round(variance, 6)})
        except Exception as exc:
            logger.error("WavLM inference error: %s", exc)
            return BranchResult(self.name, 0.5, 0.0, error=str(exc))

    def _predict_proxy(self, features: Dict[str, Any]) -> BranchResult:
        """Proxy: use MFCC delta entropy as a WavLM-style naturalness score."""
        delta = features.get("delta_mfcc", np.zeros((40, 100)))
        entropy = float(-np.sum(np.abs(delta) * np.log(np.abs(delta) + 1e-8)))
        fake_score = float(np.clip(1.0 - entropy / 500.0, 0.0, 1.0))
        return BranchResult(self.name, fake_score, 0.5, {"proxy_entropy": round(entropy, 4)})
