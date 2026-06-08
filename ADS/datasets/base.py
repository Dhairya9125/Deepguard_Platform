"""Base dataset class and data utilities for ADS training.

All dataset loaders inherit from AudioDeepfakeDataset and provide
standardized access to audio samples and labels.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from ads.config.settings import settings
from ads.signal_processing.features import SignalProcessor

logger = logging.getLogger(__name__)


@dataclass
class AudioSample:
    """A single audio sample with metadata and label."""

    audio_path: str
    label: int  # 0 = bonafide, 1 = spoof/deepfake
    sample_id: str = ""
    speaker_id: str = ""
    dataset_name: str = ""
    manipulation_type: str = ""
    language: str = ""


class AudioDeepfakeDataset(Dataset, ABC):
    """Abstract base class for all deepfake audio datasets.

    Subclasses must implement _load_filelist() to return the list
    of AudioSample objects.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        target_sample_rate: int = 16000,
        max_duration: float = 30.0,
        subset: str = "train",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform or self._default_transform
        self.target_sample_rate = target_sample_rate
        self.max_duration = max_duration
        self.subset = subset
        self.signal_processor = SignalProcessor()

        self.samples: List[AudioSample] = []
        self._load_filelist()

        if not self.samples:
            logger.warning(f"No samples found in {root_dir} for subset '{subset}'")

    @abstractmethod
    def _load_filelist(self) -> None:
        """Populate self.samples from the dataset directory structure."""

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        waveform, sr = self._load_audio(sample.audio_path)
        waveform = self._resample(waveform, sr)
        waveform = self._normalize_length(waveform)
        features = self.transform(waveform)

        return {
            "waveform": waveform,
            "features": features,
            "label": torch.tensor(sample.label, dtype=torch.float32),
            "sample_id": sample.sample_id,
            "speaker_id": sample.speaker_id,
            "audio_path": sample.audio_path,
        }

    def _load_audio(self, path: str) -> Tuple[torch.Tensor, int]:
        import torchaudio

        try:
            waveform, sr = torchaudio.load(path)
            return waveform, sr
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return torch.zeros(1, self.target_sample_rate), self.target_sample_rate

    def _resample(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        if orig_sr != self.target_sample_rate:
            import torchaudio

            resampler = torchaudio.transforms.Resample(orig_sr, self.target_sample_rate)
            return resampler(waveform)
        return waveform

    def _normalize_length(self, waveform: torch.Tensor) -> torch.Tensor:
        max_samples = int(self.max_duration * self.target_sample_rate)
        if waveform.shape[-1] > max_samples:
            waveform = waveform[..., :max_samples]
        elif waveform.shape[-1] < max_samples:
            pad = max_samples - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        return waveform

    @staticmethod
    def _default_transform(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0)

    def get_sample_weights(self) -> torch.Tensor:
        """Compute class-balanced sample weights for weighted sampling."""
        labels = torch.tensor([s.label for s in self.samples])
        n_real = (labels == 0).sum().float()
        n_fake = (labels == 1).sum().float()
        weights = torch.where(labels == 0, 1.0 / n_real, 1.0 / n_fake)
        return weights


class CompositeDataset(Dataset):
    """Combines multiple datasets into one unified dataset."""

    def __init__(self, datasets: List[AudioDeepfakeDataset]) -> None:
        self.datasets = datasets
        self.cumulative_sizes: List[int] = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_sizes.append(total)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        for ds_idx, ds in enumerate(self.datasets):
            if idx < self.cumulative_sizes[ds_idx]:
                local_idx = idx - (self.cumulative_sizes[ds_idx - 1] if ds_idx > 0 else 0)
                return ds[local_idx]
        raise IndexError(f"Index {idx} out of range")
