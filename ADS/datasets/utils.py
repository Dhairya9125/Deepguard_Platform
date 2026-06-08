"""Dataset utility functions for training."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


def create_data_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    use_weighted_sampler: bool = False,
    sample_weights: Optional[torch.Tensor] = None,
) -> DataLoader:
    """Create a DataLoader with optional weighted sampling.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Shuffle data (ignored if weighted sampler used)
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
        use_weighted_sampler: Use weighted random sampler
        sample_weights: Weights for each sample

    Returns:
        DataLoader instance
    """
    sampler = None
    if use_weighted_sampler and sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=_collate_fn,
    )


def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function for variable-length audio batches."""
    waveforms = [item["waveform"] for item in batch]
    features_list = [item["features"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    max_len = max(w.shape[-1] for w in waveforms)
    padded_waveforms = []
    for w in waveforms:
        if w.shape[-1] < max_len:
            pad = max_len - w.shape[-1]
            w = torch.nn.functional.pad(w, (0, pad))
        padded_waveforms.append(w)
    waveforms = torch.stack(padded_waveforms)

    max_feat_len = max(f.shape[-1] for f in features_list)
    padded_features = []
    for f in features_list:
        if f.shape[-1] < max_feat_len:
            pad = max_feat_len - f.shape[-1]
            f = torch.nn.functional.pad(f, (0, pad))
        padded_features.append(f)
    features = torch.stack(padded_features)

    return {
        "waveform": waveforms,
        "features": features,
        "label": labels,
        "sample_ids": [item.get("sample_id", "") for item in batch],
    }


def split_dataset(
    dataset: torch.utils.data.Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple:
    """Split dataset into train/val/test sets."""
    total = len(dataset)
    lengths = [
        int(total * train_ratio),
        int(total * val_ratio),
        total - int(total * train_ratio) - int(total * val_ratio),
    ]
    return torch.utils.data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(seed))
