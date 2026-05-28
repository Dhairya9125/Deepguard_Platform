"""
DeepGuard Platform — Image Detection Subsystem (IDS)
Package: engines.image-engine.data
Layer  : Training Pipeline — Dataset Loaders
"""

from .datasets import (
    FaceForensicsDataset,
    GenImageDataset,
    CelebDFDataset,
    ForgeryNetDataset,
    UFDDataset
)

__all__ = [
    "FaceForensicsDataset",
    "GenImageDataset",
    "CelebDFDataset",
    "ForgeryNetDataset",
    "UFDDataset"
]
