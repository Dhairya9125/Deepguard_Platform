"""
DeepGuard Platform — Image Detection Subsystem (IDS)
Module  : engines.image-engine.data.datasets
Layer   : Training Pipeline — Dataset Loaders

Production-ready PyTorch Dataset loaders for the 5 target datasets.
Expects specific directory structures and metadata formats for each dataset.
"""

import os
import glob
import logging
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

logger = logging.getLogger(__name__)


class BaseIDSDataset(Dataset):
    """Base class providing common transform pipelines."""
    def __init__(self, root_dir: str, split: str = 'train', transform: Optional[T.Compose] = None):
        self.root_dir = Path(root_dir)
        self.split = split
        
        # Default DeepGuard transform pipeline if none provided
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            # Normalization parameters typically used for ImageNet backbones
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.transform(img)


class FaceForensicsDataset(BaseIDSDataset):
    """
    Loader for FaceForensics++ (Baseline)
    Expected structure:
    root/
       real/
           c23/...
       fake/
           Deepfakes/c23/...
           FaceSwap/c23/...
    """
    def __init__(self, root_dir: str, split: str = 'train', compression: str = 'c23', transform=None):
        super().__init__(root_dir, split, transform)
        self.compression = compression
        self.samples = [] # List of (filepath, label)
        
        logger.info(f"Scanning FaceForensics++ Dataset | compression={compression} | split={split}")
        # In a real environment, this would parse the train/val/test split JSON files.
        # For this skeleton, we assume the directory structure is pre-split.
        
        real_dir = self.root_dir / split / 'real' / compression
        fake_dir = self.root_dir / split / 'fake'
        
        if real_dir.exists():
            for img_path in real_dir.rglob("*.png"):
                self.samples.append((img_path, 0.0))
                
        if fake_dir.exists():
            for manipulation in ['Deepfakes', 'FaceSwap', 'Face2Face', 'NeuralTextures']:
                manip_dir = fake_dir / manipulation / compression
                if manip_dir.exists():
                    for img_path in manip_dir.rglob("*.png"):
                        self.samples.append((img_path, 1.0))
                        
        logger.info(f"Found {len(self.samples)} FaceForensics++ samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        img_tensor = self.load_image(path)
        return img_tensor, torch.tensor([label], dtype=torch.float32)


class GenImageDataset(BaseIDSDataset):
    """
    Loader for GenImage Dataset (Diffusion models)
    Tests realism against Midjourney, Stable Diffusion, etc.
    """
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        super().__init__(root_dir, split, transform)
        self.samples = []
        
        logger.info(f"Scanning GenImage Dataset | split={split}")
        split_dir = self.root_dir / split
        
        if split_dir.exists():
            for generator in split_dir.iterdir():
                if generator.is_dir():
                    # GenImage typically has 'nature' (real) and 'ai' (fake) subfolders per generator
                    real_dir = generator / 'nature'
                    fake_dir = generator / 'ai'
                    
                    if real_dir.exists():
                        for img_path in real_dir.glob("*.jpg"):
                            self.samples.append((img_path, 0.0))
                    if fake_dir.exists():
                        for img_path in fake_dir.glob("*.jpg"):
                            self.samples.append((img_path, 1.0))
                            
        logger.info(f"Found {len(self.samples)} GenImage samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        img_tensor = self.load_image(path)
        return img_tensor, torch.tensor([label], dtype=torch.float32)


class CelebDFDataset(BaseIDSDataset):
    """
    Loader for Celeb-DF (High-quality realism)
    """
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        super().__init__(root_dir, split, transform)
        self.samples = []
        
        logger.info(f"Scanning Celeb-DF Dataset | split={split}")
        # Celeb-DF has 'Celeb-real' and 'Celeb-synthesis'
        real_dir = self.root_dir / split / 'Celeb-real'
        fake_dir = self.root_dir / split / 'Celeb-synthesis'
        
        if real_dir.exists():
            for img_path in real_dir.rglob("*.jpg"):
                self.samples.append((img_path, 0.0))
        if fake_dir.exists():
            for img_path in fake_dir.rglob("*.jpg"):
                self.samples.append((img_path, 1.0))
                
        logger.info(f"Found {len(self.samples)} Celeb-DF samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        img_tensor = self.load_image(path)
        return img_tensor, torch.tensor([label], dtype=torch.float32)


class UFDDataset(BaseIDSDataset):
    """
    Loader for UniversalFakeDetect (UFD) - Out of Distribution Testing
    Contains 20 unseen generators.
    """
    def __init__(self, root_dir: str, transform=None):
        # UFD is typically used just for OOD testing, so no split arg needed
        super().__init__(root_dir, split='test', transform=transform)
        self.samples = []
        
        logger.info(f"Scanning UFD Dataset for OOD Evaluation.")
        
        # Assumes structure: UFD/real/... and UFD/fake/generator_name/...
        real_dir = self.root_dir / 'real'
        fake_dir = self.root_dir / 'fake'
        
        if real_dir.exists():
            for img_path in real_dir.rglob("*.jpg"):
                self.samples.append((img_path, 0.0))
        if fake_dir.exists():
            for img_path in fake_dir.rglob("*.jpg"):
                self.samples.append((img_path, 1.0))
                
        logger.info(f"Found {len(self.samples)} UFD samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        img_tensor = self.load_image(path)
        return img_tensor, torch.tensor([label], dtype=torch.float32)


class ForgeryNetDataset(BaseIDSDataset):
    """
    Loader for ForgeryNet (Localization)
    Crucial for training the Layer 3 Spatial Decoder.
    Returns: (image_tensor, label_tensor, mask_tensor)
    """
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        super().__init__(root_dir, split, transform)
        self.samples = [] # List of (image_path, mask_path, label)
        
        # Mask specific transforms
        self.mask_transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor() # Converts to [0, 1]
        ])
        
        logger.info(f"Scanning ForgeryNet Dataset (Localization) | split={split}")
        
        images_dir = self.root_dir / split / 'images'
        masks_dir = self.root_dir / split / 'masks'
        
        if images_dir.exists() and masks_dir.exists():
            # ForgeryNet typically provides paired files
            for img_path in images_dir.rglob("*.jpg"):
                # Construct expected mask path
                rel_path = img_path.relative_to(images_dir)
                mask_path = masks_dir / rel_path.with_suffix('.png')
                
                if mask_path.exists():
                    # Check if fake or real based on directory or filename rules
                    # Assuming ForgeryNet standard rule: real images have blank masks (sum == 0)
                    # For performance, we assume directory structure dictates this
                    label = 1.0 if 'fake' in str(img_path) else 0.0
                    self.samples.append((img_path, mask_path, label))
                    
        logger.info(f"Found {len(self.samples)} ForgeryNet image-mask pairs.")

    def load_mask(self, path: Path) -> torch.Tensor:
        mask = Image.open(path).convert("L")
        return self.mask_transform(mask)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path, mask_path, label = self.samples[idx]
        
        img_tensor = self.load_image(img_path)
        mask_tensor = self.load_mask(mask_path)
        
        return img_tensor, torch.tensor([label], dtype=torch.float32), mask_tensor
