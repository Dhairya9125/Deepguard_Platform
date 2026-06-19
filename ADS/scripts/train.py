#!/usr/bin/env python3
"""Training script for ADS — Audio Deepfake Detection System.

Trains the multi-branch deepfake detection model using:
- ASVspoof 2021, WaveFake, FakeAVCeleb datasets
- MLflow experiment tracking
- Configurable branches, hyperparameters, and augmentation

Usage:
    python scripts/train.py --dataset asvspoof --data-dir ./data/ASVspoof2021
    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --resume ./models/checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from ads.config.settings import settings
from ads.config.logging import setup_logging
from ads.models.ensemble.trainable import TrainableADS
from ads.pipeline.training import Trainer
from datasets.utils import create_data_loader, split_dataset

setup_logging()
logger = logging.getLogger("scripts.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ADS - Audio Deepfake Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="asvspoof",
                        choices=["asvspoof", "wavefake", "fakeavcelebrity", "composite"],
                        help="Dataset to train on")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Root directory containing dataset(s)")
    parser.add_argument("--dataset-task", type=str, default="LA",
                        help="ASVspoof task: LA (Logical Access) or DF (Deepfake)")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Weight decay")
    parser.add_argument("--grad-accum", type=int, default=None,
                        help="Gradient accumulation steps")
    parser.add_argument("--freeze-epochs", type=int, default=None,
                        help="Epochs to freeze encoder before unfreezing")

    # Model arguments
    parser.add_argument("--branches", type=str, nargs="+", default=None,
                        help="Branches to train: wavlm xlsr spectral voice_biometrics temporal diffusion adversarial")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory to save checkpoints")

    # MLflow arguments
    parser.add_argument("--mlflow-uri", type=str, default=None,
                        help="MLflow tracking URI")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="MLflow experiment name")

    # Hardware arguments
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Data loading workers")
    parser.add_argument("--amp", action="store_true", default=None,
                        help="Enable mixed precision (AMP)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU training")

    # Other
    parser.add_argument("--eval-only", action="store_true",
                        help="Run evaluation only (requires --resume)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file (overrides defaults)")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(args: argparse.Namespace) -> torch.utils.data.Dataset:
    """Load the specified dataset."""
    data_dir = Path(args.data_dir)

    if args.dataset == "asvspoof":
        from datasets.asvspoof import ASVspoofDataset

        dataset = ASVspoofDataset(
            root_dir=str(data_dir / "ASVspoof2021"),
            subset="train",
            task=args.dataset_task,
            target_sample_rate=settings.audio.sample_rate,
            max_duration=settings.training.max_audio_duration,
        )
        logger.info(
            f"Loaded ASVspoof2021 ({args.dataset_task}): "
            f"{len(dataset)} samples "
            f"(real={dataset.num_real}, fake={dataset.num_fake})"
        )

    elif args.dataset == "wavefake":
        from datasets.wavefake import WaveFakeDataset

        dataset = WaveFakeDataset(
            root_dir=str(data_dir / "WaveFake"),
            subset="train",
            target_sample_rate=settings.audio.sample_rate,
            max_duration=settings.training.max_audio_duration,
        )
        logger.info(f"Loaded WaveFake: {len(dataset)} samples")

    elif args.dataset == "fakeavcelebrity":
        from datasets.fakeavcelebrity import FakeAVCelebDataset

        dataset = FakeAVCelebDataset(
            root_dir=str(data_dir / "FakeAVCeleb"),
            subset="train",
            target_sample_rate=settings.audio.sample_rate,
            max_duration=settings.training.max_audio_duration,
        )
        logger.info(f"Loaded FakeAVCeleb: {len(dataset)} samples")

    elif args.dataset == "composite":
        from datasets.asvspoof import ASVspoofDataset
        from datasets.wavefake import WaveFakeDataset
        from datasets.fakeavcelebrity import FakeAVCelebDataset
        from datasets.base import CompositeDataset

        datasets_list = []
        for ds_name, ds_cls, ds_kwargs in [
            ("ASVspoof2021", ASVspoofDataset, {"task": args.dataset_task}),
            ("WaveFake", WaveFakeDataset, {}),
            ("FakeAVCeleb", FakeAVCelebDataset, {}),
        ]:
            ds_path = data_dir / ds_name
            if ds_path.exists():
                try:
                    ds = ds_cls(
                        root_dir=str(ds_path),
                        subset="train",
                        target_sample_rate=settings.audio.sample_rate,
                        max_duration=settings.training.max_audio_duration,
                        **ds_kwargs,
                    )
                    if len(ds) > 0:
                        datasets_list.append(ds)
                        logger.info(f"  Added {ds_name}: {len(ds)} samples")
                except Exception as e:
                    logger.warning(f"  Skipped {ds_name}: {e}")

        dataset = CompositeDataset(datasets_list)
        logger.info(f"Composite dataset: {len(dataset)} total samples from {len(datasets_list)} sources")

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    return dataset


def build_train_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Build training config dict from args + defaults."""
    config = {}

    if args.epochs is not None:
        config["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["learning_rate"] = args.lr
    if args.weight_decay is not None:
        config["weight_decay"] = args.weight_decay
    if args.grad_accum is not None:
        config["gradient_accumulation_steps"] = args.grad_accum
    if args.freeze_epochs is not None:
        config["freeze_encoder_epochs"] = args.freeze_epochs
    if args.checkpoint_dir is not None:
        config["checkpoint_dir"] = args.checkpoint_dir
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.amp is not None:
        config["use_amp"] = args.amp
    if args.branches is not None:
        config["train_branches"] = args.branches
    if args.mlflow_uri is not None:
        config["mlflow_tracking_uri"] = args.mlflow_uri
    if args.experiment_name is not None:
        config["experiment_name"] = args.experiment_name

    return config


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.cpu:
        settings.model_backend = "cpu"
        settings.device = "cpu"

    train_config = build_train_config(args)
    dataset = load_dataset(args)

    train_ds, val_ds, test_ds = split_dataset(
        dataset, train_ratio=0.7, val_ratio=0.15, seed=args.seed
    )
    logger.info(
        f"Split: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test"
    )

    batch_size = train_config.get("batch_size", settings.training.batch_size)
    num_workers = train_config.get("num_workers", settings.training.num_workers)
    use_ws = settings.training.use_weighted_sampler

    weights = None
    if use_ws and hasattr(dataset, "get_sample_weights"):
        full_weights = dataset.get_sample_weights()
        if isinstance(train_ds, Subset):
            weights = full_weights[train_ds.indices]

    train_loader = create_data_loader(
        train_ds,
        batch_size=batch_size,
        shuffle=weights is None,
        num_workers=num_workers,
        use_weighted_sampler=weights is not None,
        sample_weights=weights,
    )
    val_loader = create_data_loader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = create_data_loader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
    )

    if args.resume and Path(args.resume).exists():
        logger.info(f"Resuming from checkpoint: {args.resume}")
        model = TrainableADS(config=train_config)
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        trainer = Trainer.from_checkpoint(args.resume, train_loader, val_loader)
        trainer.config.update(train_config)
    else:
        model = TrainableADS(config=train_config)
        trainer = Trainer(model, train_loader, val_loader, config=train_config)

    if args.eval_only:
        logger.info("Running evaluation only...")
        val_metrics = trainer._validate()
        logger.info(json.dumps(val_metrics, indent=2))
        return

    logger.info("Starting training...")
    result = trainer.train()

    logger.info(
        f"Training complete. "
        f"Best {result['best_metric_name']}: {result['best_metric']:.4f} "
        f"Checkpoint: {result['best_checkpoint']}"
    )

    logger.info("Running test evaluation...")
    trainer.model.eval()
    test_metrics = trainer._validate()
    logger.info(f"Test metrics: {json.dumps(test_metrics, indent=2)}")

    with open(Path(trainer.checkpoint_dir) / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    logger.info(f"Test metrics saved to {trainer.checkpoint_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
