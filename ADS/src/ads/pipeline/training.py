"""Training pipeline for end-to-end ADS model training.

Provides:
- Training loop with gradient accumulation
- Validation with EER, AUC, F1 metrics
- Checkpointing (top-k by val metric)
- MLflow experiment tracking
- Learning rate scheduling
- Early stopping
- Mixed precision support
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

from ads.config.settings import settings
from ads.mlops.tracking import ExperimentTracker
from ads.models.ensemble.trainable import TrainableADS

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for ADS model with full training loop, metrics, and checkpointing.

    Usage:
        model = TrainableADS()
        trainer = Trainer(model, train_loader, val_loader)
        trainer.train()
    """

    def __init__(
        self,
        model: TrainableADS,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[dict] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}

        self.device = torch.device("cpu")
        self.model.to(self.device)

        self.num_epochs = self.config.get("num_epochs", settings.training.num_epochs)
        self.learning_rate = self.config.get("learning_rate", settings.training.learning_rate)
        self.weight_decay = self.config.get("weight_decay", settings.training.weight_decay)
        self.warmup_steps = self.config.get("warmup_steps", settings.training.warmup_steps)
        self.max_grad_norm = self.config.get("max_grad_norm", settings.training.max_grad_norm)
        self.grad_accum_steps = self.config.get("gradient_accumulation_steps", settings.training.gradient_accumulation_steps)
        self.use_amp = self.config.get("use_amp", settings.training.use_amp)
        self.early_stop_patience = self.config.get("early_stop_patience", settings.training.early_stop_patience)
        self.early_stop_metric = self.config.get("early_stop_metric", settings.training.early_stop_metric)
        self.save_top_k = self.config.get("save_top_k", settings.training.save_top_k)
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", settings.training.checkpoint_dir))
        self.log_every = self.config.get("log_every_n_steps", settings.training.log_every_n_steps)
        self.eval_every = self.config.get("eval_every_n_steps", settings.training.eval_every_n_steps)
        self.freeze_encoder_epochs = self.config.get("freeze_encoder_epochs", settings.training.freeze_encoder_epochs)

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler(enabled=self.use_amp) if self.use_amp else None

        self.tracker = ExperimentTracker(config or {})

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float("inf")
        self.best_checkpoint_path: Optional[str] = None
        self.early_stop_counter = 0
        self.top_k_checkpoints: List[Tuple[float, str]] = []

        self.train_metrics: Dict[str, List[float]] = {}
        self.val_metrics: Dict[str, List[float]] = {}

    def _build_optimizer(self) -> torch.optim.Optimizer:
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        params = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(params, lr=self.learning_rate, betas=(0.9, 0.999))

    def _build_scheduler(self) -> Any:
        total_steps = len(self.train_loader) * self.num_epochs // self.grad_accum_steps
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - self.warmup_steps,
            eta_min=self.learning_rate * 0.01,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps],
        )

    def train(self) -> Dict[str, Any]:
        """Run full training loop.

        Returns:
            Dict with best metrics and checkpoint path
        """
        logger.info(
            f"Starting training: {self.num_epochs} epochs, "
            f"{len(self.train_loader.dataset)} train samples, "
            f"{len(self.val_loader.dataset)} val samples"
        )

        self.tracker.start_run(tags={"model": "TrainableADS", "device": str(self.device)})
        self.tracker.log_params({
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.train_loader.batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "optimizer": "adamw",
            "scheduler": "warmup_cosine",
            "branches": list(self.model.branches.keys()),
            "freeze_encoder_epochs": self.freeze_encoder_epochs,
        })

        for epoch in range(1, self.num_epochs + 1):
            self.current_epoch = epoch
            self._handle_encoder_freeze(epoch)

            train_metrics = self._train_epoch()
            val_metrics = self._validate()

            self.train_metrics.setdefault("loss", []).append(train_metrics.get("loss", 0))
            for k, v in val_metrics.items():
                self.val_metrics.setdefault(k, []).append(v)

            self.tracker.log_metrics(
                {f"train_{k}": v for k, v in train_metrics.items()},
                step=epoch,
            )
            self.tracker.log_metrics(
                {f"val_{k}": v for k, v in val_metrics.items()},
                step=epoch,
            )

            current_metric = val_metrics.get(self.early_stop_metric, float("inf"))
            self._save_checkpoint(epoch, val_metrics)
            self._check_early_stop(current_metric, epoch)

            logger.info(
                f"Epoch {epoch}/{self.num_epochs} | "
                f"Train Loss: {train_metrics.get('loss', 0):.4f} | "
                f"Val Loss: {val_metrics.get('loss', 0):.4f} | "
                f"EER: {val_metrics.get('eer', 0):.4f} | "
                f"AUC: {val_metrics.get('auc', 0):.4f} | "
                f"F1: {val_metrics.get('f1', 0):.4f}"
            )

            if self.early_stop_counter >= self.early_stop_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        self.tracker.log_artifact(str(self.best_checkpoint_path)) if self.best_checkpoint_path else None
        logger.info(f"Training complete. Best {self.early_stop_metric}: {self.best_metric:.4f}")

        return {
            "best_metric": self.best_metric,
            "best_metric_name": self.early_stop_metric,
            "best_checkpoint": self.best_checkpoint_path,
            "total_epochs": epoch,
            "final_train_loss": train_metrics.get("loss", 0),
            "final_val_metrics": val_metrics,
        }

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_focal = 0.0
        num_batches = 0
        epoch_start = time.time()

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            waveform = batch["waveform"].to(self.device)
            labels = batch["label"].to(self.device)

            if self.use_amp:
                with autocast():
                    output = self.model(waveform)
                    loss_dict = self.model.compute_loss(output["logits"], labels)
                    loss = loss_dict["loss"] / self.grad_accum_steps
                self.scaler.scale(loss).backward()
            else:
                output = self.model(waveform)
                loss_dict = self.model.compute_loss(output["logits"], labels)
                loss = loss_dict["loss"] / self.grad_accum_steps
                loss.backward()

            total_loss += loss_dict["loss"].item()
            total_bce += loss_dict["bce_loss"].item()
            total_focal += loss_dict["focal_loss"].item()
            num_batches += 1

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                if self.global_step % self.log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"Step {self.global_step} | Loss: {loss_dict['loss'].item():.4f} "
                        f"| LR: {lr:.2e}"
                    )

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(num_batches, 1)

        return {
            "loss": avg_loss,
            "bce_loss": total_bce / max(num_batches, 1),
            "focal_loss": total_focal / max(num_batches, 1),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "epoch_time_s": epoch_time,
        }

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        all_probs: List[float] = []
        all_labels: List[float] = []
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            waveform = batch["waveform"].to(self.device)
            labels = batch["label"].to(self.device)

            output = self.model(waveform)
            loss_dict = self.model.compute_loss(output["logits"], labels)

            total_loss += loss_dict["loss"].item()
            num_batches += 1
            all_probs.extend(output["probs"].cpu().numpy().flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())

        metrics = compute_metrics(np.array(all_probs), np.array(all_labels))
        metrics["loss"] = total_loss / max(num_batches, 1)

        return metrics

    def _handle_encoder_freeze(self, epoch: int) -> None:
        if epoch == self.freeze_encoder_epochs + 1:
            for name, branch in self.model.branches.items():
                if hasattr(branch, "freeze_encoder") and branch.freeze_encoder:
                    for param in branch.parameters():
                        param.requires_grad = True
                    logger.info(f"Unfrozen encoder at epoch {epoch}")

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        metric_value = metrics.get(self.early_stop_metric, 0)
        save_metric = metric_value if self.early_stop_metric != "val_eer" else -metric_value

        ckpt_name = f"epoch_{epoch:03d}_{self.early_stop_metric}_{metric_value:.4f}.pt"
        ckpt_path = str(self.checkpoint_dir / ckpt_name)

        is_lower_better = self.early_stop_metric in ("val_eer", "val_loss")
        is_best = (save_metric < self.best_metric) if is_lower_better else (save_metric > self.best_metric)

        if is_best:
            self.best_metric = metric_value
            self.best_checkpoint_path = ckpt_path
            self.early_stop_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "metrics": metrics,
                "config": settings.model_dump(mode="json"),
            }, ckpt_path)

            best_path = str(self.checkpoint_dir / "best_model.pt")
            torch.save(self.model.state_dict(), best_path)
            logger.info(f"New best model saved: {ckpt_path} ({self.early_stop_metric}={metric_value:.4f})")

            self._register_model(ckpt_path, metrics)

        self.top_k_checkpoints.append((metric_value if is_lower_better else -metric_value, ckpt_path))
        self.top_k_checkpoints.sort(key=lambda x: x[0])
        self.top_k_checkpoints = self.top_k_checkpoints[: self.save_top_k]

    def _check_early_stop(self, current_metric: float, epoch: int) -> None:
        is_lower_better = self.early_stop_metric in ("val_eer", "val_loss")
        if is_lower_better:
            if current_metric >= self.best_metric:
                self.early_stop_counter += 1
            else:
                self.early_stop_counter = 0
        else:
            if current_metric <= self.best_metric:
                self.early_stop_counter += 1
            else:
                self.early_stop_counter = 0

    def _register_model(self, checkpoint_path: str, metrics: Dict[str, float]) -> None:
        try:
            self.tracker.log_model(
                model=self.model,
                artifact_path="model",
                input_example=torch.randn(1, 16000),
            )
            self.tracker.register_model(
                model_name="ads-detector",
                stage="Staging",
            )
        except Exception as e:
            logger.warning(f"Model registration failed: {e}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> "Trainer":
        """Load trainer from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = TrainableADS()
        model.load_state_dict(checkpoint["model_state_dict"])
        trainer = cls(model, train_loader, val_loader)
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.current_epoch = checkpoint.get("epoch", 0)
        trainer.best_metric = checkpoint.get("metrics", {}).get(settings.training.early_stop_metric, float("inf"))
        return trainer


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute EER, AUC, F1, accuracy from probabilities and labels.

    Args:
        probs: Predicted probabilities (N,)
        labels: Binary labels (N,)

    Returns:
        Dict with eer, auc, f1, accuracy, precision, recall
    """
    from sklearn.metrics import (
        accuracy_score,
        auc,
        f1_score,
        precision_score,
        recall_score,
        roc_curve,
    )

    preds = (probs >= 0.5).astype(int)

    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)

    return {
        "eer": float(eer),
        "auc": float(roc_auc),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
    }
