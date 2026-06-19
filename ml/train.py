"""
DeepGuard Platform — Master Model Training Pipeline

This is an industry-level PyTorch Lightning training script.
It supports distributed training (DDP), mixed precision (16-bit),
and tracks all metrics, hyperparameters, and model checkpoints directly into MLflow.

Supported Modalities & Datasets:
--------------------------------
1. AUDIO:
   - ASVspoof 2019/2021 (Logical Access & Physical Access)
   - VoxCeleb2 (For 'Real' class augmentation & contrastive pre-training)
   - VCTK / LibriTTS (Real clean speech baseline)
   - WaveFake (For vocoder artifact detection)

2. IMAGE / VIDEO (Spatial & Temporal):
   - FaceForensics++ (Deepfakes, Face2Face, FaceSwap, NeuralTextures)
   - DFDC (Deepfake Detection Challenge - Meta/Kaggle)
   - Celeb-DF v2 (High quality deepfakes, less artifacts than FF++)
   - WildDeepfake (In-the-wild internet deepfakes)

Prerequisites:
  pip install torch torchvision torchaudio pytorch-lightning mlflow torchmetrics wandb albumentations librosa
"""

import os
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
from torchmetrics import Accuracy, AUROC, F1Score

import sys
# Ensure engines module is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.datasets.streaming_loader import StreamingDataModule

# ---------------------------------------------------------------------------
# Lightning Module (The Engine)
# ---------------------------------------------------------------------------

class DeepGuardModel(pl.LightningModule):
    def __init__(self, model_class: nn.Module, lr=1e-4, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters(ignore=['model_class'])
        self.model = model_class
        
        # Loss function with Label Smoothing to prevent overconfidence on synthetic data
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Metrics
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.val_f1 = F1Score(task="binary")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(-1)
        loss = self.criterion(logits, y)
        
        preds = torch.sigmoid(logits)
        self.train_acc(preds, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(-1)
        loss = self.criterion(logits, y)
        
        preds = torch.sigmoid(logits)
        self.val_acc(preds, y)
        self.val_auroc(preds, y)
        self.val_f1(preds, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_auroc', self.val_auroc, prog_bar=True)
        self.log('val_f1', self.val_f1, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        # Cosine Annealing with Warm Restarts for aggressive learning
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}
        }

# ---------------------------------------------------------------------------
# Training Orchestration
# ---------------------------------------------------------------------------

def main(args):
    pl.seed_everything(42)

    # 1. Setup MLflow Logger
    mlflow_logger = MLFlowLogger(
        experiment_name=args.experiment_name,
        tracking_uri=args.mlflow_uri,
    )
    
    # Log the datasets being used
    mlflow_logger.log_hyperparams({"datasets": args.datasets, "modality": args.modality})

    # 2. Callbacks
    # val_check_interval: run a validation pass every N training steps so
    # val_loss/val_auroc are always available when callbacks fire.
    val_interval = max(1, args.max_steps // 5)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"models/checkpoints/{args.experiment_name}",
        filename="{step:05d}-{val_auroc:.4f}",
        save_top_k=3,
        monitor="val_auroc",
        mode="max",
        every_n_train_steps=val_interval,
    )
    # Monitor train_loss so early-stopping works even when val hasn't run yet.
    early_stop = EarlyStopping(monitor="train_loss", patience=5, mode="min",
                               check_on_train_epoch_end=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 3. Initialize DataModule & Model
    if args.modality == 'image':
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../engines/image-engine')))
        from img_feature_extraction.spatial_branch import ClipSpatialBranch
        # We output a projection dim of 1 for BCEWithLogitsLoss directly
        backbone = ClipSpatialBranch(proj_dim=1, freeze_backbone=False)
    elif args.modality == 'audio':
        # Since WavLM expects inputs, we'll use a standard Wav2Vec/WavLM based sequential model
        # Or load the WavLMBranch and adapt it. For simplicity in streaming, we use a basic CNN or imported WavLMBranch
        from transformers import WavLMModel
        class CustomAudioBranch(nn.Module):
            def __init__(self):
                super().__init__()
                self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
                self.classifier = nn.Linear(768, 1)
            def forward(self, x):
                # x shape: (B, 1, seq_len) -> (B, seq_len)
                x = x.squeeze(1)
                out = self.wavlm(x).last_hidden_state
                # Pool over time
                out = out.mean(dim=1)
                return self.classifier(out)
        backbone = CustomAudioBranch()
    elif args.modality == 'video':
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../engines/image-engine')))
        from img_feature_extraction.spatial_branch import ClipSpatialBranch
        
        class CustomVideoBranch(nn.Module):
            def __init__(self):
                super().__init__()
                # Output a dense vector instead of logits for temporal pooling
                self.spatial = ClipSpatialBranch(proj_dim=256, freeze_backbone=False)
                self.classifier = nn.Linear(256, 1)
                
            def forward(self, x):
                # x shape: [B, T, C, H, W]
                B, T, C, H, W = x.shape
                x = x.view(B * T, C, H, W)
                spatial_feats = self.spatial(x) # [B*T, 256]
                spatial_feats = spatial_feats.view(B, T, -1)
                pooled = spatial_feats.mean(dim=1) # Average pool over time
                return self.classifier(pooled)
                
        backbone = CustomVideoBranch()
    else:
        raise ValueError("Unsupported modality for actual training branch injection")
    
    model = DeepGuardModel(model_class=backbone, lr=args.lr)
    datamodule = StreamingDataModule(modality=args.modality, batch_size=args.batch_size)

    # 4. Trainer
    trainer = pl.Trainer(
        max_steps=args.max_steps,         # Streaming datasets use steps, not epochs
        val_check_interval=val_interval,  # Run validation every N steps
        accelerator="auto",               # Automatically uses GPUs/TPUs if available
        devices="auto",
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, early_stop, lr_monitor],
        precision="16-mixed",             # Mixed precision (falls back to bf16 on CPU)
        gradient_clip_val=1.0,
        log_every_n_steps=1,              # Log every step so metrics are always fresh
    )

    # 5. Execute Training
    logging.info(f"Starting training for {args.modality} modality on {args.datasets}")
    trainer.fit(model, datamodule=datamodule)
    
    # 6. Save final weights for API usage
    final_path = Path(f"models/production/{args.modality}_final.pt")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.model.state_dict(), final_path)
    logging.info(f"Production model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepGuard Core Training Pipeline")
    parser.add_argument("--experiment_name", type=str, default="deepguard_v1")
    parser.add_argument("--modality", type=str, choices=['image', 'audio', 'video'], required=True)
    parser.add_argument("--datasets", type=str, help="Comma separated list (e.g. dfdc,faceforensics,celebdf)")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--mlflow_uri", type=str, default="sqlite:///mlflow.db")
    
    args = parser.parse_args()
    main(args)
