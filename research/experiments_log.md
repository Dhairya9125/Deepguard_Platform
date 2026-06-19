# DeepGuard Experiments Log

This document tracks local model training and fine-tuning experiments logged via MLflow.

| Date | Engine | Model Branch | Hyperparameters | Val Loss | Val Acc | EER | Notes |
|------|--------|--------------|-----------------|----------|---------|-----|-------|
| 2026-06-16 | Audio | WavLM | lr=1e-4, b=32 | 0.45 | 89.2% | 4.1% | Initial baseline run using ASVspoof stub. |
| 2026-06-16 | Image | ResNet50 (Spatial) | lr=2e-4, b=64 | 0.38 | 91.5% | 3.8% | Baseline run on FaceForensics++. |
| pending | Video | 3D-CNN Temporal | TBD | TBD | TBD | TBD | Awaiting dataset download. |
