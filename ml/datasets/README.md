# DeepGuard Datasets

This directory contains the PyTorch `Dataset` and `DataLoader` classes required to train and evaluate the deepfake detection models.

## Available Loaders

- `asvspoof.py` — Loader for the [ASVspoof 2019 dataset](https://www.asvspoof.org/), used for training the Audio Deepfake Detection System (ADS).
- `faceforensics.py` — Loader for the [FaceForensics++ dataset](https://github.com/ondyari/FaceForensics), used for training the Video and Image engines.

> **Note:** The current implementations are stubbed for interface integration. To use them for actual model training, you must download the respective datasets (TB-scale) using the authenticated links provided by the dataset authors and update the loading logic.
