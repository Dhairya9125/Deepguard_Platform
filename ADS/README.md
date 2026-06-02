# Audio Deepfake Detection System (ADS)

Enterprise-grade audio deepfake detection system capable of detecting AI voice cloning, synthetic speech, voice conversion, diffusion-based audio attacks, and audio splicing with temporal localization and explainable forensic reports.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│               Multi-Branch Detection Engine               │
├──────────┬──────────┬──────────┬──────────┬───────────────┤
│  WavLM   │  XLS-R   │ Spectral │  Voice   │   Temporal    │
│  Large   │          │   CNN    │Biometrics│   Splice      │
│ Branch   │ Branch   │ Branch   │ Branch   │   Branch      │
├──────────┴──────────┴──────────┴──────────┴───────────────┤
│                 Fusion Transformer                         │
├──────────────────────────────────────────────────────────┤
│              Localization + Explainability                 │
└──────────────────────────────────────────────────────────┘
```

### Detection Capabilities

- AI Voice Cloning
- Synthetic Speech Generation
- Voice Conversion Attacks
- Diffusion-based Audio Attacks
- Audio Splicing & Partial Manipulation
- Speaker Impersonation
- Adversarial Audio Attacks
- Multilingual Synthetic Speech

### Performance Targets

| Metric | Target |
|--------|--------|
| ASVspoof EER | < 1.5% |
| Wild Audio EER | < 5% |
| AUC | > 99% |
| F1 | > 97% |
| Localization Accuracy | > 85% |
| Latency | < 2s |

## Quick Start

### Installation

```bash
git clone https://github.com/your-org/ads.git
cd ads
pip install -e ".[dev]"
```

### CLI Usage

```bash
# Analyze a single audio file
python -m ads detect path/to/audio.wav

# Generate forensic report
python -m ads detect path/to/audio.wav --report --output ./reports

# JSON output
python -m ads detect path/to/audio.wav --json

# Batch processing
python -m ads batch path/to/audio1.wav path/to/audio2.wav --output results.json

# Show extracted features
python -m ads features path/to/audio.wav
```

### API Usage

```bash
# Start the API server
python -m ads serve --port 8000

# Analyze audio
curl -X POST http://localhost:8000/v1/detect \
  -H "Authorization: Bearer <token>" \
  -F "file=@audio.wav"

# Get forensic report
curl http://localhost:8000/v1/report/<result_id>?format=json
```

### Docker

```bash
docker compose -f docker/docker-compose.yml up -d
```

## Project Structure

```
src/
├── ads/                      # Main package
│   ├── config/               # YAML/env-based configuration
│   ├── domain/               # DDD domain entities
│   ├── preprocessing/        # VAD, diarization, segmentation
│   ├── signal_processing/    # MFCC, LFCC, CQCC, spectrograms
│   ├── models/               # 7 model branches
│   │   ├── wavlm_branch/
│   │   ├── xlsr_branch/
│   │   ├── spectral_cnn_branch/
│   │   ├── voice_biometrics/
│   │   ├── temporal_splice/
│   │   ├── diffusion_detector/
│   │   └── adversarial_detector/
│   ├── fusion/               # Cross-branch fusion
│   ├── localization/         # Temporal boundary detection
│   ├── explainability/       # SHAP, attention, reports
│   ├── application/          # Service orchestration
│   ├── infrastructure/       # Database, storage
│   ├── pipeline/             # Training/inference pipelines
│   ├── mlops/                # MLflow, experiment tracking
│   └── security/             # JWT, RBAC, encryption
├── api/                      # FastAPI routes
├── cli/                      # CLI interface
└── worker/                   # Celery async tasks
```

## Deployment

### Kubernetes

```bash
kubectl create namespace ads
kubectl apply -f k8s/manifests/ -n ads
```

### Terraform (AWS)

```bash
cd terraform
terraform init
terraform plan
terraform apply
```

## Testing

```bash
# All tests
pytest tests/

# Fast tests (no model loading)
pytest tests/ -m "not slow"

# Coverage report
pytest --cov=ads --cov-report=html
```

## License

MIT
