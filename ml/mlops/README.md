# MLOps

File-based MLflow experiment tracking and model registry for DeepGuard Platform.
**No server, no Docker required.**

## Experiment Tracking (MLflow)

```bash
# Install MLflow
pip install mlflow

# View experiments in browser (runs locally on port 5000, ~100 MB RAM)
mlflow ui --port 5000
```

All runs are stored in `./mlruns/` — pure filesystem, no database needed.

## Usage

```python
from ml.mlops.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker("audio-engine-v1")
with tracker.run(run_name="spectral-cnn-sweep"):
    tracker.log_params({"lr": 0.001, "epochs": 50})
    tracker.log_metrics({"eer": 0.018, "auc": 0.991})
```

## Model Registry

```python
from ml.mlops.model_registry import ModelRegistry

registry = ModelRegistry()
registry.register(
    model_name="ads-pipeline",
    version="1.0.0",
    artifact_path="./checkpoints/ads_v1.pt",
    metrics={"eer": 0.018, "auc": 0.991},
    stage="staging",
)
registry.promote("ads-pipeline", "1.0.0", stage="production")
```
