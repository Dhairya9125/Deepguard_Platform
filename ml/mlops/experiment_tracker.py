"""
DeepGuard Platform — MLflow Experiment Tracker
Module  : ml.mlops.experiment_tracker

File-based MLflow tracking — no server required.
Set MLFLOW_TRACKING_URI=file:./mlruns in your .env to use local file tracking.

Usage:
    tracker = ExperimentTracker("audio-engine-training")
    with tracker.run(run_name="spectral-cnn-v1") as run:
        tracker.log_params({"lr": 0.001, "batch_size": 32})
        tracker.log_metrics({"eer": 0.018, "auc": 0.991})
        tracker.log_artifact("model_weights.pt")
"""
import os
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)

# Default to file-based tracking (no server needed)
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")


class ExperimentTracker:
    """
    Thin wrapper around MLflow for DeepGuard experiment tracking.
    File-based by default — set MLFLOW_TRACKING_URI to a remote server for production.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = DEFAULT_TRACKING_URI,
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._active_run = None

        try:
            import mlflow  # type: ignore
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info(
                "MLflow tracking: experiment='%s' uri='%s'",
                experiment_name, tracking_uri,
            )
        except ImportError:
            logger.warning(
                "mlflow not installed. Install with: pip install mlflow\n"
                "Experiment tracking will be disabled."
            )

    @contextmanager
    def run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Generator[None, None, None]:
        """Context manager that wraps an MLflow run."""
        try:
            import mlflow  # type: ignore
            with mlflow.start_run(run_name=run_name, tags=tags) as run:
                self._active_run = run
                yield
                self._active_run = None
        except ImportError:
            # MLflow not installed — just yield without tracking
            yield
        except Exception as exc:
            logger.error("MLflow run error: %s", exc)
            yield

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters for the current run."""
        try:
            import mlflow
            mlflow.log_params(params)
        except Exception:
            pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log evaluation metrics."""
        try:
            import mlflow
            mlflow.log_metrics(metrics, step=step)
        except Exception:
            pass

    def log_artifact(self, artifact_path: str) -> None:
        """Log a file artifact (model weights, config, etc.)."""
        try:
            import mlflow
            mlflow.log_artifact(artifact_path)
        except Exception:
            pass

    def log_model_summary(self, summary: str) -> None:
        """Log a text description of the model."""
        try:
            import mlflow
            mlflow.set_tag("model_summary", summary[:500])
        except Exception:
            pass
