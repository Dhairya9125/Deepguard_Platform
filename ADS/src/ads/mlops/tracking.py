"""MLflow integration for experiment tracking and model registry."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ads.config.settings import settings

try:
    import mlflow
    from mlflow.tracking import MlflowClient

    _MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore
    MlflowClient = None  # type: ignore
    _MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """MLflow experiment tracker for ADS.

    Manages experiment tracking, metric logging, and model registry.
    Gracefully degrades to no-op when mlflow is not installed.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        self.tracking_uri = self.config.get("tracking_uri", settings.mlflow.tracking_uri)
        self.experiment_name = self.config.get("experiment_name", settings.mlflow.experiment_name)
        self.registry_uri = self.config.get("registry_uri", settings.mlflow.registry_uri)
        self.client = None

        if not _MLFLOW_AVAILABLE:
            logger.warning("mlflow not installed; tracking will be a no-op")
            return

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Start a new MLflow run."""
        if not _MLFLOW_AVAILABLE:
            logger.debug("start_run: mlflow not available")
            return None
        base_name = self.config.get("run_name_prefix", "ads-run")
        name = run_name or f"{base_name}"
        return mlflow.start_run(run_name=name, tags=tags or {})

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to current run."""
        if not _MLFLOW_AVAILABLE:
            return
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to current run."""
        if not _MLFLOW_AVAILABLE:
            return
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact file."""
        if not _MLFLOW_AVAILABLE:
            return
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None,
    ) -> None:
        """Log a PyTorch model to MLflow."""
        if not _MLFLOW_AVAILABLE:
            return
        import torch

        if isinstance(model, torch.nn.Module):
            mlflow.pytorch.log_model(
                model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=input_example,
            )
        else:
            mlflow.sklearn.log_model(
                model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=input_example,
            )

    def register_model(
        self,
        model_name: str,
        run_id: Optional[str] = None,
        stage: str = "None",
    ) -> Any:
        """Register a model in the MLflow Model Registry."""
        if not _MLFLOW_AVAILABLE or self.client is None:
            return None
        source_run_id = run_id or mlflow.active_run().info.run_id
        result = mlflow.register_model(
            f"runs:/{source_run_id}/model",
            model_name,
        )

        if stage != "None":
            self.client.transition_model_version_stage(
                name=model_name,
                version=result.version,
                stage=stage,
            )

        return result

    def get_best_model(
        self, metric: str = "val_auc", stage: str = "Production"
    ) -> Optional[Any]:
        """Get the best model from registry by metric."""
        if not _MLFLOW_AVAILABLE:
            return None
        try:
            model_uri = f"models:/{self.experiment_name}/{stage}"
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            logger.warning(f"Could not load best model: {e}")
            return None
