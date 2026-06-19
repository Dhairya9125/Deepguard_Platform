"""
DeepGuard Platform — Model Registry
Module  : ml.mlops.model_registry

Tracks which model versions are deployed to production vs staging.
File-based (uses JSON manifest) — no MLflow Model Registry server needed.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REGISTRY_FILE = Path(__file__).parent / "model_registry.json"


class ModelRegistry:
    """
    Simple file-based model registry.
    Tracks registered model versions, their paths, and deployment status.
    """

    def __init__(self, registry_path: Path = REGISTRY_FILE) -> None:
        self.registry_path = registry_path
        self._data: Dict[str, List[Dict[str, Any]]] = self._load()

    def _load(self) -> Dict[str, List[Dict[str, Any]]]:
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, default=str)

    def register(
        self,
        model_name: str,
        version: str,
        artifact_path: str,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
        stage: str = "staging",
    ) -> Dict[str, Any]:
        """
        Register a new model version.

        Args:
            model_name: e.g. 'ads-spectral-cnn', 'ids-pipeline'
            version:    semantic version string e.g. '1.2.0'
            artifact_path: path to model weights/checkpoint
            metrics:    evaluation metrics dict
            tags:       arbitrary string tags
            stage:      'staging' or 'production'
        """
        entry = {
            "version": version,
            "artifact_path": artifact_path,
            "metrics": metrics or {},
            "tags": tags or {},
            "stage": stage,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        if model_name not in self._data:
            self._data[model_name] = []
        self._data[model_name].append(entry)
        self._save()
        logger.info("Registered model '%s' v%s (stage=%s)", model_name, version, stage)
        return entry

    def get_latest(
        self, model_name: str, stage: str = "production"
    ) -> Optional[Dict[str, Any]]:
        """Get the latest version of a model at the given stage."""
        versions = [
            v for v in self._data.get(model_name, [])
            if v["stage"] == stage
        ]
        return versions[-1] if versions else None

    def list_models(self) -> List[str]:
        """Return all registered model names."""
        return list(self._data.keys())

    def promote(
        self, model_name: str, version: str, new_stage: str = "production"
    ) -> bool:
        """Promote a model version to a new stage."""
        for v in self._data.get(model_name, []):
            if v["version"] == version:
                v["stage"] = new_stage
                self._save()
                logger.info("Promoted '%s' v%s → %s", model_name, version, new_stage)
                return True
        return False
