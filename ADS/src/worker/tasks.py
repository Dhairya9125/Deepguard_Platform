"""Celery worker tasks for async audio processing."""

from __future__ import annotations

import logging
from typing import Any, Dict

from celery import Celery

from ads.application.detector import DeepfakeDetector
from ads.config.settings import settings

logger = logging.getLogger(__name__)

celery_app = Celery(
    "ads",
    broker=settings.redis.url,
    backend=settings.redis.url,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

celery_app.conf.task_routes = {
    "ads.worker.tasks.analyze_audio": {"queue": "analysis"},
    "ads.worker.tasks.generate_report": {"queue": "reports"},
}


@celery_app.task(bind=True, max_retries=3, soft_time_limit=300, time_limit=600)
def analyze_audio(self, audio_path: str, return_report: bool = True) -> Dict[str, Any]:
    """Async task to analyze audio for deepfakes."""
    try:
        detector = DeepfakeDetector()
        result = detector.analyze(audio_path, return_report=return_report)
        return result.model_dump(mode="json")
    except Exception as e:
        logger.error(f"Analysis task failed: {e}")
        self.retry(exc=e, countdown=60)


@celery_app.task(bind=True, max_retries=2, soft_time_limit=120)
def generate_report(self, analysis_data: Dict[str, Any], output_dir: str) -> str:
    """Generate forensic report from analysis data."""
    try:
        from ads.explainability.engine import ExplainabilityEngine

        engine = ExplainabilityEngine()
        report = engine.generate_report(analysis_data, output_dir)
        return report.id
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        self.retry(exc=e, countdown=30)


@celery_app.task
def health_check() -> Dict[str, Any]:
    """Health check task."""
    return {"status": "healthy", "worker": "active"}
