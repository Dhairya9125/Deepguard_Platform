"""
ADS Pipeline Orchestrator Service.

Executes the full audio analysis pipeline as a FastAPI background task.
Job state is persisted to Neon DB (AnalysisJob + ForensicResult tables).
"""
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from apps.api.db.models import AnalysisJob, ForensicResult, JobStatus

logger = logging.getLogger(__name__)

# Inject audio-engine into sys.path
_root_dir = Path(__file__).resolve().parent.parent.parent.parent
_audio_engine_dir = _root_dir / "engines" / "audio-engine"
if str(_audio_engine_dir) not in sys.path:
    sys.path.insert(0, str(_audio_engine_dir))


# ---------------------------------------------------------------------------
# DB helper
# ---------------------------------------------------------------------------

async def _set_job_status(
    db: AsyncSession,
    job_id: str,
    status: str,
    message: str,
    results: dict | None = None,
) -> None:
    result = await db.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        logger.warning("_set_job_status: job %s not found in DB", job_id)
        return

    job.status = status
    job.message = message
    job.updated_at = datetime.now(timezone.utc)
    db.add(job)

    if status == JobStatus.COMPLETED and results:
        existing = await db.execute(
            select(ForensicResult).where(ForensicResult.job_id == job_id)
        )
        forensic = existing.scalar_one_or_none()
        if not forensic:
            forensic = ForensicResult(
                job_id=job_id,
                verdict=results.get("verdict", "UNCERTAIN"),
                fake_probability=float(results.get("fake_probability", 0.5)),
                authenticity_score=float(results.get("authenticity_score", 50.0)),
                confidence=float(results.get("confidence", 50.0)),
                raw_results=results,
            )
            db.add(forensic)

    await db.commit()


# ---------------------------------------------------------------------------
# Main pipeline — runs as a background task
# ---------------------------------------------------------------------------

async def run_ads_pipeline(job_id: str, audio_path: str) -> None:
    """
    Full ADS pipeline as a FastAPI BackgroundTask.
    Reads the audio file, runs all 7 detection branches, persists results to Neon DB.
    """
    from apps.api.db.database import async_session
    async with async_session() as db:
        await _set_job_status(db, job_id, JobStatus.PROCESSING, "Initialising ADS pipeline...")

        try:
            await _set_job_status(db, job_id, JobStatus.PROCESSING, "Loading audio file...")
            from pipeline_ads import ADSPipeline  # noqa: E402 — engine path set above

            pipeline = ADSPipeline(device="cpu")

            await _set_job_status(db, job_id, JobStatus.PROCESSING, "Running 7 detection branches...")
            detection_result = pipeline.predict(audio_path)

            results_payload = detection_result.to_dict()

            await _set_job_status(
                db, job_id, JobStatus.COMPLETED,
                "Audio analysis completed successfully.",
                results=results_payload,
            )
            logger.info("ADS Job %s completed: verdict=%s", job_id, detection_result.verdict)

        except Exception as exc:
            logger.error("ADS Job %s failed: %s", job_id, exc)
            logger.error(traceback.format_exc())
            await _set_job_status(
                db, job_id, JobStatus.FAILED,
                f"Audio analysis failed: {exc}",
            )
