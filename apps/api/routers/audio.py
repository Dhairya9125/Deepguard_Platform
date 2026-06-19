"""
Audio Analysis API endpoints.

Integrates with the Audio Detection Subsystem (ADS) pipeline at
engines/audio-engine/pipeline_ads.py.

Flow:
  POST /api/v1/analyze/audio   — Upload audio → start background job → return job_id
  GET  /api/v1/jobs/{job_id}   — Poll for results (shared with video router)

Audio files are processed asynchronously using FastAPI BackgroundTasks.
Job state + results are persisted to Neon DB.
"""
import shutil
import uuid
from pathlib import Path
from typing import Set

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.core.config import settings
from apps.api.db.database import get_session
from apps.api.db.models import AnalysisJob, JobStatus, User
from apps.api.routers.auth import get_current_user
from apps.api.schemas.jobs import JobCreateResponse
from apps.api.services.ads_pipeline import run_ads_pipeline

router = APIRouter()

ALLOWED_AUDIO_TYPES: Set[str] = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/flac",
    "audio/x-flac",
    "audio/ogg",
    "audio/mp4",
    "audio/m4a",
    "audio/aac",
    "audio/opus",
    "application/octet-stream",  # browsers sometimes use this for audio
}
ALLOWED_EXTENSIONS: Set[str] = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
MAX_AUDIO_MB = 100


@router.post(
    "/analyze/audio",
    response_model=JobCreateResponse,
    status_code=202,
    summary="Analyse an audio clip for AI voice synthesis or manipulation",
    description=(
        "Upload a WAV, MP3, FLAC, OGG, or M4A audio file. "
        "The Audio Detection Subsystem (ADS) runs 7 forensic branches "
        "(Spectral CNN, Temporal Splice, Voice Biometrics, WavLM, XLS-R, "
        "Diffusion Detector, Adversarial Detector) and returns a job ID "
        "which you can poll at GET /api/v1/jobs/{job_id} for the full report."
    ),
)
async def analyze_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> JobCreateResponse:
    """Upload an audio file and start ADS analysis in the background."""

    # ── Validate filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file extension '{suffix}'. "
                   f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # ── Validate size
    content = await file.read()
    if len(content) > MAX_AUDIO_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_AUDIO_MB} MB limit.",
        )

    # ── Save to uploads directory
    job_id = str(uuid.uuid4())
    safe_filename = f"{job_id}_{file.filename}"
    file_path = settings.UPLOAD_DIR / safe_filename
    file_path.write_bytes(content)

    # ── Create PENDING job row in Neon DB
    job = AnalysisJob(
        id=job_id,
        user_id=current_user.id,
        modality="audio",
        status=JobStatus.PENDING,
        message="Audio job queued for processing.",
        file_name=file.filename,
    )
    db.add(job)
    await db.commit()

    # ── Queue background task
    background_tasks.add_task(run_ads_pipeline, job_id, str(file_path))

    return JobCreateResponse(
        job_id=job_id,
        message="Audio uploaded successfully. Analysis is running in the background. "
                f"Poll GET /api/v1/jobs/{job_id} for results.",
    )
