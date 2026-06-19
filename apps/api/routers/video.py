"""
Video Analysis API endpoints.

Job status is persisted to PostgreSQL (AnalysisJob table) instead of the
old in-memory JOB_STORE dict, so jobs survive server restarts.
"""

import shutil
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from apps.api.core.config import settings
from apps.api.db.database import get_session
from apps.api.db.models import AnalysisJob, ForensicResult, JobStatus, User
from apps.api.routers.auth import get_current_user
from apps.api.schemas.jobs import JobCreateResponse, JobStatusResponse
from apps.api.services.vds_pipeline import run_vds_pipeline

router = APIRouter()


@router.post("/analyze/video", response_model=JobCreateResponse, status_code=202)
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Upload a video file to begin deepfake forensic analysis.
    Returns a job_id which can be used to poll for the analysis status and results.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    from pathlib import Path
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file extension '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # Secure a path in the uploads directory
    safe_filename = f"{job_id}_{file.filename}"
    file_path = settings.UPLOAD_DIR / safe_filename

    # Save the file to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create a PENDING job row in the database
    job = AnalysisJob(
        id=job_id,
        user_id=current_user.id,
        modality="video",
        status=JobStatus.PENDING,
        message="Job queued for processing.",
        file_name=file.filename,
    )
    db.add(job)
    await db.commit()

    # Kick off background pipeline
    background_tasks.add_task(run_vds_pipeline, job_id, str(file_path))

    return JobCreateResponse(
        job_id=job_id,
        message="Video uploaded successfully. Analysis is running in the background.",
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, db: AsyncSession = Depends(get_session)):
    """
    Retrieve the status and results of a video analysis job.
    Reads from PostgreSQL — survives server restarts unlike the old in-memory dict.
    """
    result = await db.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found.")

    # If completed, load the forensic result payload
    results_payload = None
    if job.status == JobStatus.COMPLETED:
        fr = await db.execute(
            select(ForensicResult).where(ForensicResult.job_id == job_id)
        )
        forensic = fr.scalar_one_or_none()
        if forensic:
            results_payload = forensic.raw_results

    return JobStatusResponse(
        job_id=job_id,
        status=job.status,
        message=job.message,
        modality=job.modality,
        results=results_payload,
    )
