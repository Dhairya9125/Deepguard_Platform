"""
Video Analysis API endpoints.
"""

import shutil
import uuid
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from apps.api.core.config import settings
from apps.api.schemas.jobs import JobCreateResponse, JobStatusResponse
from apps.api.services.vds_pipeline import JOB_STORE, run_vds_pipeline

router = APIRouter()


@router.post("/analyze/video", response_model=JobCreateResponse, status_code=202)
async def analyze_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a video file to begin deepfake forensic analysis.
    Returns a job_id which can be used to poll for the analysis status and results.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Secure a path in the uploads directory
    safe_filename = f"{job_id}_{file.filename}"
    file_path = settings.UPLOAD_DIR / safe_filename
    
    # Save the file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Register the job
    JOB_STORE[job_id] = {"status": "PENDING", "message": "Job queued for processing.", "results": None}
    
    # Kick off background pipeline
    background_tasks.add_task(run_vds_pipeline, job_id, str(file_path))
    
    return JobCreateResponse(
        job_id=job_id,
        message="Video uploaded successfully. Analysis is running in the background."
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Retrieve the status and results of a video analysis job.
    """
    job = JOB_STORE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found.")
        
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        message=job.get("message"),
        results=job.get("results")
    )
