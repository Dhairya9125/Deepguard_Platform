"""
Jobs router — cross-modality job listing and platform statistics.

Endpoints
---------
GET /jobs  — paginated list of all AnalysisJob rows, joined with ForensicResult
GET /stats — aggregate counts and recent activity for the dashboard
"""

import math
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import FileResponse
import mimetypes
from sqlalchemy import func, outerjoin, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.db.database import get_session
from apps.api.core.config import settings
from apps.api.db.models import AnalysisJob, ForensicResult, JobStatus, User
from apps.api.routers.auth import get_current_user
from apps.api.schemas.jobs import JobListItem, JobListResponse, PlatformStats, JobStatusResponse

router = APIRouter()


# ---------------------------------------------------------------------------
# Helper: convert an (AnalysisJob, ForensicResult | None) row to JobListItem
# ---------------------------------------------------------------------------

def _row_to_item(job: AnalysisJob, forensic: Optional[ForensicResult]) -> JobListItem:
    return JobListItem(
        job_id=job.id,
        modality=job.modality,
        status=job.status,
        file_name=job.file_name,
        created_at=job.created_at.isoformat(),
        fake_probability=forensic.fake_probability if forensic else None,
        verdict=forensic.verdict if forensic else None,
    )


# ---------------------------------------------------------------------------
# GET /jobs
# ---------------------------------------------------------------------------

@router.get(
    "/jobs",
    response_model=JobListResponse,
    summary="List all analysis jobs",
    description=(
        "Returns a paginated list of AnalysisJob rows ordered by creation time (newest first). "
        "Optionally filter by modality and/or status. "
        "Each item includes the ForensicResult verdict and fake_probability when available."
    ),
)
async def list_jobs(
    page: int = Query(1, ge=1, description="Page number (1-indexed)."),
    limit: int = Query(20, ge=1, le=200, description="Items per page."),
    modality: Optional[str] = Query(None, description="Filter by modality: image | video | audio | fusion"),
    status: Optional[str] = Query(None, description="Filter by status: PENDING | PROCESSING | COMPLETED | FAILED"),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> JobListResponse:
    """List all analysis jobs with optional filters and pagination."""

    # ── Build filter conditions ──────────────────────────────────────────────
    filters = [AnalysisJob.user_id == current_user.id]
    if modality:
        filters.append(AnalysisJob.modality == modality)
    if status:
        filters.append(AnalysisJob.status == status)

    # ── Count total matching rows ────────────────────────────────────────────
    count_stmt = select(func.count()).select_from(AnalysisJob)
    if filters:
        count_stmt = count_stmt.where(*filters)
    total_result = await db.execute(count_stmt)
    total: int = total_result.scalar_one()

    # ── Fetch page of jobs with left-joined ForensicResult ───────────────────
    offset = (page - 1) * limit

    stmt = (
        select(AnalysisJob, ForensicResult)
        .select_from(
            outerjoin(AnalysisJob, ForensicResult, AnalysisJob.id == ForensicResult.job_id)
        )
        .order_by(AnalysisJob.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    if filters:
        stmt = stmt.where(*filters)

    rows = await db.execute(stmt)
    pairs = rows.all()  # list of (AnalysisJob, ForensicResult | None)

    items: List[JobListItem] = [_row_to_item(job, forensic) for job, forensic in pairs]
    pages = math.ceil(total / limit) if total > 0 else 1

    return JobListResponse(items=items, total=total, page=page, pages=pages)


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}
# ---------------------------------------------------------------------------

@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Get details for a specific analysis job",
)
async def get_job(
    job_id: str,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> JobStatusResponse:
    """Fetch a specific job and its forensic results."""
    
    # 1. Fetch the AnalysisJob
    job_result = await db.execute(
        select(AnalysisJob).where(AnalysisJob.id == job_id).where(AnalysisJob.user_id == current_user.id)
    )
    job = job_result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or access denied")
        
    # 2. Fetch ForensicResult if COMPLETED
    results_payload = None
    if job.status == JobStatus.COMPLETED:
        forensic_result = await db.execute(
            select(ForensicResult).where(ForensicResult.job_id == job_id)
        )
        forensic = forensic_result.scalar_one_or_none()
        if forensic and forensic.raw_results:
            results_payload = forensic.raw_results
            
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        message=job.message,
        modality=job.modality,
        results=results_payload
    )


@router.get(
    "/jobs/{job_id}/media",
    summary="Get the media file associated with a job",
)
async def get_job_media(
    job_id: str,
    db: AsyncSession = Depends(get_session),
    # Optional: we could require current_user, but we might want this to be public or pass a token in query string if it's used in <img> tags.
    # For now, let's keep it protected and use fetchWithAuth on frontend, BUT <img> tag doesn't support headers easily.
    # We will pass token via query parameter or cookie for <img> tag.
    token: Optional[str] = Query(None, description="Auth token if not passing Authorization header"),
    current_user: User = Depends(get_current_user),
):
    """Fetch the media file uploaded for a job."""
    job_result = await db.execute(
        select(AnalysisJob).where(AnalysisJob.id == job_id).where(AnalysisJob.user_id == current_user.id)
    )
    job = job_result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or access denied")
        
    safe_filename = f"{job_id}_{job.file_name}"
    file_path = settings.UPLOAD_DIR / safe_filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Media file not found on server")
        
    media_type, _ = mimetypes.guess_type(str(file_path))
    return FileResponse(str(file_path), media_type=media_type or "application/octet-stream")



# ---------------------------------------------------------------------------
# GET /stats
# ---------------------------------------------------------------------------

@router.get(
    "/stats",
    response_model=PlatformStats,
    summary="Platform-wide deepfake detection statistics",
    description=(
        "Returns aggregate counts (total jobs, fake/real verdicts, pending jobs), "
        "a per-modality breakdown, and the 5 most recently created jobs."
    ),
)
async def get_stats(
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> PlatformStats:
    """Compute and return platform-wide statistics."""

    # ── Total jobs ────────────────────────────────────────────────────────────
    total_jobs_result = await db.execute(
        select(func.count())
        .select_from(AnalysisJob)
        .where(AnalysisJob.user_id == current_user.id)
    )
    total_jobs: int = total_jobs_result.scalar_one()

    # ── Pending + Processing count ────────────────────────────────────────────
    pending_result = await db.execute(
        select(func.count())
        .select_from(AnalysisJob)
        .where(AnalysisJob.user_id == current_user.id)
        .where(AnalysisJob.status.in_([JobStatus.PENDING, JobStatus.PROCESSING]))
    )
    total_pending: int = pending_result.scalar_one()

    # ── FAKE verdict count (from ForensicResult) ──────────────────────────────
    fake_result = await db.execute(
        select(func.count())
        .select_from(ForensicResult)
        .join(AnalysisJob, ForensicResult.job_id == AnalysisJob.id)
        .where(AnalysisJob.user_id == current_user.id)
        .where(ForensicResult.verdict == "FAKE")
    )
    total_fake: int = fake_result.scalar_one()

    # ── REAL verdict count ────────────────────────────────────────────────────
    real_result = await db.execute(
        select(func.count())
        .select_from(ForensicResult)
        .join(AnalysisJob, ForensicResult.job_id == AnalysisJob.id)
        .where(AnalysisJob.user_id == current_user.id)
        .where(ForensicResult.verdict == "REAL")
    )
    total_real: int = real_result.scalar_one()

    # ── Per-modality job counts ───────────────────────────────────────────────
    modality_rows = await db.execute(
        select(AnalysisJob.modality, func.count().label("cnt"))
        .where(AnalysisJob.user_id == current_user.id)
        .group_by(AnalysisJob.modality)
    )
    by_modality: dict = {"image": 0, "audio": 0, "video": 0, "fusion": 0}
    for row in modality_rows.all():
        by_modality[row.modality] = row.cnt

    # ── Last 5 jobs (left-joined with ForensicResult) ─────────────────────────
    recent_stmt = (
        select(AnalysisJob, ForensicResult)
        .select_from(
            outerjoin(AnalysisJob, ForensicResult, AnalysisJob.id == ForensicResult.job_id)
        )
        .where(AnalysisJob.user_id == current_user.id)
        .order_by(AnalysisJob.created_at.desc())
        .limit(5)
    )
    recent_rows = await db.execute(recent_stmt)
    recent_jobs: List[JobListItem] = [
        _row_to_item(job, forensic) for job, forensic in recent_rows.all()
    ]

    return PlatformStats(
        total_jobs=total_jobs,
        total_fake=total_fake,
        total_real=total_real,
        total_pending=total_pending,
        by_modality=by_modality,
        recent_jobs=recent_jobs,
    )
