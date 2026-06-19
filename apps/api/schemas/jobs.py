"""
Pydantic schemas for job-related API requests and responses.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Existing schemas (kept for backward compatibility with audio / video routers)
# ---------------------------------------------------------------------------

class JobCreateResponse(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the processing job.")
    message: str = Field(..., description="Status message indicating job was queued.")


class JobStatusResponse(BaseModel):
    job_id: str
    status: str = Field(..., description="Current status: PENDING, PROCESSING, COMPLETED, FAILED")
    message: Optional[str] = None
    modality: Optional[str] = None
    results: Optional[Dict[str, Any]] = Field(
        None,
        description="The parsed forensic_summary.json payload if completed.",
    )


# ---------------------------------------------------------------------------
# Job list schemas (used by GET /jobs)
# ---------------------------------------------------------------------------

class JobListItem(BaseModel):
    """A compact summary of a single analysis job, enriched with ForensicResult data."""

    job_id: str = Field(..., description="Unique identifier for the job.")
    modality: str = Field(..., description="'image', 'video', 'audio', or 'fusion'.")
    status: str = Field(..., description="PENDING | PROCESSING | COMPLETED | FAILED")
    file_name: Optional[str] = Field(None, description="Original uploaded filename.")
    created_at: str = Field(..., description="ISO-8601 UTC timestamp of job creation.")
    fake_probability: Optional[float] = Field(
        None, description="Probability [0,1] that the media is fake. None if not complete."
    )
    verdict: Optional[str] = Field(
        None, description="'REAL', 'FAKE', or 'UNCERTAIN'. None if not complete."
    )


class JobListResponse(BaseModel):
    """Paginated list of analysis jobs."""

    items: List[JobListItem]
    total: int = Field(..., description="Total number of jobs matching the filter.")
    page: int = Field(..., description="Current page number (1-indexed).")
    pages: int = Field(..., description="Total number of pages.")


# ---------------------------------------------------------------------------
# Platform stats schema (used by GET /stats)
# ---------------------------------------------------------------------------

class PlatformStats(BaseModel):
    """Aggregate statistics across all analysis jobs."""

    total_jobs: int = Field(..., description="Total number of jobs ever created.")
    total_fake: int = Field(..., description="Jobs whose ForensicResult verdict is 'FAKE'.")
    total_real: int = Field(..., description="Jobs whose ForensicResult verdict is 'REAL'.")
    total_pending: int = Field(
        ..., description="Jobs currently in PENDING or PROCESSING state."
    )
    by_modality: Dict[str, int] = Field(
        ...,
        description="Job counts broken down by modality key (image, audio, video, fusion).",
    )
    recent_jobs: List[JobListItem] = Field(
        ..., description="The 5 most recently created jobs."
    )
