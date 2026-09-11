"""
Pydantic schemas for API requests and responses.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class JobCreateResponse(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the processing job.")
    message: str = Field(..., description="Status message indicating job was queued.")


class JobStatusResponse(BaseModel):
    job_id: str
    status: str = Field(..., description="Current status: PENDING, PROCESSING, COMPLETED, FAILED")
    message: Optional[str] = None
    results: Optional[Dict[str, Any]] = Field(None, description="The parsed forensic_summary.json payload if completed.")
