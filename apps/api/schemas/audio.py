"""
Pydantic schemas for the Audio Detection API (ADS).
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AudioBranchSignal(BaseModel):
    branch_name: str
    fake_score: float = Field(..., description="Fake probability from this branch [0, 1]")
    confidence: float = Field(..., description="Branch confidence [0, 1]")
    error: Optional[str] = Field(None, description="Error message if branch failed")


class ManipulationSegment(BaseModel):
    start_s: float = Field(..., description="Segment start time in seconds")
    end_s: float   = Field(..., description="Segment end time in seconds")
    fake_score: float = Field(..., description="Fake score for this segment [0, 1]")


class AudioAnalysisResponse(BaseModel):
    status: str        = Field(..., description="'success' or 'error'")
    verdict: str       = Field(..., description="'FAKE', 'REAL', or 'UNCERTAIN'")
    fake_probability: float     = Field(..., description="Raw fake probability [0, 1]")
    authenticity_score: float   = Field(..., description="Authenticity percentage [0, 100]")
    confidence: float           = Field(..., description="Model confidence percentage [0, 100]")
    duration_s: float           = Field(..., description="Audio clip duration in seconds")
    sample_rate: int            = Field(..., description="Audio sample rate in Hz")
    branch_signals: List[AudioBranchSignal] = Field(default_factory=list)
    manipulation_segments: List[ManipulationSegment] = Field(
        default_factory=list,
        description="Time ranges flagged as manipulated",
    )
    forensic_report: str        = Field(..., description="Full forensic narrative")
    processing_time_s: float    = Field(..., description="Wall-clock inference time in seconds")
    file_name: Optional[str]    = Field(None, description="Original uploaded filename")


class AudioAnalysisError(BaseModel):
    status: str = "error"
    detail: str
