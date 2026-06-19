"""
Pydantic schemas for the Image Detection API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: float = Field(..., description="Left edge (normalised 0-1)")
    y: float = Field(..., description="Top edge (normalised 0-1)")
    width: float = Field(..., description="Box width (normalised 0-1)")
    height: float = Field(..., description="Box height (normalised 0-1)")
    label: Optional[str] = None


class BranchSignals(BaseModel):
    spatial_signal: float = Field(..., description="Mean activation from CLIP spatial branch")
    frequency_signal: float = Field(..., description="Mean activation from EfficientNet frequency branch")
    discrepancy_signal: float = Field(..., description="Mean activation from discrepancy branch")
    noise_signal: float = Field(..., description="Mean activation from noise residual branch")
    fingerprint_signal: float = Field(..., description="Mean activation from fingerprint branch")


class ImageAnalysisResponse(BaseModel):
    job_id: str = Field(..., description="The unique job identifier for this analysis")
    status: str = Field(..., description="'success' or 'error'")
    verdict: str = Field(..., description="'FAKE', 'REAL', or 'UNCERTAIN'")
    fake_probability: float = Field(..., description="Raw fake probability [0, 1]")
    authenticity_score: float = Field(..., description="Authenticity percentage [0, 100]; = (1 - fake_probability) * 100")
    confidence: float = Field(..., description="Model confidence percentage [0, 100]")
    ood_score: float = Field(..., description="Out-of-Distribution score; high = novel/unseen generator")
    bounding_boxes: List[BoundingBox] = Field(default_factory=list, description="Localised manipulation regions")
    detected_artifacts: List[str] = Field(default_factory=list, description="Human-readable list of detected artifact types")
    forensic_report: str = Field(..., description="Full natural language forensic narrative from ExplainabilityEngine")
    branch_signals: BranchSignals = Field(..., description="Per-branch activation signals")
    processing_time_s: float = Field(..., description="Wall-clock inference time in seconds")
    file_name: Optional[str] = Field(None, description="Original uploaded file name")


class ImageAnalysisError(BaseModel):
    status: str = "error"
    detail: str
