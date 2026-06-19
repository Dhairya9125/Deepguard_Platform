"""
Image Analysis API endpoints.

Integrates with the Image Detection Subsystem (IDS) pipeline located at
engines/image-engine/pipeline_ids.py. The pipeline runs synchronously since
image analysis is fast (< 5 s on CPU). Upload → analyse → return in one request.

Every successful analysis is persisted to PostgreSQL (AnalysisJob + ForensicResult)
so users have a full searchable history of all image checks.
"""

import shutil
import sys
import time
import uuid
import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.core.config import settings
from apps.api.db.database import get_session
from apps.api.db.models import AnalysisJob, ForensicResult, JobStatus, User
from apps.api.routers.auth import get_current_user
from apps.api.schemas.image import (
    BoundingBox,
    BranchSignals,
    ImageAnalysisError,
    ImageAnalysisResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Engine path injection ────────────────────────────────────────────────────
# The image-engine lives outside the apps/ package tree, so we resolve its
# absolute path and add it to sys.path once at import time.
_ENGINE_DIR = (
    Path(__file__).parent.parent.parent.parent / "engines" / "image-engine"
).resolve()

if str(_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_ENGINE_DIR))

# Lazy-loaded pipeline singleton — avoids loading torch at startup if not needed
_pipeline = None


def _get_pipeline():
    """Return (or lazily create) the IDS pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        try:
            from pipeline_ids import IDSPipeline  # noqa: E402 – engine path is set above
            _pipeline = IDSPipeline(device="cpu")
            logger.info("IDSPipeline loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load IDSPipeline: %s", exc, exc_info=True)
            raise RuntimeError(f"Image engine not available: {exc}") from exc
    return _pipeline


# ── Helpers ──────────────────────────────────────────────────────────────────

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}
MAX_IMAGE_MB = 50


def _parse_bboxes(raw_boxes) -> List[BoundingBox]:
    """Convert whatever the localization engine returns to BoundingBox list."""
    result: List[BoundingBox] = []
    if not raw_boxes:
        return result
    for box in raw_boxes:
        try:
            if isinstance(box, dict):
                result.append(BoundingBox(
                    x=float(box.get("x", 0)),
                    y=float(box.get("y", 0)),
                    width=float(box.get("width", box.get("w", 0))),
                    height=float(box.get("height", box.get("h", 0))),
                    label=box.get("label"),
                ))
            elif isinstance(box, (list, tuple)) and len(box) >= 4:
                result.append(BoundingBox(x=float(box[0]), y=float(box[1]),
                                          width=float(box[2]), height=float(box[3])))
        except Exception:
            pass
    return result


def _build_artifact_list(report_text: str, fake_prob: float) -> List[str]:
    """
    Extract bullet-point evidence from the forensic report for display
    in the frontend artifact list.
    """
    artifacts = []
    for line in report_text.split("\n"):
        line = line.strip()
        if line.startswith("- "):
            artifacts.append(line[2:])
    if not artifacts and fake_prob > 0.5:
        artifacts.append("Anomalous patterns detected by fusion engine")
    return artifacts


# ── Endpoint ─────────────────────────────────────────────────────────────────

@router.post(
    "/analyze/image",
    response_model=ImageAnalysisResponse,
    status_code=200,
    summary="Analyze an image for deepfake manipulation",
    description=(
        "Upload a PNG, JPEG, or WEBP image. The Image Detection Subsystem (IDS) "
        "runs five forensic branches (Spatial, Frequency, Discrepancy, Noise, "
        "Fingerprint) through a Transformer fusion engine and returns a verdict "
        "with bounding boxes and a forensic narrative."
    ),
)
async def analyze_image(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Run IDS on an uploaded image and return forensic results."""

    # ── Validate ──────────────────────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    if file.content_type and file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. "
                   f"Allowed: {', '.join(ALLOWED_IMAGE_TYPES)}",
        )

    # ── Save upload ───────────────────────────────────────────────────────────
    file_id = str(uuid.uuid4())
    safe_filename = f"{file_id}_{file.filename}"
    file_path = settings.UPLOAD_DIR / safe_filename

    content = await file.read()
    if len(content) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_IMAGE_MB} MB limit.",
        )

    file_path.write_bytes(content)
    logger.info("Image saved: %s (%d bytes)", file_path, len(content))

    # ── Run IDS pipeline ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        pipeline = _get_pipeline()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    try:
        raw = pipeline.predict(str(file_path))
    except Exception as exc:
        logger.error("IDS pipeline error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")
    finally:
        elapsed = time.perf_counter() - t0

    # ── Handle engine-level errors (e.g. no face detected) ───────────────────
    if "error" in raw:
        raise HTTPException(status_code=422, detail=raw["error"])

    # ── Map results ───────────────────────────────────────────────────────────
    fake_prob: float = float(raw.get("fake_probability", 0.0))
    ood_score: float = float(raw.get("ood_score", 0.0))
    forensic_report: str = raw.get("forensic_report", "")

    # Verdict thresholds
    if fake_prob >= 0.75:
        verdict = "FAKE"
    elif fake_prob >= 0.45:
        verdict = "UNCERTAIN"
    else:
        verdict = "REAL"

    # Confidence: distance from the decision boundary (0.5), mapped to 50-100 %
    confidence = 50.0 + abs(fake_prob - 0.5) * 100.0

    # Branch signals — the pipeline returns mean activations
    branch_signals = BranchSignals(
        spatial_signal=float(raw.get("spatial_signal", 0.0)),
        frequency_signal=float(raw.get("frequency_signal", 0.0)),
        discrepancy_signal=float(raw.get("discrepancy_signal", 0.0)),
        noise_signal=float(raw.get("noise_signal", 0.0)),
        fingerprint_signal=float(raw.get("fingerprint_signal", 0.0)),
    )

    bboxes = _parse_bboxes(raw.get("bounding_boxes", []))
    artifacts = _build_artifact_list(forensic_report, fake_prob)

    response = ImageAnalysisResponse(
        job_id=file_id,
        status="success",
        verdict=verdict,
        fake_probability=round(fake_prob, 6),
        authenticity_score=round((1.0 - fake_prob) * 100.0, 2),
        confidence=round(confidence, 2),
        ood_score=round(ood_score, 6),
        bounding_boxes=bboxes,
        detected_artifacts=artifacts,
        forensic_report=forensic_report,
        branch_signals=branch_signals,
        processing_time_s=round(elapsed, 3),
        file_name=file.filename,
    )

    # ── Persist to PostgreSQL ─────────────────────────────────────────────────
    # Create a completed AnalysisJob + ForensicResult row for history tracking.
    try:
        job_id = file_id  # reuse the UUID already assigned to this upload
        job = AnalysisJob(
            id=job_id,
            user_id=current_user.id,
            modality="image",
            status=JobStatus.COMPLETED,
            message="Analysis completed successfully.",
            file_name=file.filename,
        )
        db.add(job)
        await db.flush()

        forensic = ForensicResult(
            job_id=job_id,
            verdict=verdict,
            fake_probability=round(fake_prob, 6),
            authenticity_score=round((1.0 - fake_prob) * 100.0, 2),
            confidence=round(confidence, 2),
            raw_results=raw,
        )
        db.add(forensic)
        await db.commit()
        logger.info("Image result persisted to DB: job_id=%s verdict=%s", job_id, verdict)
    except Exception as db_exc:
        # Never fail the API response because of a DB write error
        logger.warning("Failed to persist image result to DB: %s", db_exc)
        await db.rollback()

    return response
