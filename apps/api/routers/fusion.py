"""
Fusion Analysis API endpoints.

POST /analyze/fusion — upload an audio file AND a video file, run both the
ADS and VDS pipelines in parallel in the background, then merge their
ForensicResult payloads into a single 'fusion' verdict.

Flow
----
1. Validate and save both uploaded files.
2. Create a single AnalysisJob with modality='fusion' (status PENDING).
3. Schedule run_fusion_pipeline as a BackgroundTask.
4. Return job_id immediately (202 Accepted).
5. Background task:
   a. Runs run_ads_pipeline logic (audio) and run_vds_pipeline logic (video)
      sequentially inside an asyncio gather so they share one event-loop turn.
   b. Merges results — averages fake_probability, concatenates forensic signals.
   c. Writes a single ForensicResult for the fusion job.
"""

import asyncio
import logging
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Set

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from apps.api.core.config import settings
from apps.api.db.database import get_session
from apps.api.db.models import AnalysisJob, ForensicResult, JobStatus, User
from apps.api.routers.auth import get_current_user
from apps.api.schemas.jobs import JobCreateResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Allowed file types
# ---------------------------------------------------------------------------

ALLOWED_AUDIO_EXTENSIONS: Set[str] = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
ALLOWED_VIDEO_EXTENSIONS: Set[str] = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
MAX_FILE_MB = 500


# ---------------------------------------------------------------------------
# DB helper (duplicated locally to avoid coupling to pipeline internals)
# ---------------------------------------------------------------------------

async def _set_job_status(
    db: AsyncSession,
    job_id: str,
    status: str,
    message: str,
    results: Optional[dict] = None,
) -> None:
    """Update the AnalysisJob row; create a ForensicResult on COMPLETED."""
    result = await db.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        logger.warning("_set_job_status: fusion job %s not found in DB", job_id)
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
# Individual sub-pipeline helpers (run inside asyncio.gather)
# ---------------------------------------------------------------------------

async def _run_audio_sub_pipeline(audio_path: str) -> dict:
    """
    Run the ADS pipeline for the audio component of a fusion job.
    Returns a raw-results dict (or a minimal error dict on failure).
    """
    # Inject engine path (idempotent)
    import sys
    _root = Path(__file__).resolve().parent.parent.parent.parent
    _audio_engine = _root / "engines" / "audio-engine"
    if str(_audio_engine) not in sys.path:
        sys.path.insert(0, str(_audio_engine))

    try:
        from pipeline_ads import ADSPipeline  # noqa: E402
        pipeline = ADSPipeline(device="cpu")
        detection_result = pipeline.predict(audio_path)
        return detection_result.to_dict()
    except Exception as exc:
        logger.error("Fusion audio sub-pipeline failed: %s", exc)
        logger.error(traceback.format_exc())
        return {
            "verdict": "UNCERTAIN",
            "fake_probability": 0.5,
            "authenticity_score": 50.0,
            "confidence": 0.0,
            "error": str(exc),
        }


async def _run_video_sub_pipeline(job_id: str, video_path: str) -> dict:
    """
    Run the VDS pipeline for the video component of a fusion job.
    Returns a raw-results dict (or a minimal error dict on failure).
    """
    import json
    import sys
    _root = Path(__file__).resolve().parent.parent.parent.parent
    _video_engine = _root / "engines" / "video-engine"
    _fusion_engine = _root / "engines" / "fusion-engine"
    for _p in (_video_engine, _fusion_engine):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

    try:
        output_dir = settings.ARTIFACTS_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        device = "cpu"

        from vid_preprocessing.video_ingestor import VideoIngestor
        from vid_preprocessing.scene_detector import SceneDetector
        from vid_preprocessing.face_tracker import FaceTracker
        from vid_preprocessing.landmark_tracer import LandmarkTracer
        from vid_preprocessing.result_types import VideoPreprocessingResult

        ingestor = VideoIngestor(target_fps=8.0)
        frames, metadata = ingestor.ingest(Path(video_path))
        tracker = FaceTracker()
        tracker.track_faces(frames)
        tracer = LandmarkTracer()
        landmarks = tracer.trace_landmarks(frames)
        detector = SceneDetector()
        scenes = detector.detect_scenes(frames)
        layer1 = VideoPreprocessingResult(
            metadata=metadata,
            frames=frames,
            scenes=scenes,
            landmarks=landmarks,
            unique_track_ids=list(landmarks.keys()),
        )

        from vid_feature_extraction.frame_visual_analyzer import FrameVisualAnalyzer
        from vid_feature_extraction.temporal_consistency_analyzer import TemporalConsistencyAnalyzer
        from vid_feature_extraction.temporal_motion_analyzer import TemporalMotionAnalyzer
        from vid_feature_extraction.temporal_sync_analyzer import TemporalSyncAnalyzer
        from vid_feature_extraction.temporal_rppg_analyzer import TemporalRPPGAnalyzer
        from vid_feature_extraction.temporal_continuity_analyzer import TemporalContinuityAnalyzer
        from vid_feature_extraction.temporal_semantic_analyzer import TemporalSemanticAnalyzer

        branch_a = FrameVisualAnalyzer(device=device).analyze(layer1)
        branch_b = TemporalConsistencyAnalyzer().analyze(layer1, branch_a)
        branch_c = TemporalMotionAnalyzer(device=device).analyze(layer1)
        branch_d = TemporalSyncAnalyzer(device=device).analyze(layer1)
        branch_e = TemporalRPPGAnalyzer(device=device).analyze(layer1)
        branch_f = TemporalContinuityAnalyzer(device=device).analyze(layer1)
        branch_g = TemporalSemanticAnalyzer(device=device).analyze(layer1)

        from fusion_core.multimodal_fusion_engine import CrossModalFusionEngine
        fusion_engine = CrossModalFusionEngine(device=device)
        fusion_result = fusion_engine.process_layer2_results(
            branch_a_result=branch_a,
            branch_b_result=branch_b,
            branch_c_result=branch_c,
            branch_d_result=branch_d,
            branch_e_result=branch_e,
            branch_f_result=branch_f,
            branch_g_result=branch_g,
        )

        from vid_localization.video_localizer import VideoLocalizer
        localizer = VideoLocalizer()
        localization_result = localizer.localize(
            preprocessing_result=layer1,
            fusion_result=fusion_result,
            branch_a_result=branch_a,
            branch_d_result=branch_d,
        )

        from vid_explainability.explainability_engine import ExplainabilityEngine
        explain_engine = ExplainabilityEngine(output_dir=output_dir)
        explain_engine.generate_explanation(
            preprocessing_result=layer1,
            branch_e_result=branch_e,
            fusion_result=fusion_result,
            localization_result=localization_result,
        )

        json_path = output_dir / "forensic_summary.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "verdict": "UNCERTAIN",
            "fake_probability": 0.5,
            "authenticity_score": 50.0,
            "confidence": 0.0,
        }

    except Exception as exc:
        logger.error("Fusion video sub-pipeline failed: %s", exc)
        logger.error(traceback.format_exc())
        return {
            "verdict": "UNCERTAIN",
            "fake_probability": 0.5,
            "authenticity_score": 50.0,
            "confidence": 0.0,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------

def _merge_results(audio_res: dict, video_res: dict) -> dict:
    """
    Merge audio and video pipeline results into a single fusion payload.

    Strategy
    --------
    - fake_probability : arithmetic mean of the two probabilities
    - authenticity_score: arithmetic mean
    - confidence       : minimum of the two (weakest link determines certainty)
    - verdict          : re-derived from merged fake_probability using standard thresholds
    - raw_results      : both payloads nested under 'audio' and 'video' keys
    """
    audio_prob = float(audio_res.get("fake_probability", 0.5))
    video_prob = float(video_res.get("fake_probability", 0.5))
    merged_prob = (audio_prob + video_prob) / 2.0

    audio_auth = float(audio_res.get("authenticity_score", 50.0))
    video_auth = float(video_res.get("authenticity_score", 50.0))
    merged_auth = (audio_auth + video_auth) / 2.0

    audio_conf = float(audio_res.get("confidence", 50.0))
    video_conf = float(video_res.get("confidence", 50.0))
    merged_conf = min(audio_conf, video_conf)

    # Derive verdict using the same thresholds as the image router
    if merged_prob >= 0.75:
        verdict = "FAKE"
    elif merged_prob >= 0.45:
        verdict = "UNCERTAIN"
    else:
        verdict = "REAL"

    return {
        "verdict": verdict,
        "fake_probability": round(merged_prob, 6),
        "authenticity_score": round(merged_auth, 2),
        "confidence": round(merged_conf, 2),
        "modality": "fusion",
        "audio": audio_res,
        "video": video_res,
    }


# ---------------------------------------------------------------------------
# Background task: run both pipelines and persist merged result
# ---------------------------------------------------------------------------

async def run_fusion_pipeline(job_id: str, audio_path: str, video_path: str) -> None:
    """
    Orchestrates the fusion analysis as a FastAPI BackgroundTask.

    Runs the ADS (audio) and VDS (video) sub-pipelines concurrently via
    asyncio.gather, then merges and persists the combined ForensicResult.
    """
    from apps.api.db.database import async_session

    async with async_session() as db:
        await _set_job_status(db, job_id, JobStatus.PROCESSING, "Initialising fusion pipeline...")

        try:
            await _set_job_status(
                db, job_id, JobStatus.PROCESSING,
                "Running audio and video sub-pipelines in parallel...",
            )

            audio_res, video_res = await asyncio.gather(
                _run_audio_sub_pipeline(audio_path),
                _run_video_sub_pipeline(job_id, video_path),
            )

            merged = _merge_results(audio_res, video_res)

            await _set_job_status(
                db, job_id, JobStatus.COMPLETED,
                "Fusion analysis completed successfully.",
                results=merged,
            )
            logger.info(
                "Fusion job %s completed: verdict=%s fake_prob=%.4f",
                job_id, merged["verdict"], merged["fake_probability"],
            )

        except Exception as exc:
            logger.error("Fusion job %s failed: %s", job_id, exc)
            logger.error(traceback.format_exc())
            await _set_job_status(
                db, job_id, JobStatus.FAILED,
                f"Fusion analysis failed: {exc}",
            )


# ---------------------------------------------------------------------------
# POST /analyze/fusion endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/analyze/fusion",
    response_model=JobCreateResponse,
    status_code=202,
    summary="Analyse audio + video together for multi-modal deepfake detection",
    description=(
        "Upload one audio file and one video file. "
        "DeepGuard runs the full ADS (7 audio branches) and VDS (5-layer video pipeline) "
        "concurrently, then merges their verdicts into a single fusion result. "
        "Poll GET /api/v1/jobs/{job_id} for the combined report."
    ),
)
async def analyze_fusion(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, OGG, M4A, AAC, OPUS)."),
    video_file: UploadFile = File(..., description="Video file (MP4, AVI, MOV, MKV, WEBM, FLV, WMV)."),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> JobCreateResponse:
    """Upload audio + video and start the fusion deepfake analysis pipeline."""

    # ── Validate audio ────────────────────────────────────────────────────────
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No audio filename provided.")
    audio_suffix = Path(audio_file.filename).suffix.lower()
    if audio_suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio extension '{audio_suffix}'. "
                   f"Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}",
        )

    # ── Validate video ────────────────────────────────────────────────────────
    if not video_file.filename:
        raise HTTPException(status_code=400, detail="No video filename provided.")
    video_suffix = Path(video_file.filename).suffix.lower()
    if video_suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported video extension '{video_suffix}'. "
                   f"Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}",
        )

    # ── Read and size-check both files ────────────────────────────────────────
    audio_content = await audio_file.read()
    if len(audio_content) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Audio file exceeds {MAX_FILE_MB} MB limit.")

    video_content = await video_file.read()
    if len(video_content) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Video file exceeds {MAX_FILE_MB} MB limit.")

    # ── Save uploads ──────────────────────────────────────────────────────────
    job_id = str(uuid.uuid4())

    safe_audio_name = f"{job_id}_audio_{audio_file.filename}"
    audio_path = settings.UPLOAD_DIR / safe_audio_name
    audio_path.write_bytes(audio_content)

    safe_video_name = f"{job_id}_video_{video_file.filename}"
    video_path = settings.UPLOAD_DIR / safe_video_name
    video_path.write_bytes(video_content)

    # ── Create PENDING job row ────────────────────────────────────────────────
    combined_filename = f"{audio_file.filename} + {video_file.filename}"
    job = AnalysisJob(
        id=job_id,
        user_id=current_user.id,
        modality="fusion",
        status=JobStatus.PENDING,
        message="Fusion job queued for processing.",
        file_name=combined_filename,
    )
    db.add(job)
    await db.commit()

    # ── Queue background task ─────────────────────────────────────────────────
    background_tasks.add_task(
        run_fusion_pipeline, job_id, str(audio_path), str(video_path)
    )

    return JobCreateResponse(
        job_id=job_id,
        message=(
            "Audio and video uploaded successfully. "
            "Fusion analysis is running in the background. "
            f"Poll GET /api/v1/jobs/{job_id} for results."
        ),
    )
