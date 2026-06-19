"""
VDS Pipeline Orchestrator Service.

Executes the full video analysis pipeline (Layers 1-5) as a FastAPI background task.
Job state is persisted to PostgreSQL (AnalysisJob table) instead of the fragile
in-memory JOB_STORE dict — so job status survives server restarts.
"""

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from apps.api.core.config import settings
from apps.api.db.models import AnalysisJob, ForensicResult, JobStatus

logger = logging.getLogger(__name__)

# Add engine paths to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent.parent
video_engine_dir = root_dir / "engines" / "video-engine"
fusion_engine_dir = root_dir / "engines" / "fusion-engine"

if str(video_engine_dir) not in sys.path:
    sys.path.insert(0, str(video_engine_dir))
if str(fusion_engine_dir) not in sys.path:
    sys.path.insert(0, str(fusion_engine_dir))


# ---------------------------------------------------------------------------
# DB helpers — called inside background task
# ---------------------------------------------------------------------------

async def _set_job_status(
    db: AsyncSession,
    job_id: str,
    status: str,
    message: str,
    results: dict | None = None,
) -> None:
    """
    Update the AnalysisJob row's status + message in the database.
    If the job is COMPLETED and results are provided, also creates a ForensicResult row.
    """
    result = await db.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        logger.warning("_set_job_status: job %s not found in DB", job_id)
        return

    job.status = status
    job.message = message
    job.updated_at = datetime.now(timezone.utc)
    db.add(job)

    if status == JobStatus.COMPLETED and results:
        # Upsert a ForensicResult row
        existing_result = await db.execute(
            select(ForensicResult).where(ForensicResult.job_id == job_id)
        )
        forensic = existing_result.scalar_one_or_none()
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
# Main pipeline — runs as a background task
# ---------------------------------------------------------------------------

async def run_vds_pipeline(job_id: str, video_path: str) -> None:
    """
    Executes the entire Layer 1-5 VDS pipeline asynchronously.

    This function is registered as a FastAPI BackgroundTask by the video router.
    It manages its own db session so it can write progress updates to PostgreSQL.
    """
    from apps.api.db.database import async_session
    async with async_session() as db:
        await _set_job_status(db, job_id, JobStatus.PROCESSING, "Initializing VDS pipeline...")

        try:
            output_dir = settings.ARTIFACTS_DIR / job_id
            output_dir.mkdir(parents=True, exist_ok=True)

            device = "cpu"

            # --- Layer 1: Preprocessing ---
            await _set_job_status(db, job_id, JobStatus.PROCESSING, "Layer 1: Preprocessing video...")
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

            # --- Layer 2: Feature Extraction ---
            await _set_job_status(db, job_id, JobStatus.PROCESSING, "Layer 2: Extracting spatial-temporal features...")
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

            # --- Layer 3: Cross-Modal Fusion ---
            await _set_job_status(db, job_id, JobStatus.PROCESSING, "Layer 3: Cross-Modal Fusion Engine analyzing modalities...")
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

            # --- Layer 4: Video Localization ---
            await _set_job_status(db, job_id, JobStatus.PROCESSING, "Layer 4: Localizing deepfake artifacts...")
            from vid_localization.video_localizer import VideoLocalizer

            localizer = VideoLocalizer()
            localization_result = localizer.localize(
                preprocessing_result=layer1,
                fusion_result=fusion_result,
                branch_a_result=branch_a,
                branch_d_result=branch_d,
            )

            # --- Layer 5: Explainability ---
            await _set_job_status(db, job_id, JobStatus.PROCESSING, "Layer 5: Generating visual forensic reports...")
            from vid_explainability.explainability_engine import ExplainabilityEngine

            explain_engine = ExplainabilityEngine(output_dir=output_dir)
            explain_engine.generate_explanation(
                preprocessing_result=layer1,
                branch_e_result=branch_e,
                fusion_result=fusion_result,
                localization_result=localization_result,
            )

            # Read the JSON payload generated by the report generator
            json_path = output_dir / "forensic_summary.json"
            results_payload: dict = {}
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    results_payload = json.load(f)

            # Persist SUCCESS + ForensicResult row
            await _set_job_status(
                db, job_id, JobStatus.COMPLETED,
                "Analysis completed successfully.",
                results=results_payload,
            )
            logger.info("Job %s completed successfully.", job_id)

        except Exception as exc:
            logger.error("Job %s failed: %s", job_id, exc)
            logger.error(traceback.format_exc())
            await _set_job_status(
                db, job_id, JobStatus.FAILED,
                f"An error occurred during processing: {exc}",
            )
