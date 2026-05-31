"""
VDS Pipeline Orchestrator Service.
"""

import json
import logging
import sys
import traceback
from pathlib import Path

from apps.api.core.config import settings

logger = logging.getLogger(__name__)

# Add engine paths to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent.parent
video_engine_dir = root_dir / "engines" / "video-engine"
fusion_engine_dir = root_dir / "engines" / "fusion-engine"

if str(video_engine_dir) not in sys.path:
    sys.path.insert(0, str(video_engine_dir))
if str(fusion_engine_dir) not in sys.path:
    sys.path.insert(0, str(fusion_engine_dir))

# Now we can import the VDS components
from preprocessing.video_ingestor import VideoIngestor
from preprocessing.scene_detector import SceneDetector
from preprocessing.face_tracker import FaceTracker
from preprocessing.landmark_tracer import LandmarkTracer

from feature_extraction.frame_visual_analyzer import FrameVisualAnalyzer
from feature_extraction.temporal_consistency_analyzer import TemporalConsistencyAnalyzer
from feature_extraction.temporal_motion_analyzer import TemporalMotionAnalyzer
from feature_extraction.temporal_sync_analyzer import TemporalSyncAnalyzer
from feature_extraction.temporal_rppg_analyzer import TemporalRPPGAnalyzer
from feature_extraction.temporal_continuity_analyzer import TemporalContinuityAnalyzer
from feature_extraction.temporal_semantic_analyzer import TemporalSemanticAnalyzer

from core.multimodal_fusion_engine import CrossModalFusionEngine
from localization.video_localizer import VideoLocalizer
from explainability.explainability_engine import ExplainabilityEngine

# In-memory job store
# Format: {"job_id": {"status": "PENDING|PROCESSING|COMPLETED|FAILED", "message": "...", "results": dict}}
JOB_STORE = {}


def run_vds_pipeline(job_id: str, video_path: str):
    """
    Executes the entire Layer 1-5 pipeline asynchronously.
    """
    JOB_STORE[job_id] = {"status": "PROCESSING", "message": "Initializing VDS pipeline..."}
    
    try:
        output_dir = settings.ARTIFACTS_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        device = "cpu"
        
        # --- Layer 1: Preprocessing ---
        JOB_STORE[job_id]["message"] = "Layer 1: Preprocessing video..."
        ingestor = VideoIngestor(target_fps=8.0)
        frames, metadata = ingestor.ingest(Path(video_path))
        
        tracker = FaceTracker()
        tracker.track_faces(frames)
        
        tracer = LandmarkTracer()
        landmarks = tracer.trace_landmarks(frames)
        
        detector = SceneDetector()
        scenes = detector.detect_scenes(frames)
        
        # We need the VideoPreprocessingResult class
        from preprocessing.result_types import VideoPreprocessingResult
        layer1 = VideoPreprocessingResult(
            metadata=metadata,
            frames=frames,
            scenes=scenes,
            landmarks=landmarks,
            unique_track_ids=list(landmarks.keys())
        )
        
        # --- Layer 2: Feature Extraction ---
        JOB_STORE[job_id]["message"] = "Layer 2: Extracting spatial-temporal features..."
        branch_a = FrameVisualAnalyzer(device=device).analyze(layer1)
        branch_b = TemporalConsistencyAnalyzer().analyze(layer1, branch_a)
        branch_c = TemporalMotionAnalyzer(device=device).analyze(layer1)
        branch_d = TemporalSyncAnalyzer(device=device).analyze(layer1)
        branch_e = TemporalRPPGAnalyzer(device=device).analyze(layer1)
        branch_f = TemporalContinuityAnalyzer(device=device).analyze(layer1)
        branch_g = TemporalSemanticAnalyzer(device=device).analyze(layer1)
        
        # --- Layer 3: Cross-Modal Fusion ---
        JOB_STORE[job_id]["message"] = "Layer 3: Cross-Modal Fusion Engine analyzing modalities..."
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
        JOB_STORE[job_id]["message"] = "Layer 4: Localizing deepfake artifacts..."
        localizer = VideoLocalizer()
        localization_result = localizer.localize(
            preprocessing_result=layer1,
            fusion_result=fusion_result,
            branch_a_result=branch_a,
            branch_d_result=branch_d,
        )
        
        # --- Layer 5: Explainability ---
        JOB_STORE[job_id]["message"] = "Layer 5: Generating visual forensic reports..."
        explain_engine = ExplainabilityEngine(output_dir=output_dir)
        explain_result = explain_engine.generate_explanation(
            preprocessing_result=layer1,
            branch_e_result=branch_e,
            fusion_result=fusion_result,
            localization_result=localization_result,
        )
        
        # Read the JSON payload generated by the report generator
        json_path = output_dir / "forensic_summary.json"
        results_payload = {}
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                results_payload = json.load(f)
                
        # Update job store with SUCCESS
        JOB_STORE[job_id] = {
            "status": "COMPLETED",
            "message": "Analysis completed successfully.",
            "results": results_payload
        }
        logger.info(f"Job {job_id} completed successfully.")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        logger.error(traceback.format_exc())
        JOB_STORE[job_id] = {
            "status": "FAILED",
            "message": f"An error occurred during processing: {str(e)}",
            "results": None
        }
