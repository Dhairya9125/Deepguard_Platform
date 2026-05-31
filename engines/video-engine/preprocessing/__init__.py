"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Package : engines.video-engine.preprocessing
Layer   : Layer 1 — Temporal Preprocessing

Public API surface:
    Classes:
        VideoIngestor           — FFmpeg frame & audio extraction
        FaceTracker             — DeepSORT multi-face tracking
        LandmarkTracer          — MediaPipe FaceMesh 478-pt landmarks
        SceneDetector           — PySceneDetect scene boundary detection
        VideoLayer1Pipeline     — Orchestrator (preferred entry point)

    Data types:
        VideoMetadata           — Static video properties
        FramePacket             — Per-frame data container
        TrackedFace             — Face with stable track_id
        FaceLandmarkResult      — 478-pt landmarks + derived metrics
        SceneBoundary           — Scene segment descriptor
        VideoPreprocessingResult— Unified Layer 1 output (→ Layer 2 input)
"""

from .face_tracker            import FaceTracker
from .landmark_tracer         import LandmarkTracer
from .result_types            import (
    FaceLandmarkResult,
    FramePacket,
    SceneBoundary,
    TrackedFace,
    VideoMetadata,
    VideoPreprocessingResult,
)
from .scene_detector          import SceneDetector
from .video_ingestor          import VideoIngestor
from .video_layer1_pipeline   import VideoLayer1Pipeline

__all__ = [
    # Modules
    "VideoIngestor",
    "FaceTracker",
    "LandmarkTracer",
    "SceneDetector",
    "VideoLayer1Pipeline",
    # Data types
    "VideoMetadata",
    "FramePacket",
    "TrackedFace",
    "FaceLandmarkResult",
    "SceneBoundary",
    "VideoPreprocessingResult",
]
