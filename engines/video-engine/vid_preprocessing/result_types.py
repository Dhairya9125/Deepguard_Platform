"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.preprocessing.result_types
Layer   : Layer 1 — Temporal Preprocessing
Task    : Shared dataclasses and result types for all Layer 1 modules

Design contract:
    Every module in Layer 1 reads and/or writes objects defined here.
    Downstream layers (Layer 2, Layer 3) consume `VideoPreprocessingResult`
    as their sole input — no raw video data leaks past this layer.

Data flow:
    VideoIngestor    → VideoMetadata + List[FramePacket]
    FaceTracker      → List[TrackedFace]  (attached to FramePacket)
    LandmarkTracer   → FaceLandmarkResult (per tracked face per frame)
    SceneDetector    → List[SceneBoundary] (scene_id stamped on FramePacket)
    ─────────────────────────────────────────────────────────────────
    All of the above → VideoPreprocessingResult (unified Layer 1 output)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Video-level metadata
# ---------------------------------------------------------------------------

@dataclass
class VideoMetadata:
    """
    Static properties of the source video file, extracted by VideoIngestor.

    Attributes:
        source_path        : Absolute path to the original video file.
        duration_s         : Total duration in seconds.
        native_fps         : Original frames-per-second of the container.
        width              : Frame width in pixels (original resolution).
        height             : Frame height in pixels (original resolution).
        total_frames_native: Total frame count at native FPS.
        total_frames_extracted: Frames actually extracted (after FPS sub-sampling).
        target_fps         : The FPS at which frames were extracted.
        audio_path         : Absolute path to the extracted WAV audio file.
                             None if the video has no audio stream.
        has_audio          : True if an audio stream was found and extracted.
        audio_sample_rate  : Sample rate of the extracted WAV (e.g. 16000).
        audio_channels     : 1 (mono) — always, audio engine requirement.
        codec_video        : Video codec string from the container (e.g. 'h264').
        codec_audio        : Audio codec string from the container (e.g. 'aac').
    """
    source_path:             Path
    duration_s:              float
    native_fps:              float
    width:                   int
    height:                  int
    total_frames_native:     int
    total_frames_extracted:  int
    target_fps:              float
    audio_path:              Optional[Path]
    has_audio:               bool
    audio_sample_rate:       int
    audio_channels:          int = 1
    codec_video:             str = "unknown"
    codec_audio:             str = "unknown"

    def to_dict(self) -> Dict:
        return {
            "source_path":            str(self.source_path),
            "duration_s":             round(self.duration_s, 4),
            "native_fps":             round(self.native_fps, 4),
            "width":                  self.width,
            "height":                 self.height,
            "total_frames_native":    self.total_frames_native,
            "total_frames_extracted": self.total_frames_extracted,
            "target_fps":             self.target_fps,
            "audio_path":             str(self.audio_path) if self.audio_path else None,
            "has_audio":              self.has_audio,
            "audio_sample_rate":      self.audio_sample_rate,
            "audio_channels":         self.audio_channels,
            "codec_video":            self.codec_video,
            "codec_audio":            self.codec_audio,
        }


# ---------------------------------------------------------------------------
# Frame-level data
# ---------------------------------------------------------------------------

@dataclass
class FramePacket:
    """
    All data associated with a single extracted video frame.

    This is the primary unit of work passed between Layer 1 modules.
    The `tracked_faces` and `scene_id` fields are populated in-place
    by FaceTracker and SceneDetector respectively.

    Attributes:
        frame_id        : Zero-based sequential index in the extracted frame list.
        timestamp_ms    : Timestamp of this frame in the source video (milliseconds).
        rgb_array       : (H, W, 3) uint8 numpy array in RGB colour space.
                          None if save_frames=False and the frame has been released
                          after face detection to free memory.
        tracked_faces   : Faces detected and tracked in this frame (populated by
                          FaceTracker). Empty list = no face in frame.
        scene_id        : Integer scene index assigned by SceneDetector.
                          -1 until SceneDetector has run.
        frame_path      : Absolute path to the saved JPEG if save_frames=True,
                          else None.
    """
    frame_id:       int
    timestamp_ms:   float
    rgb_array:      Optional[np.ndarray]     = field(default=None, repr=False)
    tracked_faces:  List["TrackedFace"]      = field(default_factory=list)
    scene_id:       int                      = -1
    frame_path:     Optional[Path]           = None

    def to_dict(self) -> Dict:
        return {
            "frame_id":      self.frame_id,
            "timestamp_ms":  round(self.timestamp_ms, 2),
            "scene_id":      self.scene_id,
            "frame_path":    str(self.frame_path) if self.frame_path else None,
            "n_faces":       len(self.tracked_faces),
            "tracked_faces": [f.to_dict() for f in self.tracked_faces],
        }


# ---------------------------------------------------------------------------
# Face tracking
# ---------------------------------------------------------------------------

@dataclass
class TrackedFace:
    """
    A single face detection result enriched with a persistent DeepSORT track ID.

    Inherits the bounding-box, landmark, and crop data produced by the image
    engine's FaceDetector, and adds temporal continuity fields.

    Attributes:
        track_id        : Stable string ID across frames (e.g. 'track_001').
        track_age       : Number of consecutive frames this track has been active.
        detection_score : RetinaFace / OpenCV DNN confidence in [0, 1].
        bbox            : (x1, y1, x2, y2) in pixel coords of the original frame.
        bbox_5lm        : 5-point landmarks [(x,y), …] from the detector.
                          None if the lightweight OpenCV DNN backend was used.
        aligned_crop    : Aligned face crop as an RGB uint8 array (H, W, 3).
                          Ready for MediaPipe FaceMesh input.
        frame_id        : The frame_id of the FramePacket this face belongs to.
        timestamp_ms    : Timestamp of the parent frame.
        detector_backend: 'dnn' or 'retinaface' — which detector produced this.
    """
    track_id:         str
    track_age:        int
    detection_score:  float
    bbox:             Tuple[int, int, int, int]
    frame_id:         int
    timestamp_ms:     float
    aligned_crop:     Optional[np.ndarray]             = field(default=None, repr=False)
    bbox_5lm:         Optional[List[Tuple[float, float]]] = None
    detector_backend: str                              = "dnn"

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def bbox_as_xywh(self) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.bbox
        return x1, y1, x2 - x1, y2 - y1

    def to_dict(self) -> Dict:
        return {
            "track_id":         self.track_id,
            "track_age":        self.track_age,
            "detection_score":  round(self.detection_score, 6),
            "bbox":             list(self.bbox),
            "bbox_xywh":        list(self.bbox_as_xywh()),
            "bbox_5lm":         [list(pt) for pt in self.bbox_5lm] if self.bbox_5lm else None,
            "frame_id":         self.frame_id,
            "timestamp_ms":     round(self.timestamp_ms, 2),
            "detector_backend": self.detector_backend,
            "crop_shape":       list(self.aligned_crop.shape) if self.aligned_crop is not None else None,
        }


# ---------------------------------------------------------------------------
# Landmark tracing
# ---------------------------------------------------------------------------

@dataclass
class FaceLandmarkResult:
    """
    MediaPipe FaceMesh output for a single tracked face in a single frame.

    Attributes:
        track_id            : Matches TrackedFace.track_id — the temporal identity.
        frame_id            : Matches FramePacket.frame_id.
        timestamp_ms        : Timestamp of the parent frame.
        landmarks_478       : Raw (478, 3) float32 array — (x, y, z) in pixel
                              coords of the aligned crop.
        smoothed_landmarks  : EMA-smoothed (478, 3) array — primary signal for
                              downstream temporal feature extraction.
        jaw_open_ratio      : Scalar [0, 1] — ratio of mouth open height to
                              face height. Forensically useful for lip-sync checks.
        eye_blink_l         : Left eye aspect ratio (EAR) scalar — blink signal.
        eye_blink_r         : Right eye aspect ratio (EAR) scalar.
        face_area_px        : Area of the aligned crop region in the original frame
                              (pixels²). Used to weight landmark confidence.
        detection_success   : False if MediaPipe failed to find a face in the crop.
    """
    track_id:           str
    frame_id:           int
    timestamp_ms:       float
    landmarks_478:      Optional[np.ndarray]    = field(default=None, repr=False)
    smoothed_landmarks: Optional[np.ndarray]    = field(default=None, repr=False)
    jaw_open_ratio:     float                   = 0.0
    eye_blink_l:        float                   = 0.0
    eye_blink_r:        float                   = 0.0
    face_area_px:       int                     = 0
    detection_success:  bool                    = True

    def to_dict(self) -> Dict:
        return {
            "track_id":           self.track_id,
            "frame_id":           self.frame_id,
            "timestamp_ms":       round(self.timestamp_ms, 2),
            "jaw_open_ratio":     round(self.jaw_open_ratio, 6),
            "eye_blink_l":        round(self.eye_blink_l, 6),
            "eye_blink_r":        round(self.eye_blink_r, 6),
            "face_area_px":       self.face_area_px,
            "detection_success":  self.detection_success,
            "landmarks_shape":    list(self.landmarks_478.shape) if self.landmarks_478 is not None else None,
        }


# ---------------------------------------------------------------------------
# Scene segmentation
# ---------------------------------------------------------------------------

@dataclass
class SceneBoundary:
    """
    A single scene segment detected by PySceneDetect.

    Attributes:
        scene_id    : Zero-based scene index.
        start_frame : Index of the first FramePacket in this scene
                      (in the extracted-frame index space, not native).
        end_frame   : Index of the last FramePacket in this scene (inclusive).
        start_ms    : Timestamp (ms) of the scene start.
        end_ms      : Timestamp (ms) of the scene end.
        frame_count : Number of extracted frames in this scene.
    """
    scene_id:    int
    start_frame: int
    end_frame:   int
    start_ms:    float
    end_ms:      float
    frame_count: int

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    def to_dict(self) -> Dict:
        return {
            "scene_id":    self.scene_id,
            "start_frame": self.start_frame,
            "end_frame":   self.end_frame,
            "start_ms":    round(self.start_ms, 2),
            "end_ms":      round(self.end_ms, 2),
            "duration_ms": round(self.duration_ms, 2),
            "frame_count": self.frame_count,
        }


# ---------------------------------------------------------------------------
# Unified Layer 1 output
# ---------------------------------------------------------------------------

@dataclass
class VideoPreprocessingResult:
    """
    The complete output of VDS Layer 1.

    This is the sole input to VDS Layer 2 (Temporal Feature Extraction).
    It contains all derived data from the source video — the raw video file
    is no longer needed after this object is produced.

    Attributes:
        metadata        : Static video properties (resolution, fps, duration …).
        frames          : Ordered list of FramePacket objects — each carries its
                          tracked faces, scene ID, and optional disk path.
        scenes          : Ordered list of SceneBoundary objects.
        landmarks       : Dict mapping track_id → list of FaceLandmarkResult
                          (one entry per frame where that track was active).
        unique_track_ids: Sorted list of all track IDs observed in the video.
        n_scenes        : Number of scenes detected.
        n_frames        : Total frames extracted.
        processing_ok   : False if a non-fatal error occurred during processing
                          (e.g. audio extraction failed). Check `warnings` for detail.
        warnings        : List of non-fatal warning messages from Layer 1 modules.
    """
    metadata:         VideoMetadata
    frames:           List[FramePacket]
    scenes:           List[SceneBoundary]
    landmarks:        Dict[str, List[FaceLandmarkResult]]
    unique_track_ids: List[str]               = field(default_factory=list)
    processing_ok:    bool                    = True
    warnings:         List[str]               = field(default_factory=list)

    @property
    def n_scenes(self) -> int:
        return len(self.scenes)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def n_unique_tracks(self) -> int:
        return len(self.unique_track_ids)

    def frames_for_scene(self, scene_id: int) -> List[FramePacket]:
        """Return all frames belonging to a given scene_id."""
        return [f for f in self.frames if f.scene_id == scene_id]

    def landmarks_for_track(self, track_id: str) -> List[FaceLandmarkResult]:
        """Return the landmark timeline for a specific track."""
        return self.landmarks.get(track_id, [])

    def summary(self) -> Dict:
        """Return a lightweight JSON-serialisable summary (no arrays)."""
        return {
            "source":          str(self.metadata.source_path),
            "duration_s":      round(self.metadata.duration_s, 2),
            "n_frames":        self.n_frames,
            "n_scenes":        self.n_scenes,
            "n_unique_tracks": self.n_unique_tracks,
            "unique_track_ids": self.unique_track_ids,
            "processing_ok":   self.processing_ok,
            "warnings":        self.warnings,
            "has_audio":       self.metadata.has_audio,
            "audio_path":      str(self.metadata.audio_path) if self.metadata.audio_path else None,
        }
