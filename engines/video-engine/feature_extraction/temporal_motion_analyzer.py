"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.temporal_motion_analyzer
Layer   : Layer 2 — Temporal Feature Extraction
Branch  : Branch C — Temporal Motion Modeling

Orchestrator for Branch C. Combines 3D CNN model inferences with pure NumPy
geometric and motion signals to determine video authenticity.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .motion_signals import (
    compute_blink_anomalies,
    compute_frame_inconsistency,
    compute_jitter_entropy,
    compute_velocity_variance,
)
from .result_types_l2c import BranchCResult, TrackMotionMetrics
from .slowfast_model import SlowFastMotionModel
from .timesformer_model import TimeSformerMotionModel
from .videomae_model import VideoMAEMotionModel

logger = logging.getLogger(__name__)


class TemporalMotionAnalyzer:
    """
    VDS Layer 2, Branch C — Temporal Motion Modeling.
    
    Processes the sequence of face crops and landmarks for each track,
    evaluating motion continuity, temporal drift, blinking anomalies, and 
    learned motion signatures via TimeSformer, VideoMAE, and SlowFast.
    """
    def __init__(
        self,
        device: str = "cpu",
        timesformer_model_name: str = "facebook/timesformer-base-finetuned-k400",
        videomae_model_name: str = "MCG-NJU/videomae-base",
        timesformer_checkpoint: str | Path | None = None,
        videomae_checkpoint: str | Path | None = None,
        slowfast_checkpoint: str | Path | None = None,
        sequence_length: int = 16,
        thresholds: Dict[str, float] | None = None,
        weights: Dict[str, float] | None = None,
    ) -> None:
        self.device = device
        self.timesformer_model_name = timesformer_model_name
        self.videomae_model_name = videomae_model_name
        self.timesformer_checkpoint = timesformer_checkpoint
        self.videomae_checkpoint = videomae_checkpoint
        self.slowfast_checkpoint = slowfast_checkpoint
        self.sequence_length = sequence_length

        # ── Calibration System ─────────────────────────────────────────────
        self.thresholds = {
            "timesformer_score": 0.60,
            "videomae_score":    0.60,
            "slowfast_score":    0.60,
            "jitter_entropy":    3.50,
            "velocity_variance": 0.08,
            "blink_asymmetry":   0.25,
            "blink_duration_anomaly": 0.50,
            "frame_inconsistency": 0.15,
        }
        if thresholds:
            self.thresholds.update(thresholds)

        self.weights = {
            "timesformer_score": 3.0,
            "videomae_score":    3.0,
            "slowfast_score":    2.0,
            "jitter_entropy":    1.5,
            "velocity_variance": 1.5,
            "blink_asymmetry":   1.0,
            "blink_duration_anomaly": 1.0,
            "frame_inconsistency": 2.0,
        }
        if weights:
            self.weights.update(weights)

        self.max_possible_score = sum(self.weights.values())

        # Lazy models container
        self._models_loaded = False
        self.timesformer: Optional[TimeSformerMotionModel] = None
        self.videomae: Optional[VideoMAEMotionModel] = None
        self.slowfast: Optional[SlowFastMotionModel] = None

        logger.info(
            "TemporalMotionAnalyzer initialized | device=%s | sequence_length=%d",
            device, sequence_length
        )

    def _lazy_load_models(self) -> None:
        """Initialize and load deep learning models when needed."""
        if not self._models_loaded:
            logger.info("Lazily loading Branch C 3D models ...")
            self.timesformer = TimeSformerMotionModel(
                model_name=self.timesformer_model_name,
                checkpoint_path=self.timesformer_checkpoint,
                device=self.device
            )
            self.videomae = VideoMAEMotionModel(
                model_name=self.videomae_model_name,
                checkpoint_path=self.videomae_checkpoint,
                device=self.device
            )
            self.slowfast = SlowFastMotionModel(
                checkpoint_path=self.slowfast_checkpoint,
                device=self.device
            )
            self._models_loaded = True
            logger.info("All Branch C models loaded.")

    def _preprocess_crops(self, crops: List[np.ndarray]) -> torch.Tensor:
        """
        Resize, normalize, and interpolate/pad face crops to shape (1, T, 3, 224, 224).
        """
        T_target = self.sequence_length
        valid_crops = [c for c in crops if c is not None]

        # Handle empty crops
        if not valid_crops:
            # Return dummy zero-tensor
            return torch.zeros((1, T_target, 3, 224, 224), dtype=torch.float32)

        # 1. Resize crops to 224x224 and scale to float [0, 1]
        processed_crops = []
        for crop in valid_crops:
            if crop.shape[:2] != (224, 224):
                resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
            else:
                resized = crop
            # Convert to float CHW
            chw = resized.astype(np.float32) / 255.0
            # Normalize with ImageNet stats
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized = (chw - mean) / std
            # Transpose HWC -> CHW
            chw_transposed = np.transpose(normalized, (2, 0, 1))
            processed_crops.append(chw_transposed)

        # 2. Resample or Pad timeline to match T_target
        N = len(processed_crops)
        if N >= T_target:
            # Sub-sample evenly
            indices = np.linspace(0, N - 1, num=T_target, dtype=int)
            final_crops = [processed_crops[idx] for idx in indices]
        else:
            # Pad by repeating the last frame
            final_crops = list(processed_crops)
            while len(final_crops) < T_target:
                final_crops.append(processed_crops[-1])

        # Stack into shape (T, 3, 224, 224)
        stacked = np.stack(final_crops, axis=0)
        # Add batch dimension: (1, T, 3, 224, 224)
        tensor = torch.from_numpy(stacked).unsqueeze(0)
        return tensor

    def _analyze_track(
        self,
        track_id: str,
        layer1_result,
        crops: List[np.ndarray]
    ) -> TrackMotionMetrics:
        """Evaluate Branch C signals for a single identity track."""
        n_frames = len(crops)
        valid_crops = [c for c in crops if c is not None]
        n_valid = len(valid_crops)

        # Retrieve landmarks
        landmarks_list = layer1_result.landmarks_for_track(track_id)
        
        # Extract coordinates and EAR metrics
        coords = []
        ear_l = []
        ear_r = []
        for lm in landmarks_list:
            if lm.detection_success and lm.landmarks_478 is not None:
                coords.append(lm.smoothed_landmarks)
                ear_l.append(lm.eye_blink_l)
                ear_r.append(lm.eye_blink_r)

        coords_arr = np.array(coords) if coords else None
        ear_l_arr = np.array(ear_l) if ear_l else None
        ear_r_arr = np.array(ear_r) if ear_r else None

        # ── 1. Geometric Motion Features (NumPy) ───────────────────────────
        jitter_entropy = compute_jitter_entropy(coords_arr)
        velocity_variance = compute_velocity_variance(coords_arr)
        
        fps = layer1_result.metadata.target_fps
        blink_asymmetry, blink_dur_anomaly = compute_blink_anomalies(
            ear_l_arr, ear_r_arr, fps
        )

        inconsistency_score = compute_frame_inconsistency(crops)

        # ── 2. Deep Learning Video Classifications ────────────────────────
        self._lazy_load_models()

        # Prepare stacked crop sequence
        input_tensor = self._preprocess_crops(crops).to(torch.device(self.device))

        with torch.no_grad():
            timesformer_prob = float(self.timesformer.predict_probability(input_tensor).cpu().numpy()[0])
            videomae_prob = float(self.videomae.predict_probability(input_tensor).cpu().numpy()[0])
            slowfast_prob = float(self.slowfast.predict_probability(input_tensor).cpu().numpy()[0])

        # ── 3. Weighted Verdict Evaluation ──────────────────────────────────
        signals_fired = []
        weighted_score = 0.0

        # Model scores
        if timesformer_prob >= self.thresholds["timesformer_score"]:
            signals_fired.append("timesformer_score")
            weighted_score += self.weights["timesformer_score"]

        if videomae_prob >= self.thresholds["videomae_score"]:
            signals_fired.append("videomae_score")
            weighted_score += self.weights["videomae_score"]

        if slowfast_prob >= self.thresholds["slowfast_score"]:
            signals_fired.append("slowfast_score")
            weighted_score += self.weights["slowfast_score"]

        # Geometric metrics
        if jitter_entropy >= self.thresholds["jitter_entropy"]:
            signals_fired.append("jitter_entropy")
            weighted_score += self.weights["jitter_entropy"]

        if velocity_variance >= self.thresholds["velocity_variance"]:
            signals_fired.append("velocity_variance")
            weighted_score += self.weights["velocity_variance"]

        if blink_asymmetry >= self.thresholds["blink_asymmetry"]:
            signals_fired.append("blink_asymmetry")
            weighted_score += self.weights["blink_asymmetry"]

        if blink_dur_anomaly >= self.thresholds["blink_duration_anomaly"]:
            signals_fired.append("blink_duration_anomaly")
            weighted_score += self.weights["blink_duration_anomaly"]

        if inconsistency_score >= self.thresholds["frame_inconsistency"]:
            signals_fired.append("frame_inconsistency")
            weighted_score += self.weights["frame_inconsistency"]

        # Calculate final confidence
        confidence = weighted_score / self.max_possible_score

        # Determine verdict
        if confidence >= 0.50:
            verdict = "FAKE"
            risk_level = "HIGH"
        elif confidence < 0.25:
            verdict = "REAL"
            risk_level = "LOW"
        else:
            verdict = "UNCERTAIN"
            risk_level = "MEDIUM"

        return TrackMotionMetrics(
            track_id=track_id,
            n_frames=n_frames,
            n_valid_frames=n_valid,
            timesformer_score=timesformer_prob,
            videomae_score=videomae_prob,
            slowfast_score=slowfast_prob,
            jitter_entropy=jitter_entropy,
            landmark_velocity_variance=velocity_variance,
            blink_asymmetry=blink_asymmetry,
            blink_duration_anomaly=blink_dur_anomaly,
            frame_inconsistency_score=inconsistency_score,
            motion_verdict=verdict,
            motion_confidence=confidence,
            motion_signals_fired=signals_fired,
            motion_risk_level=risk_level,
        )

    def analyze(self, layer1_result) -> BranchCResult:
        """
        Execute Branch C analysis on the Layer 1 Preprocessing results.
        """
        logger.info("VDS Layer 2 Branch C START | tracks=%s", layer1_result.unique_track_ids)
        t_start = time.perf_counter()
        warnings: List[str] = []

        track_metrics: Dict[str, TrackMotionMetrics] = {}

        if not layer1_result.unique_track_ids:
            warnings.append("No tracked faces in Layer 1 preprocessing result.")
            logger.warning("No tracks to process in Branch C.")
            return BranchCResult(processing_warnings=warnings)

        # Map track crops
        track_crops: Dict[str, List[np.ndarray]] = {tid: [] for tid in layer1_result.unique_track_ids}
        for fp in layer1_result.frames:
            for tf in fp.tracked_faces:
                if tf.track_id in track_crops:
                    track_crops[tf.track_id].append(tf.aligned_crop)

        # Process each track
        for tid in layer1_result.unique_track_ids:
            crops = track_crops[tid]
            metrics = self._analyze_track(tid, layer1_result, crops)
            track_metrics[tid] = metrics
            logger.info(
                "  %s  →  timesformer=%.4f  videomae=%.4f  slowfast=%.4f  verdict=%s (conf=%.4f)",
                tid, metrics.timesformer_score, metrics.videomae_score,
                metrics.slowfast_score, metrics.motion_verdict, metrics.motion_confidence
            )

        # Global aggregations
        timesformer_scores = [m.timesformer_score for m in track_metrics.values()]
        videomae_scores = [m.videomae_score for m in track_metrics.values()]
        slowfast_scores = [m.slowfast_score for m in track_metrics.values()]
        inconsistency_scores = [m.frame_inconsistency_score for m in track_metrics.values()]

        fake_tracks = [m for m in track_metrics.values() if m.motion_verdict == "FAKE"]
        real_tracks = [m for m in track_metrics.values() if m.motion_verdict == "REAL"]
        uncertain_tracks = [m for m in track_metrics.values() if m.motion_verdict == "UNCERTAIN"]

        n_fake = len(fake_tracks)
        n_real = len(real_tracks)
        n_uncertain = len(uncertain_tracks)

        # Majority vote verdict
        if n_fake > n_real:
            overall_verdict = "FAKE"
        elif n_real > n_fake:
            overall_verdict = "REAL"
        else:
            overall_verdict = "UNCERTAIN"

        overall_conf = float(np.mean([m.motion_confidence for m in track_metrics.values()]))

        elapsed = time.perf_counter() - t_start
        logger.info(
            "VDS Layer 2 Branch C COMPLETE | elapsed=%.2fs | overall_verdict=%s (conf=%.4f)",
            elapsed, overall_verdict, overall_conf
        )

        return BranchCResult(
            track_metrics=track_metrics,
            overall_verdict=overall_verdict,
            overall_confidence=overall_conf,
            n_fake_tracks=n_fake,
            n_real_tracks=n_real,
            n_uncertain_tracks=n_uncertain,
            video_timesformer_mean=float(np.mean(timesformer_scores)),
            video_videomae_mean=float(np.mean(videomae_scores)),
            video_slowfast_mean=float(np.mean(slowfast_scores)),
            video_inconsistency_mean=float(np.mean(inconsistency_scores)),
            processing_warnings=warnings,
        )

    def unload(self) -> None:
        """Unload deep models to release GPU/CPU memory."""
        if self._models_loaded:
            try:
                del self.timesformer
                del self.videomae
                del self.slowfast
                self.timesformer = None
                self.videomae = None
                self.slowfast = None
                self._models_loaded = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Branch C model weights successfully unloaded.")
            except Exception as e:
                logger.warning("Error unloading Branch C models: %s", e)
