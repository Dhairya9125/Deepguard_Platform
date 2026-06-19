"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.temporal_continuity_analyzer
Layer   : Layer 2 — Temporal Feature Extraction
Branch  : Branch F — Identity Continuity Tracking

Orchestrates Branch F. Processes face crops sequence, bounding box coordinates,
and facial landmarks to analyze identity drift, geometry stability, and face continuity.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .facenet_model import FacenetActiveModel
from .result_types_l2f import BranchFResult, TrackContinuityMetrics
from .continuity_signals import (
    compute_geometry_instability,
    compute_identity_drift,
    compute_trajectory_jumps,
)

logger = logging.getLogger(__name__)


class TemporalContinuityAnalyzer:
    """
    VDS Layer 2, Branch F — Identity Continuity Tracking.
    
    Orchestrates identity representation stability, geometric facial structure CV,
    and physical trajectory checks to detect face-swaps, deepfakes, and continuity errors.
    """
    def __init__(
        self,
        device: str = "cpu",
        facenet_checkpoint: str | Path | None = None,
        thresholds: Dict[str, float] | None = None,
        weights: Dict[str, float] | None = None,
    ) -> None:
        self.device = device
        self.facenet_checkpoint = facenet_checkpoint

        # ── Calibration System ─────────────────────────────────────────────
        self.thresholds = {
            "facenet_drift_max":        0.35,   # Max embedding cosine distance from anchor allowed
            "facenet_drift_var":        0.02,   # Max cosine distance variance allowed
            "geometry_instability":     0.035,  # Max landmark ratios coefficient of variation allowed
            "trajectory_jump":          0.15,   # Max bbox center displacement ratio allowed per frame
            "jump_count":               2.0,    # Max allowed displacement spikes before flagging jump anomaly
        }
        if thresholds:
            self.thresholds.update(thresholds)

        self.weights = {
            "facenet_drift_max":        3.0,    # Embedding representation drift magnitude
            "facenet_drift_var":        2.0,    # Embedding representation volatility
            "geometry_instability":     2.5,    # Facial ratios non-rigid warping
            "trajectory_jump":          2.5,    # Physical trajectory jumps (displacement spikes)
        }
        if weights:
            self.weights.update(weights)

        self.max_possible_score = sum(self.weights.values())

        # Lazy models container
        self._models_loaded = False
        self.facenet: Optional[FacenetActiveModel] = None

        logger.info("TemporalContinuityAnalyzer initialized | device=%s", device)

    def _lazy_load_models(self) -> None:
        if not self._models_loaded:
            logger.info("Lazily loading Branch F identity model ...")
            self.facenet = FacenetActiveModel(
                checkpoint_path=self.facenet_checkpoint,
                device=self.device
            )
            self._models_loaded = True
            logger.info("All Branch F models loaded.")

    def _extract_crops_embeddings(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Process and resize face crop sequence to (T, 3, 160, 160), run Facenet,
        and return embeddings as a (T, 128) numpy array.
        """
        self._lazy_load_models()
        valid_crops = [cv2.resize(c, (160, 160)) for c in crops if c is not None]
        if not valid_crops:
            return np.zeros((0, 128), dtype=np.float32)

        arr = np.stack(valid_crops, axis=0).astype(np.float32) / 255.0  # (T, 160, 160, 3)
        arr = np.transpose(arr, (0, 3, 1, 2))  # (T, 3, 160, 160)
        tensor = torch.from_numpy(arr).to(torch.device(self.device))

        with torch.no_grad():
            embs_tensor = self.facenet(tensor)  # (T, 128)
            embs = embs_tensor.cpu().numpy()
        return embs

    def _analyze_track(
        self,
        track_id: str,
        layer1_result,
        crops: List[np.ndarray],
        bboxes: List[Tuple[int, int, int, int]]
    ) -> TrackContinuityMetrics:
        """Evaluate identity continuity signals for a single identity track."""
        n_frames = len(crops)
        valid_crops = [c for c in crops if c is not None]
        n_valid = len(valid_crops)
        fps = layer1_result.metadata.target_fps
        frame_width = layer1_result.metadata.width

        # Retrieve landmarks
        landmarks_list = layer1_result.landmarks_for_track(track_id)
        coords = []
        for lm in landmarks_list:
            if lm.detection_success and lm.smoothed_landmarks is not None:
                coords.append(lm.smoothed_landmarks)
        coords_arr = np.array(coords) if coords else None

        # ── 1. representation Drift Extraction ─────────────────────────────
        embeddings = self._extract_crops_embeddings(crops)
        drift_max, drift_var = compute_identity_drift(embeddings)

        # ── 2. Rigid Geometry ratios ───────────────────────────────────────
        geometry_instability = compute_geometry_instability(coords_arr)

        # ── 3. Physical trajectory leaps ────────────────────────────────────
        max_jump, jump_count = compute_trajectory_jumps(bboxes, fps, frame_width, self.thresholds["trajectory_jump"])

        # ── 4. Weighted Verdict Fusion ─────────────────────────────────────
        signals_fired = []
        weighted_score = 0.0

        # Check drift magnitude
        if drift_max >= self.thresholds["facenet_drift_max"]:
            signals_fired.append("facenet_drift_max")
            weighted_score += self.weights["facenet_drift_max"]

        # Check drift variance
        if drift_var >= self.thresholds["facenet_drift_var"]:
            signals_fired.append("facenet_drift_var")
            weighted_score += self.weights["facenet_drift_var"]

        # Check geometric instability
        if geometry_instability >= self.thresholds["geometry_instability"]:
            signals_fired.append("geometry_instability")
            weighted_score += self.weights["geometry_instability"]

        # Check physical bounding box trajectory jumps
        if max_jump >= (self.thresholds["trajectory_jump"] * frame_width) or jump_count >= self.thresholds["jump_count"]:
            signals_fired.append("trajectory_jump")
            weighted_score += self.weights["trajectory_jump"]

        # Calculate final confidence
        confidence = weighted_score / self.max_possible_score

        # Verdict
        if confidence >= 0.50:
            verdict = "FAKE"
            risk_level = "HIGH"
        elif confidence < 0.25:
            verdict = "REAL"
            risk_level = "LOW"
        else:
            verdict = "UNCERTAIN"
            risk_level = "MEDIUM"

        return TrackContinuityMetrics(
            track_id=track_id,
            n_frames=n_frames,
            n_valid_frames=n_valid,
            facenet_drift_max=drift_max,
            facenet_drift_var=drift_var,
            geometry_instability_cv=geometry_instability,
            max_trajectory_jump_px=max_jump,
            trajectory_jump_count=jump_count,
            continuity_verdict=verdict,
            continuity_confidence=confidence,
            continuity_signals_fired=signals_fired,
            continuity_risk_level=risk_level,
        )

    def analyze(self, layer1_result) -> BranchFResult:
        """
        Execute Branch F identity continuity analysis on the Layer 1 Preprocessing result.
        """
        logger.info("VDS Layer 2 Branch F START | tracks=%s", layer1_result.unique_track_ids)
        t_start = time.perf_counter()
        warnings: List[str] = []
        track_metrics: Dict[str, TrackContinuityMetrics] = {}

        if not layer1_result.unique_track_ids:
            warnings.append("No tracked faces in Layer 1 preprocessing result.")
            logger.warning("No tracks to process in Branch F.")
            return BranchFResult(processing_warnings=warnings)

        # Map track crops and bounding boxes
        track_crops: Dict[str, List[np.ndarray]] = {tid: [] for tid in layer1_result.unique_track_ids}
        track_bboxes: Dict[str, List[Tuple[int, int, int, int]]] = {tid: [] for tid in layer1_result.unique_track_ids}

        for fp in layer1_result.frames:
            for tf in fp.tracked_faces:
                if tf.track_id in track_crops:
                    track_crops[tf.track_id].append(tf.aligned_crop)
                    track_bboxes[tf.track_id].append(tf.bbox)

        # Process each identity track
        for tid in layer1_result.unique_track_ids:
            crops = track_crops[tid]
            bboxes = track_bboxes[tid]
            metrics = self._analyze_track(tid, layer1_result, crops, bboxes)
            track_metrics[tid] = metrics
            logger.info(
                "  %s  →  drift_max=%.4f  var=%.5f  geom_cv=%.4f  jump_max=%.2f px  verdict=%s (conf=%.4f)",
                tid, metrics.facenet_drift_max, metrics.facenet_drift_var,
                metrics.geometry_instability_cv, metrics.max_trajectory_jump_px,
                metrics.continuity_verdict, metrics.continuity_confidence
            )

        # Global aggregations
        drifts = [m.facenet_drift_max for m in track_metrics.values()]
        geom_cvs = [m.geometry_instability_cv for m in track_metrics.values()]
        jumps = [m.max_trajectory_jump_px for m in track_metrics.values()]

        fake_tracks = [m for m in track_metrics.values() if m.continuity_verdict == "FAKE"]
        real_tracks = [m for m in track_metrics.values() if m.continuity_verdict == "REAL"]

        n_fake = len(fake_tracks)
        n_real = len(real_tracks)
        n_uncertain = len(track_metrics) - n_fake - n_real

        # Majority vote
        if n_fake > n_real:
            overall_verdict = "FAKE"
        elif n_real > n_fake:
            overall_verdict = "REAL"
        else:
            overall_verdict = "UNCERTAIN"

        overall_conf = float(np.mean([m.continuity_confidence for m in track_metrics.values()]))

        elapsed = time.perf_counter() - t_start
        logger.info(
            "VDS Layer 2 Branch F COMPLETE | elapsed=%.2fs | overall_verdict=%s (conf=%.4f)",
            elapsed, overall_verdict, overall_conf
        )

        return BranchFResult(
            track_metrics=track_metrics,
            overall_verdict=overall_verdict,
            overall_confidence=overall_conf,
            n_fake_tracks=n_fake,
            n_real_tracks=n_real,
            n_uncertain_tracks=n_uncertain,
            video_drift_mean=float(np.mean(drifts)),
            video_geometry_instability_mean=float(np.mean(geom_cvs)),
            video_max_trajectory_jump_mean=float(np.mean(jumps)),
            processing_warnings=warnings,
        )

    def unload(self) -> None:
        """Unload models to release GPU/system memory."""
        if self._models_loaded:
            try:
                del self.facenet
                self.facenet = None
                self._models_loaded = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Branch F Facenet weights successfully unloaded.")
            except Exception as e:
                logger.warning("Error unloading Branch F models: %s", e)

    def __repr__(self) -> str:
        return f"TemporalContinuityAnalyzer(device='{self.device}', models_loaded={self._models_loaded})"
