"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.temporal_rppg_analyzer
Layer   : Layer 2 — Temporal Feature Extraction
Branch  : Branch E — rPPG Biological Signal Analysis

Orchestrates Branch E. Processes tracked face crops sequence and landmarks to
analyze pulse consistency, biological liveness, and blood-flow realism.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .deepphys_model import DeepPhysActiveModel
from .physnet_model import PhysNetActiveModel
from .result_types_l2e import BranchEResult, TrackRPPGMetrics
from .rppg_signals import (
    compute_rppg_pos,
    evaluate_blood_flow_realism,
    evaluate_pulse_consistency,
    extract_skin_rois,
    estimate_heart_rate,
)

logger = logging.getLogger(__name__)


class TemporalRPPGAnalyzer:
    """
    VDS Layer 2, Branch E — rPPG Biological Signal Analysis.
    
    Orchestrates the evaluation of blood-volume pulse (BVP) consistency,
    biological liveness, and blood-flow realism for video deepfake detection.
    """
    def __init__(
        self,
        device: str = "cpu",
        physnet_checkpoint: str | Path | None = None,
        deepphys_checkpoint: str | Path | None = None,
        thresholds: Dict[str, float] | None = None,
        weights: Dict[str, float] | None = None,
    ) -> None:
        self.device = device
        self.physnet_checkpoint = physnet_checkpoint
        self.deepphys_checkpoint = deepphys_checkpoint

        # ── Calibration System ─────────────────────────────────────────────
        self.thresholds = {
            "physnet_score":            0.60,   # Threshold for PhysNet model flag
            "deepphys_score":           0.60,   # Threshold for DeepPhys model flag
            "pulse_consistency":        15.0,   # Max heart rate std allowed before flagging inconsistency (BPM)
            "liveness_min":             45.0,   # Physiological min heart rate (BPM)
            "liveness_max":             180.0,  # Physiological max heart rate (BPM)
            "pulse_snr":                1.2,    # Min heart rate spectral SNR allowed
            "spatial_correlation_min":  0.35,   # Min regional correlation for real skin blood flow
            "spatial_correlation_max":  0.99,   # Max regional correlation (above indicates uniform artificial flow)
        }
        if thresholds:
            self.thresholds.update(thresholds)

        self.weights = {
            "physnet_score":            3.0,    # Weight on PhysNet 3D CNN
            "deepphys_score":           3.0,    # Weight on DeepPhys spatial difference network
            "pulse_consistency":        2.0,    # Heart rate variability standard deviation
            "biological_liveness":      1.5,    # Peak pulse SNR / boundary violations
            "blood_flow_realism":      1.5,    # Spatio-temporal cross-correlation of ROIs
        }
        if weights:
            self.weights.update(weights)

        self.max_possible_score = sum(self.weights.values())

        # Lazy models loading container
        self._models_loaded = False
        self.physnet: Optional[PhysNetActiveModel] = None
        self.deepphys: Optional[DeepPhysActiveModel] = None

        logger.info("TemporalRPPGAnalyzer initialized | device=%s", device)

    def _lazy_load_models(self) -> None:
        if not self._models_loaded:
            logger.info("Lazily loading Branch E biological rPPG models ...")
            self.physnet = PhysNetActiveModel(
                checkpoint_path=self.physnet_checkpoint,
                device=self.device
            )
            self.deepphys = DeepPhysActiveModel(
                checkpoint_path=self.deepphys_checkpoint,
                device=self.device
            )
            self._models_loaded = True
            logger.info("All Branch E models loaded.")

    def _prepare_video_tensor(self, crops: List[np.ndarray]) -> torch.Tensor:
        """
        Process and resize face crop sequence to (B=1, T, 3, 128, 128) normalized tensor.
        """
        valid_crops = [cv2.resize(c, (128, 128)) for c in crops if c is not None]
        if not valid_crops:
            # Return empty dummy tensor
            return torch.zeros((1, 8, 3, 128, 128), dtype=torch.float32, device=torch.device(self.device))
        
        arr = np.stack(valid_crops, axis=0).astype(np.float32) / 255.0  # (T, 128, 128, 3)
        arr = np.transpose(arr, (0, 3, 1, 2))  # (T, 3, 128, 128)
        tensor = torch.from_numpy(arr).unsqueeze(0).to(torch.device(self.device))  # (1, T, 3, 128, 128)
        return tensor

    def _analyze_track(
        self,
        track_id: str,
        layer1_result,
        crops: List[np.ndarray]
    ) -> TrackRPPGMetrics:
        """Evaluate Branch E rPPG metrics for a single identity track."""
        n_frames = len(crops)
        valid_crops = [c for c in crops if c is not None]
        n_valid = len(valid_crops)
        fps = layer1_result.metadata.target_fps

        # Retrieve landmarks
        landmarks_list = layer1_result.landmarks_for_track(track_id)
        
        # 1. Extract regional RGB timelines
        forehead_rgb = []
        left_cheek_rgb = []
        right_cheek_rgb = []
        
        lm_idx = 0
        for tf_crop in crops:
            if tf_crop is not None and lm_idx < len(landmarks_list):
                lm = landmarks_list[lm_idx]
                if lm.detection_success and lm.smoothed_landmarks is not None:
                    rois = extract_skin_rois(tf_crop, lm.smoothed_landmarks)
                    forehead_rgb.append(rois["forehead"])
                    left_cheek_rgb.append(rois["left_cheek"])
                    right_cheek_rgb.append(rois["right_cheek"])
                else:
                    forehead_rgb.append(np.zeros(3))
                    left_cheek_rgb.append(np.zeros(3))
                    right_cheek_rgb.append(np.zeros(3))
                lm_idx += 1
            else:
                forehead_rgb.append(np.zeros(3))
                left_cheek_rgb.append(np.zeros(3))
                right_cheek_rgb.append(np.zeros(3))

        forehead_arr = np.array(forehead_rgb)
        left_cheek_arr = np.array(left_cheek_rgb)
        right_cheek_arr = np.array(right_cheek_rgb)

        # ── 2. NumPy-based Regional Pulse Waveforms ────────────────────────
        forehead_pulse = compute_rppg_pos(forehead_arr, fps)
        left_cheek_pulse = compute_rppg_pos(left_cheek_arr, fps)
        right_cheek_pulse = compute_rppg_pos(right_cheek_arr, fps)

        # Blood flow realism (correlation between spatial regions)
        spatial_corr = evaluate_blood_flow_realism(
            forehead_pulse, left_cheek_pulse, right_cheek_pulse
        )

        # Overall pulse wave (average of regions)
        overall_pulse = (forehead_pulse + left_cheek_pulse + right_cheek_pulse) / 3.0

        # Heart rate estimation & spectral SNR
        hr_bpm, snr = estimate_heart_rate(overall_pulse, fps)

        # Temporal pulse rate consistency (sliding windows std)
        mean_hr, std_hr = evaluate_pulse_consistency(overall_pulse, fps)

        # ── 3. Deep Learning Pulse Analysis ────────────────────────────────
        self._lazy_load_models()
        v_tensor = self._prepare_video_tensor(crops)

        with torch.no_grad():
            physnet_prob = float(self.physnet.predict_probability(v_tensor, fps).cpu().numpy()[0])
            deepphys_prob = float(self.deepphys.predict_probability(v_tensor, fps).cpu().numpy()[0])

        # ── 4. Weighted Verdict Fusion ─────────────────────────────────────
        signals_fired = []
        weighted_score = 0.0

        # Check PhysNet
        if physnet_prob >= self.thresholds["physnet_score"]:
            signals_fired.append("physnet_score")
            weighted_score += self.weights["physnet_score"]

        # Check DeepPhys
        if deepphys_prob >= self.thresholds["deepphys_score"]:
            signals_fired.append("deepphys_score")
            weighted_score += self.weights["deepphys_score"]

        # Check Pulse Consistency (variability std)
        if std_hr >= self.thresholds["pulse_consistency"]:
            signals_fired.append("pulse_consistency")
            weighted_score += self.weights["pulse_consistency"]

        # Check Biological Liveness (HR limits or low spectral SNR)
        if (
            hr_bpm < self.thresholds["liveness_min"]
            or hr_bpm > self.thresholds["liveness_max"]
            or snr < self.thresholds["pulse_snr"]
        ):
            signals_fired.append("biological_liveness")
            weighted_score += self.weights["biological_liveness"]

        # Check Blood-Flow Realism (regional spatial correlation bounds)
        if (
            spatial_corr < self.thresholds["spatial_correlation_min"]
            or spatial_corr > self.thresholds["spatial_correlation_max"]
        ):
            signals_fired.append("blood_flow_realism")
            weighted_score += self.weights["blood_flow_realism"]

        # Calculate confidence
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

        return TrackRPPGMetrics(
            track_id=track_id,
            n_frames=n_frames,
            n_valid_frames=n_valid,
            physnet_score=physnet_prob,
            deepphys_score=deepphys_prob,
            pulse_rate_mean=mean_hr if mean_hr > 0 else hr_bpm,
            pulse_rate_std=std_hr,
            pulse_snr=snr,
            spatial_correlation_mean=spatial_corr,
            rppg_verdict=verdict,
            rppg_confidence=confidence,
            rppg_signals_fired=signals_fired,
            rppg_risk_level=risk_level,
        )

    def analyze(self, layer1_result) -> BranchEResult:
        """
        Execute Branch E rPPG biological liveness analysis on the Layer 1 Preprocessing result.
        """
        logger.info("VDS Layer 2 Branch E START | tracks=%s", layer1_result.unique_track_ids)
        t_start = time.perf_counter()
        warnings: List[str] = []
        track_metrics: Dict[str, TrackRPPGMetrics] = {}

        if not layer1_result.unique_track_ids:
            warnings.append("No tracked faces in Layer 1 preprocessing result.")
            logger.warning("No tracks to process in Branch E.")
            return BranchEResult(processing_warnings=warnings)

        # Map track crops
        track_crops: Dict[str, List[np.ndarray]] = {tid: [] for tid in layer1_result.unique_track_ids}
        for fp in layer1_result.frames:
            for tf in fp.tracked_faces:
                if tf.track_id in track_crops:
                    track_crops[tf.track_id].append(tf.aligned_crop)

        # Process each identity track
        for tid in layer1_result.unique_track_ids:
            crops = track_crops[tid]
            metrics = self._analyze_track(tid, layer1_result, crops)
            track_metrics[tid] = metrics
            logger.info(
                "  %s  →  physnet=%.4f  deepphys=%.4f  hr=%.1f bpm  std=%.2f  corr=%.3f  verdict=%s (conf=%.4f)",
                tid, metrics.physnet_score, metrics.deepphys_score,
                metrics.pulse_rate_mean, metrics.pulse_rate_std,
                metrics.spatial_correlation_mean, metrics.rppg_verdict, metrics.rppg_confidence
            )

        # Global video aggregations
        physnets = [m.physnet_score for m in track_metrics.values()]
        deepphys = [m.deepphys_score for m in track_metrics.values()]
        hrs = [m.pulse_rate_mean for m in track_metrics.values()]
        corrs = [m.spatial_correlation_mean for m in track_metrics.values()]

        fake_tracks = [m for m in track_metrics.values() if m.rppg_verdict == "FAKE"]
        real_tracks = [m for m in track_metrics.values() if m.rppg_verdict == "REAL"]

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

        overall_conf = float(np.mean([m.rppg_confidence for m in track_metrics.values()]))

        elapsed = time.perf_counter() - t_start
        logger.info(
            "VDS Layer 2 Branch E COMPLETE | elapsed=%.2fs | overall_verdict=%s (conf=%.4f)",
            elapsed, overall_verdict, overall_conf
        )

        return BranchEResult(
            track_metrics=track_metrics,
            overall_verdict=overall_verdict,
            overall_confidence=overall_conf,
            n_fake_tracks=n_fake,
            n_real_tracks=n_real,
            n_uncertain_tracks=n_uncertain,
            video_physnet_mean=float(np.mean(physnets)),
            video_deepphys_mean=float(np.mean(deepphys)),
            video_pulse_rate_mean=float(np.mean(hrs)),
            video_correlation_mean=float(np.mean(corrs)),
            processing_warnings=warnings,
        )

    def unload(self) -> None:
        """Unload models to release system/GPU memory."""
        if self._models_loaded:
            try:
                del self.physnet
                del self.deepphys
                self.physnet = None
                self.deepphys = None
                self._models_loaded = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Branch E rPPG model weights successfully unloaded.")
            except Exception as e:
                logger.warning("Error unloading Branch E models: %s", e)

    def __repr__(self) -> str:
        return f"TemporalRPPGAnalyzer(device='{self.device}', models_loaded={self._models_loaded})"
