"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.temporal_semantic_analyzer
Layer   : Layer 2 — Temporal Feature Extraction
Branch  : Branch G — Scene Semantic Consistency

Orchestrates Branch G. Processes scene boundaries and full frame sequences to
evaluate lighting discrepancies, background details warping, and global frame stability.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .result_types_l2g import BranchGResult, SceneSemanticMetrics
from .semantic_signals import (
    compute_background_inconsistency,
    compute_environment_instability,
    compute_lighting_discrepancy,
)
from .world_model_net import WorldModelActiveClassifier

logger = logging.getLogger(__name__)


class TemporalSemanticAnalyzer:
    """
    VDS Layer 2, Branch G — Scene Semantic Consistency.
    
    Orchestrates lighting discrepancy checking, background detail instability,
    and world-model classification to identify global video deepfakes and diffusion/GAN anomalies.
    """
    def __init__(
        self,
        device: str = "cpu",
        world_model_checkpoint: str | Path | None = None,
        thresholds: Dict[str, float] | None = None,
        weights: Dict[str, float] | None = None,
    ) -> None:
        self.device = device
        self.world_model_checkpoint = world_model_checkpoint

        # ── Calibration System ─────────────────────────────────────────────
        self.thresholds = {
            "lighting_discrepancy_var": 0.05,   # Max variance allowed for face-to-background luminance ratio
            "background_inconsistency": 15.0,   # Max temporal background MSE allowed (object warping)
            "environment_instability":  5.0,    # Max temporal global frame intensity variance allowed (flicker)
            "world_model_score":        0.60,   # Threshold for WorldModelNet fake scene flag
        }
        if thresholds:
            self.thresholds.update(thresholds)

        self.weights = {
            "lighting_discrepancy_var": 2.5,    # Lighting ratio variance weight
            "background_inconsistency": 2.5,    # Background details warping weight
            "environment_instability":  2.0,    # Environment frame flickering weight
            "world_model_score":        3.0,    # World-model CNN classifier weight
        }
        if weights:
            self.weights.update(weights)

        self.max_possible_score = sum(self.weights.values())

        # Lazy models loading container
        self._models_loaded = False
        self.world_model: Optional[WorldModelActiveClassifier] = None

        logger.info("TemporalSemanticAnalyzer initialized | device=%s", device)

    def _lazy_load_models(self) -> None:
        if not self._models_loaded:
            logger.info("Lazily loading Branch G scene consistency models ...")
            self.world_model = WorldModelActiveClassifier(
                checkpoint_path=self.world_model_checkpoint,
                device=self.device
            )
            self._models_loaded = True
            logger.info("All Branch G models loaded.")

    def _prepare_video_tensor(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Process and resize full-frame sequence to (B=1, T, 3, 112, 112) normalized tensor.
        """
        valid_frames = [cv2.resize(f, (112, 112)) for f in frames if f is not None]
        if not valid_frames:
            return torch.zeros((1, 8, 3, 112, 112), dtype=torch.float32, device=torch.device(self.device))
        
        arr = np.stack(valid_frames, axis=0).astype(np.float32) / 255.0  # (T, 112, 112, 3)
        arr = np.transpose(arr, (0, 3, 1, 2))  # (T, 3, 112, 112)
        tensor = torch.from_numpy(arr).unsqueeze(0).to(torch.device(self.device))  # (1, T, 3, 112, 112)
        return tensor

    def _load_frame_packet(self, fp) -> Optional[np.ndarray]:
        """Lazily load full RGB frame from packet memory or disk path."""
        if fp.rgb_array is not None:
            return fp.rgb_array.copy()
        
        if fp.frame_path and Path(fp.frame_path).exists():
            try:
                bgr = cv2.imread(str(fp.frame_path))
                if bgr is not None:
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.warning("Failed to lazily load frame from path %s: %s", fp.frame_path, e)
        return None

    def _analyze_scene(self, scene, layer1_result) -> SceneSemanticMetrics:
        """Evaluate semantic consistency signals for a single scene segment."""
        fps = layer1_result.metadata.target_fps
        frame_packets = layer1_result.frames_for_scene(scene.scene_id)
        n_frames = len(frame_packets)

        # 1. Lazily load frames
        full_frames: List[np.ndarray] = []
        for fp in frame_packets:
            img = self._load_frame_packet(fp)
            if img is not None:
                full_frames.append(img)

        # Handle empty loaded frames
        if not full_frames:
            return SceneSemanticMetrics(scene_id=scene.scene_id, n_frames=n_frames)

        # 2. Extract bounding boxes and crops
        # For background masking, we union all tracked bboxes in a frame
        # For lighting discrepancy, we extract the primary face crop (largest area)
        face_crops: List[np.ndarray] = []
        primary_bboxes: List[Tuple[int, int, int, int]] = []
        all_bboxes: List[Tuple[int, int, int, int]] = []

        frame_idx = 0
        for fp in frame_packets:
            if frame_idx >= len(full_frames):
                break
            
            frame_faces = fp.tracked_faces
            if frame_faces:
                # Primary face: largest bbox area
                best_face = max(frame_faces, key=lambda f: f.width * f.height)
                face_crops.append(best_face.aligned_crop)
                primary_bboxes.append(best_face.bbox)
                
                # Default bbox for background masking
                all_bboxes.append(best_face.bbox)
            else:
                # Mock dummy values if no face in frame
                face_crops.append(np.zeros((224, 224, 3), dtype=np.uint8))
                primary_bboxes.append((0, 0, 0, 0))
                all_bboxes.append((0, 0, 0, 0))
                
            frame_idx += 1

        # ── 3. NumPy Forensic Signals ──────────────────────────────────────
        lighting_discrepancy_var = compute_lighting_discrepancy(
            face_crops, full_frames, primary_bboxes
        )
        background_inconsistency = compute_background_inconsistency(
            full_frames, all_bboxes
        )
        environment_instability = compute_environment_instability(
            full_frames
        )

        # ── 4. Deep Learning Realism Analysis ──────────────────────────────
        self._lazy_load_models()
        v_tensor = self._prepare_video_tensor(full_frames)

        with torch.no_grad():
            world_model_prob = float(self.world_model.predict_probability(v_tensor).cpu().numpy()[0])

        # ── 5. Weighted Verdict Fusion ─────────────────────────────────────
        signals_fired = []
        weighted_score = 0.0

        # Check lighting ratios variance
        if lighting_discrepancy_var >= self.thresholds["lighting_discrepancy_var"]:
            signals_fired.append("lighting_discrepancy_var")
            weighted_score += self.weights["lighting_discrepancy_var"]

        # Check background warping (MSE)
        if background_inconsistency >= self.thresholds["background_inconsistency"]:
            signals_fired.append("background_inconsistency")
            weighted_score += self.weights["background_inconsistency"]

        # Check global flickering (instability)
        if environment_instability >= self.thresholds["environment_instability"]:
            signals_fired.append("environment_instability")
            weighted_score += self.weights["environment_instability"]

        # Check WorldModelNet
        if world_model_prob >= self.thresholds["world_model_score"]:
            signals_fired.append("world_model_score")
            weighted_score += self.weights["world_model_score"]

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

        return SceneSemanticMetrics(
            scene_id=scene.scene_id,
            n_frames=len(full_frames),
            lighting_discrepancy_var=lighting_discrepancy_var,
            background_inconsistency_score=background_inconsistency,
            environment_instability_score=environment_instability,
            world_model_score=world_model_prob,
            semantic_verdict=verdict,
            semantic_confidence=confidence,
            semantic_signals_fired=signals_fired,
            semantic_risk_level=risk_level,
        )

    def analyze(self, layer1_result) -> BranchGResult:
        """
        Execute Branch G scene semantic consistency analysis on the Layer 1 Preprocessing result.
        """
        logger.info("VDS Layer 2 Branch G START | scenes=%s", [s.scene_id for s in layer1_result.scenes])
        t_start = time.perf_counter()
        warnings: List[str] = []
        scene_metrics: Dict[int, SceneSemanticMetrics] = {}

        if not layer1_result.scenes:
            warnings.append("No scene boundaries in Layer 1 preprocessing result.")
            logger.warning("No scenes to process in Branch G.")
            return BranchGResult(processing_warnings=warnings)

        # Process each scene segment
        for scene in layer1_result.scenes:
            metrics = self._analyze_scene(scene, layer1_result)
            scene_metrics[scene.scene_id] = metrics
            logger.info(
                "  scene_%d  →  lighting_var=%.5f  bg_mse=%.3f  global_var=%.4f  realism=%.4f  verdict=%s (conf=%.4f)",
                scene.scene_id, metrics.lighting_discrepancy_var, metrics.background_inconsistency_score,
                metrics.environment_instability_score, metrics.world_model_score,
                metrics.semantic_verdict, metrics.semantic_confidence
            )

        # Global aggregations
        lightings = [m.lighting_discrepancy_var for m in scene_metrics.values()]
        backgrounds = [m.background_inconsistency_score for m in scene_metrics.values()]
        instabilities = [m.environment_instability_score for m in scene_metrics.values()]
        realisms = [m.world_model_score for m in scene_metrics.values()]

        fake_scenes = [m for m in scene_metrics.values() if m.semantic_verdict == "FAKE"]
        real_scenes = [m for m in scene_metrics.values() if m.semantic_verdict == "REAL"]

        n_fake = len(fake_scenes)
        n_real = len(real_scenes)
        n_uncertain = len(scene_metrics) - n_fake - n_real

        # Majority vote
        if n_fake > n_real:
            overall_verdict = "FAKE"
        elif n_real > n_fake:
            overall_verdict = "REAL"
        else:
            overall_verdict = "UNCERTAIN"

        overall_conf = float(np.mean([m.semantic_confidence for m in scene_metrics.values()]))

        elapsed = time.perf_counter() - t_start
        logger.info(
            "VDS Layer 2 Branch G COMPLETE | elapsed=%.2fs | overall_verdict=%s (conf=%.4f)",
            elapsed, overall_verdict, overall_conf
        )

        return BranchGResult(
            scene_metrics=scene_metrics,
            overall_verdict=overall_verdict,
            overall_confidence=overall_conf,
            n_fake_scenes=n_fake,
            n_real_scenes=n_real,
            n_uncertain_scenes=n_uncertain,
            video_lighting_mean=float(np.mean(lightings)),
            video_background_mean=float(np.mean(backgrounds)),
            video_instability_mean=float(np.mean(instabilities)),
            video_world_model_mean=float(np.mean(realisms)),
            processing_warnings=warnings,
        )

    def unload(self) -> None:
        """Unload models to release GPU/system memory."""
        if self._models_loaded:
            try:
                del self.world_model
                self.world_model = None
                self._models_loaded = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Branch G WorldModelNet weights successfully unloaded.")
            except Exception as e:
                logger.warning("Error unloading Branch G models: %s", e)

    def __repr__(self) -> str:
        return f"TemporalSemanticAnalyzer(device='{self.device}', models_loaded={self._models_loaded})"
