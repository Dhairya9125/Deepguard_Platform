"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.result_types_l2g
Layer   : Layer 2 — Temporal Feature Extraction
Branch  : Branch G — Scene Semantic Consistency

Defines output dataclasses for Branch G (Scene Semantic Consistency).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SceneSemanticMetrics:
    """
    Scene-level semantic consistency metrics and deep learning world-model scores.
    """
    scene_id: int
    n_frames: int = 0

    # --- NumPy Forensic Signals ---
    lighting_discrepancy_var: float = 0.0      # Variance of face-to-background luminance ratio
    background_inconsistency_score: float = 0.0 # BBox-masked temporal background variation (MSE)
    environment_instability_score: float = 0.0  # High-frequency global frame flickering

    # --- Deep learning scene classifier ---
    world_model_score: float = 0.0             # [0, 1] fake probability from WorldModelNet

    # --- Verdict & Calibration ---
    semantic_verdict: str = "UNCERTAIN"        # 'REAL' | 'FAKE' | 'UNCERTAIN'
    semantic_confidence: float = 0.0           # Normalized confidence score in [0, 1]
    semantic_signals_fired: List[str] = field(default_factory=list)
    semantic_risk_level: str = "LOW"           # 'LOW' | 'MEDIUM' | 'HIGH'

    @property
    def is_semantic_fake(self) -> bool:
        """Helper to quickly check if the scene is flagged as FAKE."""
        return self.semantic_verdict == "FAKE"

    def to_dict(self) -> Dict:
        """Serialize scene metrics to a JSON-safe plain dictionary."""
        return {
            "scene_id": self.scene_id,
            "n_frames": self.n_frames,
            # Forensic signals
            "lighting_discrepancy_var": round(self.lighting_discrepancy_var, 6),
            "background_inconsistency_score": round(self.background_inconsistency_score, 6),
            "environment_instability_score": round(self.environment_instability_score, 6),
            # Model score
            "world_model_score": round(self.world_model_score, 6),
            # Verdict
            "semantic_verdict": self.semantic_verdict,
            "semantic_confidence": round(self.semantic_confidence, 4),
            "semantic_signals_fired": self.semantic_signals_fired,
            "semantic_risk_level": self.semantic_risk_level,
        }


@dataclass
class BranchGResult:
    """
    Top-level output of VDS Layer 2, Branch G — Scene Semantic Consistency.
    """
    scene_metrics: Dict[int, SceneSemanticMetrics] = field(default_factory=dict)
    overall_verdict: str = "UNCERTAIN"
    overall_confidence: float = 0.0
    n_fake_scenes: int = 0
    n_real_scenes: int = 0
    n_uncertain_scenes: int = 0

    # Global video aggregates
    video_lighting_mean: float = 0.0
    video_background_mean: float = 0.0
    video_instability_mean: float = 0.0
    video_world_model_mean: float = 0.0

    processing_warnings: List[str] = field(default_factory=list)

    @property
    def unique_scene_ids(self) -> List[int]:
        return sorted(self.scene_metrics.keys())

    @property
    def n_scenes(self) -> int:
        return len(self.scene_metrics)

    def metrics_for_scene(self, scene_id: int) -> Optional[SceneSemanticMetrics]:
        return self.scene_metrics.get(scene_id)

    def summary(self) -> Dict:
        """JSON-serializable video-level summary."""
        return {
            "overall_verdict": self.overall_verdict,
            "overall_confidence": round(self.overall_confidence, 4),
            "n_scenes": self.n_scenes,
            "n_fake_scenes": self.n_fake_scenes,
            "n_real_scenes": self.n_real_scenes,
            "n_uncertain_scenes": self.n_uncertain_scenes,
            "video_lighting_mean": round(self.video_lighting_mean, 4),
            "video_background_mean": round(self.video_background_mean, 4),
            "video_instability_mean": round(self.video_instability_mean, 4),
            "video_world_model_mean": round(self.video_world_model_mean, 4),
            "n_warnings": len(self.processing_warnings),
            "warnings": self.processing_warnings,
            "scenes": {
                str(sid): m.to_dict()
                for sid, m in self.scene_metrics.items()
            },
        }
