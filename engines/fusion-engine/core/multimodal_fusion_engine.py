"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.fusion-engine.core.multimodal_fusion_engine
Layer   : Layer 3 — Cross-Modal Fusion Engine

Orchestrates cross-modal fusion by aggregating Layer 2 branches, running the
MultimodalFusionTransformer, and producing unified FusionResults.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch

from result_types import (
    FusionLocalizationResult,
    FusionResult,
    FusionTrackResult,
)
from models.fusion_transformer import MultimodalFusionTransformer

logger = logging.getLogger(__name__)


class CrossModalFusionEngine:
    """
    Orchestration engine for Layer 3 — Cross-Modal Fusion.
    
    Responsible for:
      1. Gathering outputs from Layer 2 branches (A, B, C, D, E, F, G).
      2. Structuring timelines and embeddings per face track.
      3. Applying the Cross-Attention Multimodal Transformer.
      4. Compiling localized timelines and culprit attributions into FusionResult.
    """
    def __init__(
        self,
        device: str = "cpu",
        checkpoint_path: Optional[str] = None,
        audio_dim: int = 128,
        visual_dim: int = 2560,
        temporal_dim: int = 128,
        biological_dim: int = 128,
        fused_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        threshold_fake: float = 0.6,
        threshold_real: float = 0.4,
    ) -> None:
        self.device = torch.device(device)
        self.threshold_fake = threshold_fake
        self.threshold_real = threshold_real

        # Initialize the native PyTorch Multimodal Transformer
        self.model = MultimodalFusionTransformer(
            audio_dim=audio_dim,
            visual_dim=visual_dim,
            temporal_dim=temporal_dim,
            biological_dim=biological_dim,
            fused_dim=fused_dim,
            nhead=nhead,
            num_layers=num_layers,
        )

        if checkpoint_path:
            self.load_weights(checkpoint_path)
        else:
            logger.info("Initializing CrossModalFusionEngine with random weights (offline fallback mode).")

        self.model.to(self.device)
        self.model.eval()

    def load_weights(self, path: str) -> None:
        """Load trained weights into the fusion transformer model."""
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info("Successfully loaded fusion transformer weights from %s", path)
        except Exception as e:
            logger.error("Failed to load fusion transformer weights from %s: %s", path, e)

    def process_layer2_results(
        self,
        branch_a_result: Optional[Any] = None,
        branch_b_result: Optional[Any] = None,
        branch_c_result: Optional[Any] = None,
        branch_d_result: Optional[Any] = None,
        branch_e_result: Optional[Any] = None,
        branch_f_result: Optional[Any] = None,
        branch_g_result: Optional[Any] = None,
    ) -> FusionResult:
        """
        Gathers Layer 2 results, aligns timelines per track, executes the fusion model,
        and constructs the final FusionResult.
        """
        warnings = []
        track_results: Dict[str, FusionTrackResult] = {}

        # 1. Identify all active face tracks across branches
        track_ids = set()
        if branch_a_result is not None:
            if hasattr(branch_a_result, "unique_track_ids"):
                track_ids.update(branch_a_result.unique_track_ids)
            elif hasattr(branch_a_result, "frame_results"):
                track_ids.update(branch_a_result.frame_results.keys())

        # Collect from other results if Branch A is missing
        for res, name in [
            (branch_b_result, "Branch B"),
            (branch_c_result, "Branch C"),
            (branch_d_result, "Branch D"),
            (branch_e_result, "Branch E"),
            (branch_f_result, "Branch F"),
        ]:
            if res is not None:
                if hasattr(res, "track_metrics"):
                    track_ids.update(res.track_metrics.keys())
            else:
                warnings.append(f"Missing input from {name}.")

        if branch_g_result is None:
            warnings.append("Missing input from Branch G.")

        if not track_ids:
            logger.warning("No face tracks found in any provided Layer 2 branch results.")
            return FusionResult(
                overall_verdict="UNCERTAIN",
                overall_confidence=0.0,
                processing_warnings=warnings + ["No face tracks detected."],
            )

        # 2. Process each track
        for track_id in sorted(track_ids):
            # A. Visual Timeline Extraction (Branch A)
            visual_list = []
            T_v = 0
            if branch_a_result is not None:
                frame_results = getattr(branch_a_result, "frame_results", {}).get(track_id, [])
                T_v = len(frame_results)
                for f_res in frame_results:
                    # Attempt to get artifact_embedding
                    emb = getattr(f_res, "artifact_embedding", None)
                    if emb is None:
                        # Fallback to concatenating the 5 branch embeddings (5 * 512 = 2560)
                        branch_embs = getattr(f_res, "branch_embeddings", {})
                        if branch_embs:
                            concat_emb = []
                            for k in ["spatial", "frequency", "discrepancy", "noise", "fingerprint"]:
                                if k in branch_embs and isinstance(branch_embs[k], np.ndarray):
                                    concat_emb.append(branch_embs[k])
                                else:
                                    concat_emb.append(np.zeros(512, dtype=np.float32))
                            emb = np.concatenate(concat_emb)
                        else:
                            emb = np.zeros(2560, dtype=np.float32)
                    
                    visual_list.append(emb)

            # Ensure we have at least a dummy visual sequence if Branch A is completely empty
            if not visual_list:
                T_v = max(16, T_v)  # default length
                visual_list = [np.zeros(2560, dtype=np.float32) for _ in range(T_v)]

            visual_tensor = torch.tensor(np.stack(visual_list), dtype=torch.float32).unsqueeze(0).to(self.device)

            # B. Audio FeatureTimeline Construction (Branch D)
            # Standard audio feature dimension is 128
            audio_feat = np.zeros(128, dtype=np.float32)
            if branch_d_result is not None:
                track_metrics = getattr(branch_d_result, "track_metrics", {}).get(track_id, None)
                if track_metrics is None and hasattr(branch_d_result, "metrics_for_track"):
                    track_metrics = branch_d_result.metrics_for_track(track_id)
                
                if track_metrics is not None:
                    # Map SyncNet & AV-HuBERT scalar metrics to initial indices of 128d vector
                    audio_feat[0] = getattr(track_metrics, "syncnet_confidence", 0.0)
                    audio_feat[1] = float(getattr(track_metrics, "syncnet_offset", 0))
                    audio_feat[2] = getattr(track_metrics, "av_hubert_score", 0.0)
                    audio_feat[3] = getattr(track_metrics, "phoneme_viseme_inconsistency", 0.0)
                    audio_feat[4] = getattr(track_metrics, "audio_video_delay", 0.0)
                    audio_feat[5] = getattr(track_metrics, "emotion_mismatch_score", 0.0)

            # Repeat audio feature over visual steps to form a timeline
            audio_list = [audio_feat for _ in range(T_v)]
            audio_tensor = torch.tensor(np.stack(audio_list), dtype=torch.float32).unsqueeze(0).to(self.device)

            # C. Temporal FeatureTimeline Construction (Branch B / C / F / G)
            temporal_feat = np.zeros(128, dtype=np.float32)
            # Populate from Branch C (Motion models)
            if branch_c_result is not None:
                track_metrics = getattr(branch_c_result, "track_metrics", {}).get(track_id, None)
                if track_metrics is not None:
                    temporal_feat[0] = getattr(track_metrics, "timesformer_score", 0.0)
                    temporal_feat[1] = getattr(track_metrics, "videomae_score", 0.0)
                    temporal_feat[2] = getattr(track_metrics, "slowfast_score", 0.0)
                    temporal_feat[3] = getattr(track_metrics, "jitter_entropy", 0.0)
                    temporal_feat[4] = getattr(track_metrics, "landmark_velocity_variance", 0.0)
                    temporal_feat[5] = getattr(track_metrics, "blink_asymmetry", 0.0)
                    temporal_feat[6] = getattr(track_metrics, "blink_duration_anomaly", 0.0)
                    temporal_feat[7] = getattr(track_metrics, "frame_inconsistency_score", 0.0)

            # Populate from Branch B (Consistency analyzer)
            if branch_b_result is not None:
                track_metrics = getattr(branch_b_result, "track_metrics", {}).get(track_id, None)
                if track_metrics is not None:
                    temporal_feat[8] = getattr(track_metrics, "fake_prob_mean", 0.0)
                    temporal_feat[9] = getattr(track_metrics, "fake_prob_variance", 0.0)
                    temporal_feat[10] = getattr(track_metrics, "embedding_drift_mean", 0.0)
                    temporal_feat[11] = getattr(track_metrics, "heatmap_flux_mean", 0.0)

            # Populate from Branch F (Continuity)
            if branch_f_result is not None:
                track_metrics = getattr(branch_f_result, "track_metrics", {}).get(track_id, None)
                if track_metrics is not None:
                    temporal_feat[12] = getattr(track_metrics, "identity_drift_mean", 0.0) if hasattr(track_metrics, "identity_drift_mean") else 0.0
                    temporal_feat[13] = getattr(track_metrics, "geometry_instability", 0.0) if hasattr(track_metrics, "geometry_instability") else 0.0

            # Repeat temporal feature over steps to form a timeline
            temporal_list = [temporal_feat for _ in range(T_v)]
            temporal_tensor = torch.tensor(np.stack(temporal_list), dtype=torch.float32).unsqueeze(0).to(self.device)

            # D. Biological FeatureTimeline Construction (Branch E)
            biological_feat = np.zeros(128, dtype=np.float32)
            if branch_e_result is not None:
                track_metrics = getattr(branch_e_result, "track_metrics", {}).get(track_id, None)
                if track_metrics is not None:
                    biological_feat[0] = getattr(track_metrics, "physnet_score", 0.0)
                    biological_feat[1] = getattr(track_metrics, "deepphys_score", 0.0)
                    biological_feat[2] = getattr(track_metrics, "pulse_rate_mean", 0.0)
                    biological_feat[3] = getattr(track_metrics, "pulse_rate_std", 0.0)
                    biological_feat[4] = getattr(track_metrics, "pulse_snr", 0.0)
                    biological_feat[5] = getattr(track_metrics, "spatial_correlation_mean", 0.0)

            # Repeat biological feature over steps to form a timeline
            biological_list = [biological_feat for _ in range(T_v)]
            biological_tensor = torch.tensor(np.stack(biological_list), dtype=torch.float32).unsqueeze(0).to(self.device)

            # 3. Model Inference execution
            with torch.no_grad():
                logits, attribution, localization = self.model(
                    audio_feat=audio_tensor,
                    visual_feat=visual_tensor,
                    temporal_feat=temporal_tensor,
                    biological_feat=biological_tensor,
                )

                # Compute soft fake probability
                probs = torch.softmax(logits, dim=-1)
                fake_prob = float(probs[0, 1].item())

                # Attribution values mapping
                attrib_scores = attribution[0].cpu().numpy().tolist()
                modality_ratings = {
                    "audio": float(attrib_scores[0]),
                    "visual": float(attrib_scores[1]),
                    "temporal": float(attrib_scores[2]),
                    "biological": float(attrib_scores[3]),
                }

                # Localization probabilities
                if localization is not None:
                    loc_timeline = localization[0].cpu().numpy().tolist()
                else:
                    loc_timeline = [fake_prob] * T_v

            # 4. Map logits to verdict
            if fake_prob >= self.threshold_fake:
                verdict = "FAKE"
                confidence = fake_prob
            elif fake_prob <= self.threshold_real:
                verdict = "REAL"
                confidence = 1.0 - fake_prob
            else:
                verdict = "UNCERTAIN"
                confidence = 2.0 * abs(fake_prob - 0.5)  # Calibration mapping [0, 1]

            # Store results
            loc_result = FusionLocalizationResult(
                temporal_timeline=loc_timeline,
                modality_attribution=modality_ratings,
            )
            
            # Culprits ranked by attribution
            sorted_culprits = dict(sorted(modality_ratings.items(), key=lambda item: item[1], reverse=True))

            track_results[track_id] = FusionTrackResult(
                track_id=track_id,
                verdict=verdict,
                confidence=confidence,
                localization=loc_result,
                modality_culprits=sorted_culprits,
            )

        # 5. Global verdict compilation
        overall_verdict = "REAL"
        overall_confidence = 1.0
        fake_tracks = [tr for tr in track_results.values() if tr.verdict == "FAKE"]
        uncertain_tracks = [tr for tr in track_results.values() if tr.verdict == "UNCERTAIN"]

        if fake_tracks:
            overall_verdict = "FAKE"
            # Max fake probability determines overall confidence
            overall_confidence = max(tr.confidence for tr in fake_tracks)
        elif uncertain_tracks:
            overall_verdict = "UNCERTAIN"
            overall_confidence = max(tr.confidence for tr in uncertain_tracks)
        else:
            # All are REAL, choose mean or max confidence of REAL tracks
            overall_verdict = "REAL"
            overall_confidence = sum(tr.confidence for tr in track_results.values()) / len(track_results)

        return FusionResult(
            overall_verdict=overall_verdict,
            overall_confidence=overall_confidence,
            track_results=track_results,
            processing_warnings=warnings,
            metadata={
                "n_tracks": len(track_ids),
                "device": str(self.device),
            },
        )
