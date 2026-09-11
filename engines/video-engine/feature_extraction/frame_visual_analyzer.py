"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.frame_visual_analyzer
Layer   : Layer 2 — Temporal Feature Extraction
Branch  : Branch A — Frame-Level Visual Analysis

Architecture Reference (DeepGuard_Platform_Architecture.docx — VDS Layer 2, Branch A):
  Runs the complete IDS pipeline on every tracked face crop extracted in Layer 1.
  Produces a per-track, per-frame timeline of IDS verdicts for temporal analysis.

Pipeline (per aligned_crop):
    aligned_crop (224×224 RGB uint8)
           │
           ▼  IDS Preprocessing (reused)
           │  ImageNormalizer  → spatial_tensor  (B, 3, 224, 224)
           │  FrequencyAnalyzer → freq_tensor    (B, 3, 224, 224)
           │
           ▼  IDS Feature Extraction — 5 branches (all reused)
           │  ClipSpatialBranch       → spatial_emb   (B, 512)
           │  EfficientNetFreqBranch  → freq_emb      (B, 512)
           │  DiscrepancyBranch       → disc_emb      (B, 512)
           │  NoiseResidualBranch     → noise_emb     (B, 512)
           │  FingerprintBranch       → finger_emb    (B, 512)
           │
           ▼  IDS Fusion (reused)
           │  IDSFusionEngine → fake_probability, OOD_score,
           │                    manipulation_heatmap (B,1,224,224),
           │                    artifact_embedding (B, 2560)
           │
           ▼  IDS Localization (reused)
           │  LocalizationEngine → binary_mask, blend_boundaries, bboxes
           │
           ▼
    FrameVisualResult  (one per tracked face per frame)

Reuse contract:
    NOTHING is re-implemented. All 9 IDS components are imported verbatim
    through `ids_adapter.load_ids_components()`.

Design decisions:
  1. Batched mini-batch inference (default batch_size=8):
       Crops are queued and stacked before model forward passes.
       This amortizes the per-batch GPU kernel launch overhead.
       On CPU this also improves cache locality.

  2. Lazy IDS loading:
       `IDSComponents` are not loaded until the first call to `analyze()`.
       This avoids loading large CLIP/Swin models at import time.

  3. Non-fatal per-crop error handling:
       If any IDS stage raises on a specific crop (degenerate image,
       MediaPipe-failed crop with extreme padding), the exception is caught
       and a `FrameVisualResult(detection_success=False)` is emitted.
       The pipeline continues with the next crop.

  4. Frequency tensor vs spatial tensor sharing:
       For the discrepancy, noise, and fingerprint branches, which all take
       spatial pixel_values, the same `spatial_tensor` is reused — no re-
       normalisation. The frequency stack is computed separately for Branch B.

  5. `@torch.no_grad()` on all model inference:
       Branch A is inference-only. Gradients are never needed here.

Usage:
    from feature_extraction.frame_visual_analyzer import FrameVisualAnalyzer
    from preprocessing.result_types import VideoPreprocessingResult

    analyzer = FrameVisualAnalyzer(device="cpu", batch_size=8)
    branch_a_result = analyzer.analyze(layer1_result)

    for track_id, results in branch_a_result.frame_results.items():
        probs = [r.fake_probability for r in results if r.detection_success]
        print(f"{track_id}: mean_prob={sum(probs)/len(probs):.3f}")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .ids_adapter     import IDSComponents, load_ids_components
from .result_types_l2 import BranchAResult, FrameVisualResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FrameVisualAnalyzer
# ---------------------------------------------------------------------------

class FrameVisualAnalyzer:
    """
    VDS Layer 2, Branch A — Frame-Level Visual Analysis.

    Runs the complete IDS pipeline on every aligned face crop produced by
    VDS Layer 1 FaceTracker and returns a per-track fake-probability timeline.

    Args:
        device (str):
            PyTorch device. Default: 'cpu'. Use 'cuda' for GPU acceleration.
        weights_dir (Path, optional):
            Directory containing fine-tuned IDS checkpoint .pt files.
            If None, uses ImageNet-pretrained weights (inference without
            IDS fine-tuning — still useful for structural forensic signals).
        localization_threshold (float):
            Binary mask threshold for LocalizationEngine. Default: 0.6.
        batch_size (int):
            Number of face crops processed per forward pass. Default: 8.
            Reduce to 1 for very low memory environments.
        proj_dim (int):
            IDS branch embedding dimension. Must match trained checkpoints. Default: 512.
    """

    def __init__(
        self,
        device:                 str            = "cpu",
        weights_dir:            Optional[Path] = None,
        localization_threshold: float          = 0.6,
        batch_size:             int            = 8,
        proj_dim:               int            = 512,
    ) -> None:
        self.device                 = device
        self.weights_dir            = Path(weights_dir) if weights_dir else None
        self.localization_threshold = localization_threshold
        self.batch_size             = batch_size
        self.proj_dim               = proj_dim

        # Lazy-initialised on first analyze() call
        self._ids: Optional[IDSComponents] = None

        logger.info(
            "FrameVisualAnalyzer init | device=%s | batch_size=%d | "
            "loc_threshold=%.2f | weights=%s",
            device, batch_size, localization_threshold,
            weights_dir or "pretrained",
        )

    # ------------------------------------------------------------------
    # Lazy IDS loader
    # ------------------------------------------------------------------

    def _get_ids(self) -> IDSComponents:
        """Lazily load all IDS components on first call."""
        if self._ids is None:
            logger.info("Loading IDS components (first call) …")
            self._ids = load_ids_components(
                device=self.device,
                weights_dir=self.weights_dir,
                proj_dim=self.proj_dim,
                loc_threshold=self.localization_threshold,
            )
        return self._ids

    # ------------------------------------------------------------------
    # Core: single crop inference
    # ------------------------------------------------------------------

    def analyze_crop(
        self,
        track_id:         str,
        frame_id:         int,
        timestamp_ms:     float,
        aligned_crop_rgb: Optional[np.ndarray],
    ) -> FrameVisualResult:
        """
        Run the full IDS pipeline on a single aligned face crop.

        This is the atomic unit of Branch A — called in batch from `analyze()`.
        It can also be called standalone for debugging or single-frame tests.

        Args:
            track_id         : Stable track ID from VDS Layer 1.
            frame_id         : Frame index from FramePacket.
            timestamp_ms     : Timestamp of the parent frame.
            aligned_crop_rgb : (224, 224, 3) RGB uint8 numpy array.
                               None → returns detection_success=False.

        Returns:
            FrameVisualResult — always returned, even on failure.
        """
        # ── Input validation ─────────────────────────────────────────────
        if aligned_crop_rgb is None:
            return FrameVisualResult(
                track_id=track_id, frame_id=frame_id, timestamp_ms=timestamp_ms,
                detection_success=False,
                failure_reason="aligned_crop_rgb is None",
            )

        if aligned_crop_rgb.ndim != 3 or aligned_crop_rgb.shape[2] != 3:
            return FrameVisualResult(
                track_id=track_id, frame_id=frame_id, timestamp_ms=timestamp_ms,
                detection_success=False,
                failure_reason=f"Invalid crop shape: {aligned_crop_rgb.shape}",
            )

        try:
            return self._run_ids_single(track_id, frame_id, timestamp_ms, aligned_crop_rgb)
        except Exception as exc:
            logger.warning(
                "IDS failed on track=%s frame=%d: %s", track_id, frame_id, exc
            )
            return FrameVisualResult(
                track_id=track_id, frame_id=frame_id, timestamp_ms=timestamp_ms,
                detection_success=False,
                failure_reason=str(exc),
            )

    def _run_ids_single(
        self,
        track_id:    str,
        frame_id:    int,
        timestamp_ms: float,
        crop_rgb:    np.ndarray,
    ) -> FrameVisualResult:
        """
        Execute IDS forward pass for a single crop. Internal implementation.
        All model calls are wrapped in torch.no_grad().
        """
        import torch  # noqa: PLC0415
        ids = self._get_ids()

        # ── 1. IDS Preprocessing ─────────────────────────────────────────
        # Resize crop to 224×224 if needed (DNN tracker may produce different sizes)
        if crop_rgb.shape[:2] != (224, 224):
            crop_rgb = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Spatial normalisation (ImageNormalizer): (3, 224, 224) CHW float32
        spatial_np = ids.normalizer.normalize(crop_rgb, is_bgr=False)  # CHW float32
        spatial_tensor = torch.from_numpy(spatial_np).unsqueeze(0).to(ids.device)  # (1,3,224,224)

        # Frequency stack (FrequencyAnalyzer): (3, 224, 224) CHW float32 tensor
        # FrequencyAnalyzer accepts BGR or RGB; we pass RGB (it uses cv2 internally
        # so convert to BGR first for correctness)
        crop_bgr  = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        freq_tensor = ids.freq_analyzer.get_stacked_frequency_tensor(crop_bgr)
        freq_tensor = freq_tensor.unsqueeze(0).to(ids.device)  # (1,3,224,224)

        # ── 2. IDS Feature Extraction — 5 branches ───────────────────────
        with torch.no_grad():
            spatial_emb    = ids.branch_spatial(spatial_tensor)      # (1,512)
            freq_emb       = ids.branch_freq(freq_tensor)             # (1,512)
            disc_emb       = ids.branch_disc(spatial_tensor)          # (1,512)
            noise_emb      = ids.branch_noise(spatial_tensor)         # (1,512)
            finger_emb     = ids.branch_fingerprint(spatial_tensor)   # (1,512)

        # Collect branch embeddings (CPU numpy, shape (512,))
        branch_embeddings = {
            "spatial":     spatial_emb.squeeze(0).cpu().numpy(),
            "frequency":   freq_emb.squeeze(0).cpu().numpy(),
            "discrepancy": disc_emb.squeeze(0).cpu().numpy(),
            "noise":       noise_emb.squeeze(0).cpu().numpy(),
            "fingerprint": finger_emb.squeeze(0).cpu().numpy(),
        }

        # ── 3. IDS Fusion ─────────────────────────────────────────────────
        fusion_input = {
            "spatial":     spatial_emb,
            "frequency":   freq_emb,
            "discrepancy": disc_emb,
            "noise":       noise_emb,
            "fingerprint": finger_emb,
        }

        with torch.no_grad():
            fusion_out = ids.fusion(fusion_input)

        fake_probability = float(fusion_out["fake_probability"].squeeze().cpu())
        ood_score        = float(fusion_out["OOD_score"].squeeze().cpu())
        artifact_emb     = fusion_out["artifact_embedding"].squeeze(0).cpu().numpy()  # (2560,)
        heatmap_tensor   = fusion_out["manipulation_heatmap"]                          # (1,1,224,224)

        # Heatmap to numpy (224, 224) float32
        heatmap_np = heatmap_tensor.squeeze().cpu().numpy()  # (224, 224)

        # ── 4. IDS Localization ───────────────────────────────────────────
        loc_results = ids.localization.process(heatmap_tensor)[0]
        binary_mask  = loc_results["manipulated_mask"]   # (224, 224) uint8
        blend_edges  = loc_results["blend_boundaries"]   # (224, 224) uint8
        bboxes       = loc_results["bounding_boxes"]     # List[(x,y,w,h)]

        logger.debug(
            "IDS | track=%s frame=%d | fake_prob=%.4f | ood=%.4f | bboxes=%d",
            track_id, frame_id, fake_probability, ood_score, len(bboxes),
        )

        return FrameVisualResult(
            track_id=track_id,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            fake_probability=fake_probability,
            ood_score=ood_score,
            branch_embeddings=branch_embeddings,
            artifact_embedding=artifact_emb,
            manipulation_heatmap=heatmap_np,
            manipulated_mask=binary_mask,
            blend_boundaries=blend_edges,
            bounding_boxes=bboxes,
            detection_success=True,
        )

    # ------------------------------------------------------------------
    # Batched inference
    # ------------------------------------------------------------------

    def _run_ids_batch(
        self,
        batch: List[Tuple[str, int, float, np.ndarray]],
    ) -> List[FrameVisualResult]:
        """
        Run IDS on a mini-batch of (track_id, frame_id, timestamp_ms, crop_rgb) tuples.

        Batching amortises model forward-pass overhead. The batch is processed
        as a single stacked tensor, then results are split back per-crop.

        Returns a List[FrameVisualResult] in the same order as `batch`.
        """
        import torch  # noqa: PLC0415
        ids = self._get_ids()

        results: List[FrameVisualResult] = []
        # Separate valid crops from None/invalid ones
        valid_indices: List[int]   = []
        invalid_results: Dict[int, FrameVisualResult] = {}

        spatial_list: List[np.ndarray] = []
        freq_list:    List[np.ndarray] = []

        for i, (track_id, frame_id, timestamp_ms, crop_rgb) in enumerate(batch):
            if crop_rgb is None or crop_rgb.ndim != 3 or crop_rgb.shape[2] != 3:
                invalid_results[i] = FrameVisualResult(
                    track_id=track_id, frame_id=frame_id, timestamp_ms=timestamp_ms,
                    detection_success=False,
                    failure_reason="Invalid or missing crop",
                )
                continue

            # Resize if needed
            if crop_rgb.shape[:2] != (224, 224):
                crop_rgb = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

            try:
                spatial_np = ids.normalizer.normalize(crop_rgb, is_bgr=False)
                crop_bgr   = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                freq_np    = ids.freq_analyzer.get_stacked_frequency_tensor(crop_bgr).numpy()

                spatial_list.append(spatial_np)
                freq_list.append(freq_np)
                valid_indices.append(i)
            except Exception as exc:
                logger.warning(
                    "Preprocessing failed for track=%s frame=%d: %s",
                    track_id, frame_id, exc,
                )
                invalid_results[i] = FrameVisualResult(
                    track_id=batch[i][0], frame_id=batch[i][1], timestamp_ms=batch[i][2],
                    detection_success=False, failure_reason=f"Preprocessing: {exc}",
                )

        # If no valid crops, return all as failures
        if not valid_indices:
            for i, (tid, fid, ts, _) in enumerate(batch):
                results.append(
                    invalid_results.get(i, FrameVisualResult(
                        track_id=tid, frame_id=fid, timestamp_ms=ts,
                        detection_success=False, failure_reason="No valid crops",
                    ))
                )
            return results

        # ── Stack tensors ─────────────────────────────────────────────────
        B = len(valid_indices)
        spatial_tensor = torch.from_numpy(
            np.stack(spatial_list, axis=0)  # (B, 3, 224, 224)
        ).to(ids.device)
        freq_tensor = torch.from_numpy(
            np.stack(freq_list, axis=0)     # (B, 3, 224, 224)
        ).to(ids.device)

        # ── Forward pass through all 5 branches + fusion ──────────────────
        try:
            with torch.no_grad():
                spatial_emb    = ids.branch_spatial(spatial_tensor)
                freq_emb       = ids.branch_freq(freq_tensor)
                disc_emb       = ids.branch_disc(spatial_tensor)
                noise_emb      = ids.branch_noise(spatial_tensor)
                finger_emb     = ids.branch_fingerprint(spatial_tensor)

                fusion_out = ids.fusion({
                    "spatial":     spatial_emb,
                    "frequency":   freq_emb,
                    "discrepancy": disc_emb,
                    "noise":       noise_emb,
                    "fingerprint": finger_emb,
                })

            fake_probs   = fusion_out["fake_probability"].squeeze(-1).cpu().numpy()   # (B,)
            ood_scores   = fusion_out["OOD_score"].squeeze(-1).cpu().numpy()           # (B,)
            artifact_embs = fusion_out["artifact_embedding"].cpu().numpy()             # (B, 2560)
            heatmap_t    = fusion_out["manipulation_heatmap"]                          # (B, 1, 224, 224)
            heatmaps_np  = heatmap_t.squeeze(1).cpu().numpy()                         # (B, 224, 224)

            # Per-branch embeddings (B, 512) → split later
            branch_embs_np = {
                "spatial":     spatial_emb.cpu().numpy(),
                "frequency":   freq_emb.cpu().numpy(),
                "discrepancy": disc_emb.cpu().numpy(),
                "noise":       noise_emb.cpu().numpy(),
                "fingerprint": finger_emb.cpu().numpy(),
            }

        except Exception as exc:
            logger.warning("Batch forward pass failed: %s — falling back to per-crop.", exc)
            # Fallback: run individually
            all_results_dict: Dict[int, FrameVisualResult] = dict(invalid_results)
            for j, idx in enumerate(valid_indices):
                tid, fid, ts, crp = batch[idx]
                all_results_dict[idx] = self.analyze_crop(tid, fid, ts, crp)
            return [all_results_dict[i] for i in range(len(batch))]

        # ── Localization (per-item in batch) ──────────────────────────────
        loc_results_list = ids.localization.process(heatmap_t)

        # ── Build per-crop FrameVisualResult ──────────────────────────────
        valid_frame_results: Dict[int, FrameVisualResult] = {}
        for j, idx in enumerate(valid_indices):
            tid, fid, ts, _ = batch[idx]
            loc = loc_results_list[j]

            valid_frame_results[idx] = FrameVisualResult(
                track_id=tid,
                frame_id=fid,
                timestamp_ms=ts,
                fake_probability=float(fake_probs[j]),
                ood_score=float(ood_scores[j]),
                branch_embeddings={k: v[j] for k, v in branch_embs_np.items()},
                artifact_embedding=artifact_embs[j],
                manipulation_heatmap=heatmaps_np[j],
                manipulated_mask=loc["manipulated_mask"],
                blend_boundaries=loc["blend_boundaries"],
                bounding_boxes=loc["bounding_boxes"],
                detection_success=True,
            )

        # Combine valid and invalid results in original order
        all_results: Dict[int, FrameVisualResult] = {**invalid_results, **valid_frame_results}
        return [all_results[i] for i in range(len(batch))]

    # ------------------------------------------------------------------
    # Public API: process entire Layer 1 result
    # ------------------------------------------------------------------

    def analyze(
        self,
        layer1_result,   # VideoPreprocessingResult — avoid circular import
    ) -> BranchAResult:
        """
        Run Branch A on the full VDS Layer 1 output.

        Iterates over all FramePackets, collects tracked face crops, groups
        them into mini-batches, runs IDS on each batch, and assembles the
        results into a per-track timeline.

        Args:
            layer1_result : VideoPreprocessingResult from VideoLayer1Pipeline.

        Returns:
            BranchAResult — per-track timelines of FrameVisualResult objects.
        """
        logger.info(
            "═══════════════════════════════════════════════════════════════"
        )
        logger.info(
            "VDS Layer 2 Branch A START | frames=%d | tracks=%s",
            len(layer1_result.frames),
            layer1_result.unique_track_ids,
        )
        logger.info(
            "═══════════════════════════════════════════════════════════════"
        )

        t_start = time.perf_counter()
        warnings: List[str] = []

        # Per-track result timelines
        frame_results: Dict[str, List[FrameVisualResult]] = {
            tid: [] for tid in layer1_result.unique_track_ids
        }

        # Collect all (track_id, frame_id, timestamp_ms, crop_rgb) tuples
        all_crops: List[Tuple[str, int, float, Optional[np.ndarray]]] = []
        for fp in layer1_result.frames:
            for tf in fp.tracked_faces:
                all_crops.append((
                    tf.track_id,
                    fp.frame_id,
                    fp.timestamp_ms,
                    tf.aligned_crop,
                ))

        n_total = len(all_crops)
        logger.info("Total face-frame pairs to process: %d", n_total)

        if n_total == 0:
            logger.warning("No tracked faces found in Layer 1 result — Branch A is empty.")
            warnings.append("No tracked faces in Layer 1 result.")
            return BranchAResult(
                frame_results=frame_results,
                n_frames_processed=0,
                n_faces_total=0,
                n_faces_successful=0,
                processing_warnings=warnings,
            )

        # Process in mini-batches
        n_successful = 0
        n_batches    = 0

        for batch_start in range(0, n_total, self.batch_size):
            batch = all_crops[batch_start: batch_start + self.batch_size]
            batch_results = self._run_ids_batch(batch)

            for result in batch_results:
                if result.track_id not in frame_results:
                    frame_results[result.track_id] = []
                frame_results[result.track_id].append(result)
                if result.detection_success:
                    n_successful += 1

            n_batches += 1
            processed_so_far = min(batch_start + self.batch_size, n_total)
            if n_batches % 5 == 0 or processed_so_far == n_total:
                logger.info(
                    "Branch A progress: %d/%d face-frames processed …",
                    processed_so_far, n_total,
                )

        # Sort each track's results by frame_id
        for tid in frame_results:
            frame_results[tid].sort(key=lambda r: r.frame_id)

        elapsed = time.perf_counter() - t_start

        logger.info(
            "═══════════════════════════════════════════════════════════════"
        )
        logger.info(
            "VDS Layer 2 Branch A COMPLETE | elapsed=%.2fs | "
            "total=%d | successful=%d | tracks=%d",
            elapsed, n_total, n_successful, len(frame_results),
        )
        logger.info(
            "═══════════════════════════════════════════════════════════════"
        )

        branch_a = BranchAResult(
            frame_results=frame_results,
            n_frames_processed=len(layer1_result.frames),
            n_faces_total=n_total,
            n_faces_successful=n_successful,
            processing_warnings=warnings,
        )

        # Log per-track summary
        for tid in branch_a.unique_track_ids:
            mean_p = branch_a.mean_fake_probability(tid)
            if mean_p is not None:
                verdict = "⚠ FAKE" if mean_p > 0.5 else "✓ REAL"
                logger.info(
                    "  %s  →  mean_fake_prob=%.4f  %s",
                    tid, mean_p, verdict,
                )

        return branch_a

    # ------------------------------------------------------------------
    # Convenience: quick single-frame entry point
    # ------------------------------------------------------------------

    def analyze_frame(
        self,
        frame_packet,   # FramePacket
    ) -> List[FrameVisualResult]:
        """
        Analyze all tracked faces in a single FramePacket.

        Returns:
            List of FrameVisualResult — one per tracked face in this frame.
            Returns [] if the frame has no tracked faces.
        """
        if not frame_packet.tracked_faces:
            return []

        batch = [
            (tf.track_id, frame_packet.frame_id, frame_packet.timestamp_ms, tf.aligned_crop)
            for tf in frame_packet.tracked_faces
        ]
        return self._run_ids_batch(batch)

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """
        Release IDS model weights from memory.
        Call when Branch A will not be used again in the current process.
        """
        if self._ids is not None:
            try:
                import torch  # noqa: PLC0415
                del self._ids
                self._ids = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("FrameVisualAnalyzer: IDS components unloaded.")
            except Exception as exc:
                logger.warning("Error during unload: %s", exc)

    def __repr__(self) -> str:
        return (
            f"FrameVisualAnalyzer("
            f"device={self.device!r}, "
            f"batch_size={self.batch_size}, "
            f"loc_threshold={self.localization_threshold})"
        )
