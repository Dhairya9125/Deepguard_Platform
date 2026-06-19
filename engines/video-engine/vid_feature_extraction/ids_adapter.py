"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.ids_adapter
Layer   : Layer 2 — Branch A, Frame-Level Visual Analysis
Task    : IDS import adapter & component loader

Purpose:
    The IDS (Image Detection Subsystem) lives in `engines/image-engine/`.
    It is not installed as a Python package — it is a sibling directory.
    This adapter handles sys.path management so that all IDS modules can be
    imported cleanly from `engines/video-engine/` without any repo-wide
    packaging changes.

    It mirrors the pattern already established in `face_tracker.py` for
    the RetinaFace backend, generalising it to the full IDS stack.

    This is the SINGLE place where the cross-engine import seam lives.
    All other Branch A code imports from this module, not directly from
    `engines/image-engine`.

Usage:
    from vid_feature_extraction.ids_adapter import load_ids_components, IDSComponents

    ids = load_ids_components(device="cpu")
    # ids.normalizer, ids.freq_analyzer, ids.branch_spatial, … all ready

    # With optional fine-tuned checkpoint weights:
    ids = load_ids_components(device="cpu", weights_dir=Path("checkpoints/ids/"))
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IDS path resolution
# ---------------------------------------------------------------------------

def _resolve_ids_path() -> Path:
    """
    Return the absolute path to `engines/image-engine/`.

    Searches by walking up from this file's location to find the
    `engines/image-engine` sibling directory. Raises RuntimeError if not found.
    """
    # This file lives at: engines/video-engine/feature_extraction/ids_adapter.py
    # image-engine is at: engines/image-engine/
    here = Path(__file__).resolve()

    # Walk up: feature_extraction → video-engine → engines
    engines_dir = here.parents[2]  # engines/
    ids_path    = engines_dir / "image-engine"

    if ids_path.exists():
        logger.debug("IDS path resolved: %s", ids_path)
        return ids_path

    # Fallback: search from cwd
    cwd_engines = Path.cwd() / "engines" / "image-engine"
    if cwd_engines.exists():
        logger.debug("IDS path resolved via cwd: %s", cwd_engines)
        return cwd_engines

    raise RuntimeError(
        f"[IDSAdapter] Cannot locate engines/image-engine.\n"
        f"Searched:\n  {ids_path}\n  {cwd_engines}\n"
        "Ensure you are running from the repository root."
    )


def _ensure_ids_on_path() -> None:
    """Append the IDS root to sys.path if not already present."""
    ids_path = str(_resolve_ids_path())
    if ids_path not in sys.path:
        sys.path.insert(0, ids_path)
        logger.debug("Appended to sys.path: %s", ids_path)


# ---------------------------------------------------------------------------
# IDSComponents container
# ---------------------------------------------------------------------------

@dataclass
class IDSComponents:
    """
    All IDS model components needed for Branch A inference.

    All models are already in `.eval()` mode and on the correct device
    when this dataclass is returned by `load_ids_components()`.

    Attributes:
        normalizer         : ImageNormalizer — produces (3,224,224) CHW float32.
        freq_analyzer      : FrequencyAnalyzer — produces (3,H,W) FFT/DCT/Wavelet stack.
        branch_spatial     : ClipSpatialBranch — CLIP ViT-L/14 → (B, 512) embedding.
        branch_freq        : EfficientNetFrequencyBranch → (B, 512) embedding.
        branch_disc        : DiscrepancyBranch — Siamese ResNet18 → (B, 512).
        branch_noise       : NoiseResidualBranch — SRM + ResNet18 → (B, 512).
        branch_fingerprint : FingerprintBranch — Swin-T → (B, 512).
        fusion             : IDSFusionEngine → fake_prob, heatmap, OOD, artifact_emb.
        localization       : LocalizationEngine — heatmap → binary mask + bboxes.
        device             : torch.device the models are loaded on.
    """
    normalizer:          object   # ImageNormalizer
    freq_analyzer:       object   # FrequencyAnalyzer
    branch_spatial:      object   # ClipSpatialBranch (nn.Module)
    branch_freq:         object   # EfficientNetFrequencyBranch (nn.Module)
    branch_disc:         object   # DiscrepancyBranch (nn.Module)
    branch_noise:        object   # NoiseResidualBranch (nn.Module)
    branch_fingerprint:  object   # FingerprintBranch (nn.Module)
    fusion:              object   # IDSFusionEngine (nn.Module)
    localization:        object   # LocalizationEngine
    device:              object   # torch.device


# ---------------------------------------------------------------------------
# Weight loading helper
# ---------------------------------------------------------------------------

def _try_load_checkpoint(model, checkpoint_path: Path, model_name: str) -> None:
    """
    Attempt to load a fine-tuned checkpoint into `model` in-place.
    Logs a warning and continues with pretrained weights if loading fails.

    Args:
        model           : A torch.nn.Module instance.
        checkpoint_path : Path to a .pt / .pth state-dict file.
        model_name      : Human-readable name for logging.
    """
    try:
        import torch  # noqa: PLC0415
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        logger.info(
            "Loaded fine-tuned weights for %s from %s", model_name, checkpoint_path
        )
    except Exception as exc:
        logger.warning(
            "Could not load checkpoint for %s from %s — using pretrained weights. "
            "Reason: %s",
            model_name, checkpoint_path, exc,
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def load_ids_components(
    device:       str            = "cpu",
    weights_dir:  Optional[Path] = None,
    proj_dim:     int            = 512,
    loc_threshold: float         = 0.6,
) -> IDSComponents:
    """
    Import and initialise all IDS components for Branch A inference.

    This function:
      1. Ensures `engines/image-engine` is on sys.path.
      2. Imports all IDS modules (lazy — first call only incurs this cost).
      3. Instantiates each model with pretrained weights.
      4. Optionally loads fine-tuned checkpoints from `weights_dir`.
      5. Sets all models to `.eval()` mode on the target device.
      6. Returns an `IDSComponents` dataclass.

    Args:
        device      : PyTorch device string. Default: 'cpu'.
                      Use 'cuda' if a CUDA-capable GPU is available.
        weights_dir : Optional directory containing fine-tuned checkpoint .pt files.
                      Expected filenames (all optional, falls back to pretrained):
                        spatial_branch.pt, frequency_branch.pt,
                        discrepancy_branch.pt, noise_branch.pt,
                        fingerprint_branch.pt, fusion_engine.pt
        proj_dim    : Embedding dimension for all 5 branches. Default: 512.
        loc_threshold: LocalizationEngine binary mask threshold. Default: 0.6.

    Returns:
        IDSComponents — ready for inference.

    Raises:
        RuntimeError  : If IDS path cannot be resolved.
        ImportError   : If required IDS packages are not installed.
    """
    _ensure_ids_on_path()

    logger.info("Loading IDS components | device=%s | proj_dim=%d", device, proj_dim)

    # ── Torch import ────────────────────────────────────────────────────────
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for Branch A.\n"
            "Install with:  pip install torch torchvision"
        ) from exc

    dev = torch.device(device)

    # ── IDS imports ─────────────────────────────────────────────────────────
    try:
        from vid_preprocessing.image_normalizer   import ImageNormalizer           # noqa: PLC0415
        from vid_preprocessing.frequency_analyzer import FrequencyAnalyzer         # noqa: PLC0415
        from vid_feature_extraction.spatial_branch    import ClipSpatialBranch     # noqa: PLC0415
        from vid_feature_extraction.frequency_branch  import EfficientNetFrequencyBranch  # noqa: PLC0415
        from vid_feature_extraction.discrepancy_branch import DiscrepancyBranch    # noqa: PLC0415
        from vid_feature_extraction.noise_branch      import NoiseResidualBranch   # noqa: PLC0415
        from vid_feature_extraction.fingerprint_branch import FingerprintBranch    # noqa: PLC0415
        from fusion.ids_fusion_engine             import IDSFusionEngine        # noqa: PLC0415
        from vid_localization.localization_engine     import LocalizationEngine     # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            f"[IDSAdapter] Failed to import IDS module: {exc}\n"
            "Ensure engines/image-engine/ exists and its requirements are installed.\n"
            "IDS requirements: pip install -r engines/image-engine/requirements.txt"
        ) from exc

    # ── Preprocessing (stateless, no GPU) ───────────────────────────────────
    normalizer    = ImageNormalizer(to_chw=True)
    freq_analyzer = FrequencyAnalyzer(log_scale=True)
    logger.info("IDS preprocessing components ready.")

    # ── Feature extraction branches ──────────────────────────────────────────
    logger.info("Loading IDS feature extraction branches …")

    branch_spatial     = ClipSpatialBranch(proj_dim=proj_dim, freeze_backbone=True).to(dev).eval()
    branch_freq        = EfficientNetFrequencyBranch(proj_dim=proj_dim, pretrained=True).to(dev).eval()
    branch_disc        = DiscrepancyBranch(proj_dim=proj_dim, pretrained=True).to(dev).eval()
    branch_noise       = NoiseResidualBranch(proj_dim=proj_dim, pretrained=True).to(dev).eval()
    branch_fingerprint = FingerprintBranch(proj_dim=proj_dim, pretrained=True).to(dev).eval()

    logger.info("All 5 IDS branches loaded.")

    # ── Fusion engine ────────────────────────────────────────────────────────
    fusion = IDSFusionEngine(embed_dim=proj_dim, num_branches=5).to(dev).eval()
    logger.info("IDSFusionEngine loaded.")

    # ── Localization (CPU, stateless) ─────────────────────────────────────────
    localization = LocalizationEngine(threshold=loc_threshold)
    logger.info("LocalizationEngine ready (threshold=%.2f).", loc_threshold)

    # ── Optional fine-tuned checkpoints ──────────────────────────────────────
    if weights_dir is not None:
        weights_dir = Path(weights_dir)
        if weights_dir.exists():
            logger.info("Loading fine-tuned checkpoints from: %s", weights_dir)
            checkpoint_map = {
                "spatial_branch.pt":     branch_spatial,
                "frequency_branch.pt":   branch_freq,
                "discrepancy_branch.pt": branch_disc,
                "noise_branch.pt":       branch_noise,
                "fingerprint_branch.pt": branch_fingerprint,
                "fusion_engine.pt":      fusion,
            }
            for fname, model in checkpoint_map.items():
                ckpt_path = weights_dir / fname
                if ckpt_path.exists():
                    _try_load_checkpoint(model, ckpt_path, fname.replace(".pt", ""))
        else:
            logger.warning(
                "weights_dir specified but does not exist: %s — using pretrained only.",
                weights_dir,
            )

    logger.info(
        "IDSComponents ready | device=%s | branches=5 | fusion=ready | loc=ready", device
    )

    return IDSComponents(
        normalizer=normalizer,
        freq_analyzer=freq_analyzer,
        branch_spatial=branch_spatial,
        branch_freq=branch_freq,
        branch_disc=branch_disc,
        branch_noise=branch_noise,
        branch_fingerprint=branch_fingerprint,
        fusion=fusion,
        localization=localization,
        device=dev,
    )
