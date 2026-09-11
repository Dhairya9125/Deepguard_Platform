"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.semantic_signals
Layer   : Layer 2 — Temporal Feature Extraction
Branch  : Branch G — Scene Semantic Consistency

Pure-NumPy algorithms for face-to-background lighting discrepancy, temporal
background warping inconsistency, and global frame-flickering instability.
"""

from __future__ import annotations

from typing import List, Tuple
import cv2
import numpy as np


def compute_lighting_discrepancy(
    face_crops: List[np.ndarray],
    full_frames: List[np.ndarray],
    bboxes: List[Tuple[int, int, int, int]]
) -> float:
    """
    Evaluate luminance correlation between the face crop and the background.
    
    Computes the variance of the ratio (face_luminance / background_luminance) over time.
    Real videos show highly coupled lighting changes; fakes exhibit decoupled lighting.
    
    Args:
        face_crops: List of face crops corresponding to the track.
        full_frames: List of full video frames.
        bboxes: Bounding box tuples (x1, y1, x2, y2) of the face.

    Returns:
        float: Variance of the face-to-background luminance ratio.
    """
    T = len(full_frames)
    if T < 2 or len(face_crops) != T or len(bboxes) != T:
        return 0.0

    ratios = []
    for i in range(T):
        frame = full_frames[i]
        crop = face_crops[i]
        bbox = bboxes[i]

        if frame is None or crop is None or crop.size == 0 or frame.size == 0:
            continue

        # 1. Face luminance (Y channel via grayscale)
        face_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        face_lum = np.mean(face_gray)

        # 2. Background luminance (masking out the face region)
        H, W = frame.shape[:2]
        mask = np.ones((H, W), dtype=np.uint8)
        
        x1, y1, x2, y2 = bbox
        x1 = np.clip(x1, 0, W - 1)
        x2 = np.clip(x2, 0, W - 1)
        y1 = np.clip(y1, 0, H - 1)
        y2 = np.clip(y2, 0, H - 1)
        
        mask[y1:y2, x1:x2] = 0

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        bg_pixels = frame_gray[mask == 1]

        if len(bg_pixels) > 0:
            bg_lum = np.mean(bg_pixels)
        else:
            bg_lum = np.mean(frame_gray)  # fallback

        ratio = face_lum / (bg_lum + 1e-5)
        ratios.append(ratio)

    if len(ratios) < 2:
        return 0.0

    return float(np.var(ratios))


def compute_background_inconsistency(
    full_frames: List[np.ndarray],
    bboxes: List[Tuple[int, int, int, int]]
) -> float:
    """
    Compute temporal MSE of background regions, masking out face bounding boxes.
    
    Detects local warping, blending seam ghosts, and spatial inconsistencies.
    
    Args:
        full_frames: List of full frames.
        bboxes: Face bounding boxes to exclude.

    Returns:
        float: Mean temporal squared difference (MSE) in background zones.
    """
    T = len(full_frames)
    if T < 2 or len(bboxes) != T:
        return 0.0

    mses = []
    for i in range(T - 1):
        f1 = full_frames[i]
        f2 = full_frames[i + 1]
        b1 = bboxes[i]
        b2 = bboxes[i + 1]

        if f1 is None or f2 is None or f1.shape != f2.shape:
            continue

        H, W = f1.shape[:2]
        mask = np.ones((H, W), dtype=np.uint8)

        # Exclude union of bounding boxes in consecutive frames
        for bx in (b1, b2):
            x1, y1, x2, y2 = bx
            x1 = np.clip(x1, 0, W - 1)
            x2 = np.clip(x2, 0, W - 1)
            y1 = np.clip(y1, 0, H - 1)
            y2 = np.clip(y2, 0, H - 1)
            mask[y1:y2, x1:x2] = 0

        if np.sum(mask) == 0:
            continue

        # Grayscale difference
        g1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)

        mse = np.mean((g1[mask == 1].astype(np.float32) - g2[mask == 1].astype(np.float32)) ** 2)
        mses.append(mse)

    if not mses:
        return 0.0

    return float(np.mean(mses))


def compute_environment_instability(full_frames: List[np.ndarray]) -> float:
    """
    Measure global frame flickering (variance of temporal frame differences).
    
    Args:
        full_frames: List of full frames.

    Returns:
        float: Variance of temporal frame intensity differences.
    """
    T = len(full_frames)
    if T < 2:
        return 0.0

    intensities = []
    for frame in full_frames:
        if frame is not None and frame.size > 0:
            intensities.append(float(np.mean(frame)))

    if len(intensities) < 2:
        return 0.0

    diffs = np.diff(intensities)
    return float(np.var(diffs))
