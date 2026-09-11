"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.continuity_signals
Layer   : Layer 2 — Temporal Feature Extraction
Branch  : Branch F — Identity Continuity Tracking

Pure-NumPy algorithms for face identity representation drift, facial geometric
proportions stability, and bounding box physical trajectory continuity.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np


def compute_identity_drift(embeddings: np.ndarray) -> Tuple[float, float]:
    """
    Calculate identity representation drift variance and max distance relative to
    the face track's global average embedding (anchor).
    
    Args:
        embeddings: (T, D) array of feature embeddings over time.

    Returns:
        Tuple[float, float]: (max_cosine_distance_drift, cosine_distance_variance)
    """
    T = len(embeddings)
    if T < 2:
        return 0.0, 0.0

    # 1. Compute anchor embedding (global mean)
    anchor = np.mean(embeddings, axis=0)
    anchor_norm_val = np.linalg.norm(anchor)
    if anchor_norm_val < 1e-6:
        return 0.0, 0.0
    anchor_norm = anchor / anchor_norm_val

    # 2. Normalize all embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1e-6
    emb_norm = embeddings / norms

    # 3. Compute cosine distances: 1.0 - CosineSimilarity
    cosine_sims = np.dot(emb_norm, anchor_norm)
    distances = 1.0 - cosine_sims

    max_drift = float(np.max(distances))
    drift_var = float(np.var(distances))
    
    return max_drift, drift_var


def compute_geometry_instability(landmarks: np.ndarray) -> float:
    """
    Evaluate rigid geometry proportions stability of the face track.
    
    Computes two stable skull proportions over time:
      Ratio 1: Interpupillary distance (between eyes) / Face width
      Ratio 2: Nose-to-chin distance / Face height (forehead-to-chin)
      
    Returns the average Coefficient of Variation (std / mean) of these ratios.
    
    Args:
        landmarks: (T, 478, 3) dense MediaPipe FaceMesh coordinate timeline.

    Returns:
        float: Combined coefficient of variation representing instability.
    """
    T = len(landmarks)
    if T < 3:
        return 0.0

    # MediaPipe FaceMesh landmark indices:
    # Right iris center: 468, Left iris center: 473
    # Face width limits (cheek to cheek): 234 (right), 454 (left)
    # Nose tip: 1, Chin bottom: 152, Forehead center: 10
    
    # Extract coordinates in 2D space (x, y)
    coords_2d = landmarks[:, :, :2]

    # Ratio 1: Eye distance / Face width
    eye_dist = np.linalg.norm(coords_2d[:, 473] - coords_2d[:, 468], axis=1)
    face_width = np.linalg.norm(coords_2d[:, 234] - coords_2d[:, 454], axis=1)
    face_width[face_width < 1e-5] = 1e-5
    ratio1 = eye_dist / face_width

    # Ratio 2: Nose-to-chin / Face height
    nose_to_chin = np.linalg.norm(coords_2d[:, 1] - coords_2d[:, 152], axis=1)
    face_height = np.linalg.norm(coords_2d[:, 10] - coords_2d[:, 152], axis=1)
    face_height[face_height < 1e-5] = 1e-5
    ratio2 = nose_to_chin / face_height

    # Calculate Coefficient of Variation (CV = std / mean) for each ratio
    mean1 = np.mean(ratio1)
    std1 = np.std(ratio1)
    cv1 = std1 / mean1 if mean1 > 1e-6 else 0.0

    mean2 = np.mean(ratio2)
    std2 = np.std(ratio2)
    cv2 = std2 / mean2 if mean2 > 1e-6 else 0.0

    # Combined instability is the average CV
    instability = float((cv1 + cv2) / 2.0)
    return instability


def compute_trajectory_jumps(
    bboxes: List[Tuple[int, int, int, int]], 
    fps: float, 
    frame_width: int,
    jump_threshold_ratio: float = 0.15
) -> Tuple[float, int]:
    """
    Check bounding box trajectory for physical continuity errors (sudden displacement jumps).
    
    Args:
        bboxes: List of (x1, y1, x2, y2) bounding box coordinate tuples.
        fps: Target frame rate.
        frame_width: Overall video frame width.
        jump_threshold_ratio: Fraction of frame size allowed as physical displacement per frame.

    Returns:
        Tuple[float, int]: (max_jump_displacement_px, trajectory_jump_count)
    """
    T = len(bboxes)
    if T < 2 or frame_width <= 0:
        return 0.0, 0

    # BBox centers: cx = (x1 + x2) / 2, cy = (y1 + y2) / 2
    centers = []
    for (x1, y1, x2, y2) in bboxes:
        centers.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
    centers = np.array(centers)

    # Displacement between consecutive frames
    displacements = np.linalg.norm(centers[1:] - centers[:-1], axis=1)

    max_jump = float(np.max(displacements))
    
    # Threshold for jump (e.g. 15% of frame width)
    threshold = jump_threshold_ratio * frame_width
    jumps = int(np.sum(displacements > threshold))

    return max_jump, jumps
