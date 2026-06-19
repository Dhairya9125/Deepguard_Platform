"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.motion_signals
Layer   : Layer 2 — Temporal Feature Extraction
Branch  : Branch C — Temporal Motion Modeling

NumPy-based forensic motion signals.
"""

from __future__ import annotations

from typing import List, Tuple
import cv2
import numpy as np


def compute_jitter_entropy(landmarks: np.ndarray) -> float:
    """
    Calculate the Shannon entropy of landmark displacements between consecutive frames.
    High entropy indicates erratic speed/displacement variations (unstable/jittery motion).

    Args:
        landmarks: np.ndarray of shape (T, N, 3) where N is the number of landmarks.

    Returns:
        float: Jitter entropy value.
    """
    if landmarks is None or len(landmarks) < 3:
        return 0.0

    # Ensure shape is 3D: (T, N, 3)
    if landmarks.ndim != 3:
        return 0.0

    # Compute displacements (velocities) between consecutive frames
    d = np.diff(landmarks, axis=0)  # (T-1, N, 3)
    d_norm = np.linalg.norm(d, axis=2)  # (T-1, N)
    mean_disp = np.mean(d_norm, axis=1)  # (T-1,)

    if len(mean_disp) < 2:
        return 0.0

    # Safely handle constant displacements to avoid numpy histogram bin errors on constant data
    if np.max(mean_disp) - np.min(mean_disp) < 1e-5:
        return 0.0

    # Discretize displacements into a histogram to compute Shannon entropy
    # Use 10 bins over the range of displacements
    hist, _ = np.histogram(mean_disp, bins=10, density=True)
    probs = hist / (np.sum(hist) + 1e-9)
    entropy = -np.sum(probs * np.log2(probs + 1e-9))

    # Keep value positive and return
    return float(max(0.0, entropy))


def compute_velocity_variance(landmarks: np.ndarray) -> float:
    """
    Compute the temporal variance of face landmark displacement velocity.
    Detects sudden starts/stops or extreme motion spikes.

    Args:
        landmarks: np.ndarray of shape (T, N, 3)

    Returns:
        float: Velocity variance.
    """
    if landmarks is None or len(landmarks) < 2:
        return 0.0

    if landmarks.ndim != 3:
        return 0.0

    d = np.diff(landmarks, axis=0)  # (T-1, N, 3)
    d_norm = np.linalg.norm(d, axis=2)  # (T-1, N)
    mean_disp = np.mean(d_norm, axis=1)  # (T-1,)

    if len(mean_disp) == 0:
        return 0.0

    return float(np.var(mean_disp))


def compute_blink_anomalies(
    ear_l: np.ndarray, 
    ear_r: np.ndarray, 
    fps: float
) -> Tuple[float, float]:
    """
    Measure blink duration anomalies and Left-Right Eye Aspect Ratio (EAR) asymmetry.

    Args:
        ear_l: np.ndarray of shape (T,) left eye EAR.
        ear_r: np.ndarray of shape (T,) right eye EAR.
        fps: float, target video frame rate.

    Returns:
        Tuple[float, float]: (blink_asymmetry, blink_duration_anomaly)
    """
    if ear_l is None or ear_r is None or len(ear_l) == 0 or len(ear_r) == 0:
        return 0.0, 1.0  # Suspect if no data

    # 1. Blink Asymmetry: Mean absolute difference between left and right EARs
    asymmetry = float(np.mean(np.abs(ear_l - ear_r)))

    # 2. Blink Duration Anomaly: Detect blinks as dips in EAR below threshold
    mean_ear = 0.5 * (ear_l + ear_r)
    blink_threshold = 0.22
    in_blink = mean_ear < blink_threshold

    blink_durations = []
    current_run = 0
    for val in in_blink:
        if val:
            current_run += 1
        else:
            if current_run > 0:
                blink_durations.append(current_run)
                current_run = 0
    if current_run > 0:
        blink_durations.append(current_run)

    if not blink_durations:
        # Deepfakes sometimes never blink. Flag this as high anomaly (1.0)
        duration_anomaly = 1.0
    else:
        avg_dur_frames = np.mean(blink_durations)
        video_fps = fps if fps > 0 else 30.0
        avg_dur_s = avg_dur_frames / video_fps

        # Normal blink duration is 0.1s to 0.4s
        if avg_dur_s < 0.1:
            # Too fast (unnatural flicker/incomplete blink)
            duration_anomaly = float(min(1.0, (0.1 - avg_dur_s) / 0.1))
        elif avg_dur_s > 0.4:
            # Too slow (unnatural eye closure/sleepy state)
            duration_anomaly = float(min(1.0, (avg_dur_s - 0.4) / 0.6))
        else:
            duration_anomaly = 0.0

    return asymmetry, duration_anomaly


def compute_frame_inconsistency(crops: List[np.ndarray]) -> float:
    """
    Compute frame-to-frame Mean Squared Error (MSE) of face crops to measure
    abrupt pixel-level visual changes/inconsistencies.

    Args:
        crops: List of (H, W, 3) RGB crops.

    Returns:
        float: Frame inconsistency score in range [0, 1].
    """
    # Filter out None crops
    valid_crops = [c for c in crops if c is not None]
    if len(valid_crops) < 2:
        return 0.0

    diffs = []
    for i in range(len(valid_crops) - 1):
        c1 = valid_crops[i].astype(np.float32)
        c2 = valid_crops[i+1].astype(np.float32)

        # Ensure shapes match
        if c1.shape != c2.shape:
            c2 = cv2.resize(c2, (c1.shape[1], c1.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Compute Normalized MSE
        mse = np.mean((c1 - c2) ** 2)
        normalized_mse = mse / (255.0 ** 2)
        diffs.append(normalized_mse)

    return float(np.mean(diffs))
