"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.rppg_signals
Layer   : Layer 2 — Temporal Feature Extraction
Branch  : Branch E — rPPG Biological Signal Analysis

Pure-NumPy algorithms for remote Photoplethysmography (rPPG) ROI extraction,
pulse estimation, and biological consistency checks.
"""

from __future__ import annotations

from typing import Dict, Tuple
import cv2
import numpy as np


def extract_skin_rois(face_crop: np.ndarray, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract spatial RGB averages over time for Forehead, Left Cheek, and Right Cheek.
    
    Args:
        face_crop: (H, W, 3) RGB uint8 crop.
        landmarks: (478, 3) landmarks coordinates in pixel space of the face_crop.

    Returns:
        Dict[str, np.ndarray]: ROI keys mapped to (3,) RGB average values.
    """
    rois = {"forehead": np.zeros(3), "left_cheek": np.zeros(3), "right_cheek": np.zeros(3)}
    if face_crop is None or landmarks is None or len(landmarks) < 400:
        return rois

    H, W = face_crop.shape[:2]

    # MediaPipe indices defining regional masks
    # Forehead: indices centered above brows
    forehead_indices = [54, 67, 103, 109, 10, 338, 297, 332, 284, 251]
    # Left Cheek: cheekbone region below left eye
    left_cheek_indices = [116, 118, 123, 50, 205, 207, 101]
    # Right Cheek: cheekbone region below right eye
    right_cheek_indices = [345, 347, 352, 280, 425, 427, 330]

    for roi_name, indices in [
        ("forehead", forehead_indices),
        ("left_cheek", left_cheek_indices),
        ("right_cheek", right_cheek_indices),
    ]:
        # Clip indices coordinates to frame boundaries
        pts = landmarks[indices, :2].copy()
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
        pts = pts.astype(np.int32)

        # Create mask
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)

        if np.sum(mask) > 0:
            mean_rgb = face_crop[mask == 1].mean(axis=0)
            rois[roi_name] = mean_rgb

    return rois


def _bandpass_filter_fft(signal: np.ndarray, low: float, high: float, fps: float) -> np.ndarray:
    """Zero-phase bandpass filter using FFT to avoid scipy filter version dependencies."""
    T = len(signal)
    if T < 4:
        return signal
    fft_vals = np.fft.rfft(signal)
    fft_freqs = np.fft.rfftfreq(T, d=1.0/fps)
    
    # Mask frequencies outside [low, high]
    mask = (fft_freqs >= low) & (fft_freqs <= high)
    fft_vals[~mask] = 0.0
    
    filtered = np.fft.irfft(fft_vals, n=T)
    return filtered


def compute_rppg_pos(rgb_signal: np.ndarray, fps: float) -> np.ndarray:
    """
    POS (Plane-Orthogonal-to-Skin) projection algorithm for rPPG.
    
    Args:
        rgb_signal: (T, 3) array representing channel-wise (R, G, B) averages.
        fps: Target video frame rate.

    Returns:
        np.ndarray: Filtered 1D PPG signal waveform of shape (T,).
    """
    T = len(rgb_signal)
    if T < 8 or fps <= 0:
        return np.zeros(T, dtype=np.float32)

    win_len = min(32, T)
    pos_signal = np.zeros(T, dtype=np.float32)
    weight = np.zeros(T, dtype=np.float32)

    for i in range(0, T - win_len + 1):
        window = rgb_signal[i : i + win_len]
        mean_c = np.mean(window, axis=0)
        if np.any(mean_c < 1e-5):
            continue

        # Temporal Normalization
        cn = window / mean_c

        # Orthogonal projections: S1 = R - G, S2 = R + G - 2B
        s1 = cn[:, 0] - cn[:, 1]
        s2 = cn[:, 0] + cn[:, 1] - 2.0 * cn[:, 2]

        std1 = np.std(s1)
        std2 = np.std(s2)

        if std2 < 1e-6:
            h_win = s1
        else:
            alpha = std1 / std2
            h_win = s1 - alpha * s2

        h_win = h_win - np.mean(h_win)
        pos_signal[i : i + win_len] += h_win
        weight[i : i + win_len] += 1.0

    pos_signal[weight > 0] /= weight[weight > 0]
    
    # Filter within heart rate range [0.7 Hz, 4.0 Hz] (approx 42 - 240 BPM)
    pos_signal = _bandpass_filter_fft(pos_signal, 0.7, 4.0, fps)
    return pos_signal


def compute_rppg_chrom(rgb_signal: np.ndarray, fps: float) -> np.ndarray:
    """
    CHROM (Chrominance-based method) algorithm for rPPG.
    
    Args:
        rgb_signal: (T, 3) channel-wise (R, G, B) averages.
        fps: Target frame rate.

    Returns:
        np.ndarray: Filtered 1D PPG signal.
    """
    T = len(rgb_signal)
    if T < 8 or fps <= 0:
        return np.zeros(T, dtype=np.float32)

    win_len = min(32, T)
    chrom_signal = np.zeros(T, dtype=np.float32)
    weight = np.zeros(T, dtype=np.float32)

    for i in range(0, T - win_len + 1):
        window = rgb_signal[i : i + win_len]
        mean_c = np.mean(window, axis=0)
        if np.any(mean_c < 1e-5):
            continue

        cn = window / mean_c

        # X = 3R - 2G, Y = 1.5R + G - 1.5B
        x = 3.0 * cn[:, 0] - 2.0 * cn[:, 1]
        y = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]

        std_x = np.std(x)
        std_y = np.std(y)

        if std_y < 1e-6:
            h_win = x
        else:
            alpha = std_x / std_y
            h_win = x - alpha * y

        h_win = h_win - np.mean(h_win)
        chrom_signal[i : i + win_len] += h_win
        weight[i : i + win_len] += 1.0

    chrom_signal[weight > 0] /= weight[weight > 0]
    chrom_signal = _bandpass_filter_fft(chrom_signal, 0.7, 4.0, fps)
    return chrom_signal


def estimate_heart_rate(pulse_signal: np.ndarray, fps: float) -> Tuple[float, float]:
    """
    Estimate heart rate in beats-per-minute (BPM) and its Signal-to-Noise Ratio (SNR).
    
    Args:
        pulse_signal: 1D pulse timeline.
        fps: Target frame rate.

    Returns:
        Tuple[float, float]: (heart_rate_bpm, snr)
    """
    T = len(pulse_signal)
    if T < 8 or fps <= 0:
        return 0.0, 0.0

    # FFT zero-padding to increase frequency resolution
    n_fft = max(256, T * 4)
    fft_vals = np.abs(np.fft.rfft(pulse_signal, n=n_fft))
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0/fps)

    # Human heart rate boundaries [45, 180] BPM -> [0.75, 3.0] Hz
    low_freq = 0.75
    high_freq = 3.0

    valid_mask = (fft_freqs >= low_freq) & (fft_freqs <= high_freq)
    if not np.any(valid_mask):
        return 0.0, 0.0

    valid_freqs = fft_freqs[valid_mask]
    valid_mags = fft_vals[valid_mask]

    best_idx = np.argmax(valid_mags)
    peak_freq = valid_freqs[best_idx]
    heart_rate_bpm = peak_freq * 60.0

    # Calculate SNR: Power around peak (peak ± 0.1 Hz) / Power outside peak
    bin_width_hz = 0.1
    signal_mask = np.abs(fft_freqs - peak_freq) <= bin_width_hz
    
    # Noise is within physiological band but outside signal window
    noise_mask = valid_mask & (~signal_mask)

    signal_power = np.sum(fft_vals[signal_mask] ** 2)
    noise_power = np.sum(fft_vals[noise_mask] ** 2)

    snr = float(signal_power / (noise_power + 1e-9))
    return float(heart_rate_bpm), float(snr)


def evaluate_pulse_consistency(
    pulse_signal: np.ndarray, 
    fps: float, 
    window_size_s: float = 6.0, 
    step_s: float = 1.0
) -> Tuple[float, float]:
    """
    Measure heart rate variability across sliding windows.
    
    Returns:
        Tuple[float, float]: (mean_heart_rate, std_heart_rate)
    """
    T = len(pulse_signal)
    win_len = int(window_size_s * fps)
    if T < win_len or fps <= 0:
        # Too short, estimate globally
        hr, _ = estimate_heart_rate(pulse_signal, fps)
        return hr, 0.0

    step_len = max(1, int(step_s * fps))
    hrs = []

    for start in range(0, T - win_len + 1, step_len):
        win_sig = pulse_signal[start : start + win_len]
        hr, _ = estimate_heart_rate(win_sig, fps)
        if hr > 0.0:
            hrs.append(hr)

    if not hrs:
        hr, _ = estimate_heart_rate(pulse_signal, fps)
        return hr, 0.0

    return float(np.mean(hrs)), float(np.std(hrs))


def evaluate_blood_flow_realism(
    forehead_pulse: np.ndarray, 
    left_cheek_pulse: np.ndarray, 
    right_cheek_pulse: np.ndarray
) -> float:
    """
    Check blood flow cross-correlation across forehead and cheeks.
    
    Returns:
        float: Average Pearson correlation coefficient.
    """
    T = len(forehead_pulse)
    if T < 5 or len(left_cheek_pulse) != T or len(right_cheek_pulse) != T:
        return 0.0

    def corr_coeff(s1: np.ndarray, s2: np.ndarray) -> float:
        std1 = np.std(s1)
        std2 = np.std(s2)
        if std1 < 1e-6 or std2 < 1e-6:
            return 0.0
        cov = np.mean((s1 - np.mean(s1)) * (s2 - np.mean(s2)))
        return float(cov / (std1 * std2))

    r_fl = corr_coeff(forehead_pulse, left_cheek_pulse)
    r_fr = corr_coeff(forehead_pulse, right_cheek_pulse)
    r_lr = corr_coeff(left_cheek_pulse, right_cheek_pulse)

    mean_r = (r_fl + r_fr + r_lr) / 3.0
    return float(mean_r)
