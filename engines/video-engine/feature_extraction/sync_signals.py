"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.sync_signals
Layer   : Layer 2 — Temporal Feature Extraction
Branch  : Branch D — Cross-Modal Synchronization

Pure-NumPy audio-visual synchronization and alignment primitives.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


def compute_audio_energy(
    audio_signal: np.ndarray, 
    sample_rate: float, 
    target_len: int, 
    fps: float
) -> np.ndarray:
    """
    Compute short-time energy of the audio signal aligned with video frames.

    Args:
        audio_signal: 1D float/int numpy array representing mono audio waveform.
        sample_rate: Audio sampling rate (Hz).
        target_len: Target length of the output energy timeline (equal to video frames).
        fps: Target frame rate of the video.

    Returns:
        np.ndarray: Audio energy envelope of shape (target_len,).
    """
    if audio_signal is None or len(audio_signal) == 0 or target_len <= 0 or fps <= 0:
        return np.zeros(target_len, dtype=np.float32)

    # Frame interval in audio samples
    samples_per_frame = sample_rate / fps
    energy = []

    for t in range(target_len):
        # Center of the frame window in audio samples
        center_sample = int(t * samples_per_frame)
        
        # Window size is set to the frame interval duration (1/fps seconds)
        half_win = int(samples_per_frame / 2)
        start_idx = max(0, center_sample - half_win)
        end_idx = min(len(audio_signal), center_sample + half_win)

        window = audio_signal[start_idx:end_idx]
        if len(window) == 0:
            rms = 0.0
        else:
            # Root Mean Square energy
            rms = np.sqrt(np.mean(window.astype(np.float32) ** 2))
        energy.append(rms)

    energy_arr = np.array(energy, dtype=np.float32)
    # Scale to range [0, 1]
    max_val = np.max(energy_arr)
    if max_val > 1e-4:
        energy_arr /= max_val
    else:
        energy_arr = np.zeros_like(energy_arr)

    return energy_arr


def compute_av_delay(
    audio_energy: np.ndarray, 
    jaw_open: np.ndarray, 
    fps: float
) -> Tuple[float, float]:
    """
    Calculate the delay (offset) between the audio energy timeline 
    and the mouth jaw open ratio timeline using cross-correlation.

    Args:
        audio_energy: (T,) shape array of speech energy.
        jaw_open: (T,) shape array of jaw opening ratio.
        fps: Video frames per second.

    Returns:
        Tuple[float, float]: (delay_seconds, alignment_confidence)
    """
    T = len(audio_energy)
    if T < 5 or len(jaw_open) != T or fps <= 0:
        return 0.0, 0.0

    # 1. Normalize signals to zero-mean and unit variance
    a_norm = audio_energy - np.mean(audio_energy)
    j_norm = jaw_open - np.mean(jaw_open)
    
    std_a = np.std(audio_energy)
    std_j = np.std(jaw_open)
    
    if std_a > 1e-6:
        a_norm /= std_a
    if std_j > 1e-6:
        j_norm /= std_j

    # 2. Compute cross-correlation
    corr = np.correlate(a_norm, j_norm, mode="full")
    center = T - 1  # Index corresponding to 0 shift

    # Search window: limit search to +/- 1.0 second delay
    max_shift = int(fps * 1.0)
    start_search = max(0, center - max_shift)
    end_search = min(len(corr), center + max_shift + 1)

    search_corr = corr[start_search:end_search]
    if len(search_corr) == 0:
        return 0.0, 0.0

    best_idx_in_search = np.argmax(search_corr)
    best_idx = start_search + best_idx_in_search
    shift_frames = best_idx - center
    delay_s = shift_frames / fps

    # Normalize correlation peak to form a confidence score in [0, 1]
    peak_val = search_corr[best_idx_in_search] / T
    confidence = float(np.clip(peak_val, 0.0, 1.0))

    return float(delay_s), confidence


def compute_phoneme_viseme_mismatch(
    audio_energy: np.ndarray, 
    jaw_open: np.ndarray
) -> float:
    """
    Check if mouth shape opening correlates with voice energy envelope.
    Mismatches indicate synthetic audio-to-video alignment failures.

    Args:
        audio_energy: (T,) speech envelope.
        jaw_open: (T,) mouth jaw opening ratio.

    Returns:
        float: Mismatch index in range [0, 1].
    """
    T = len(audio_energy)
    if T < 3 or len(jaw_open) != T:
        return 0.0

    # Pearson correlation coefficient between speech loudness and jaw opening
    r = np.corrcoef(audio_energy, jaw_open)[0, 1]
    if np.isnan(r):
        r = 0.0

    # Normal speech: moderate to high correlation (r > 0.4).
    # Deepfakes: low or negative correlation (r <= 0.0).
    # Map r from [-1, 1] to mismatch [0, 1]
    mismatch = 1.0 - (r + 1.0) / 2.0
    return float(mismatch)


def compute_emotion_mismatch(
    audio_signal: np.ndarray, 
    landmarks: np.ndarray,
    fps: float
) -> float:
    """
    Compare speech prosody variations (acoustic energy fluctuations) with 
    eyebrow expressions. Mismatches represent artificial alignment.

    Args:
        audio_signal: 1D WAV audio waveform.
        landmarks: (T, 478, 3) face landmarks coordinate timeline.
        fps: Target frame rate.

    Returns:
        float: Emotion mismatch index in [0, 1].
    """
    if audio_signal is None or len(audio_signal) == 0 or landmarks is None or len(landmarks) < 5 or fps <= 0:
        return 0.0

    # 1. Audio prosody: variance of short-time energy envelopes
    # High variance = lively/emotional speech, low variance = flat/monotone speech
    samples_per_frame = int(16000 / fps)
    step = max(1, len(audio_signal) // len(landmarks))
    energies = []
    for i in range(len(landmarks)):
        chunk = audio_signal[i*step : (i+1)*step]
        if len(chunk) > 0:
            rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2) + 1e-9)
            energies.append(rms)
        else:
            energies.append(0.0)
            
    audio_prosody_var = float(np.var(energies))

    # 2. Facial animation: eyebrow landmark temporal movements
    # Eyebrows correspond to MediaPipe landmarks 70 (left) and 300 (right)
    # Take difference over time
    eyebrows = landmarks[:, [70, 300], 1]  # Y dimension of eyebrows
    eyebrow_vel = np.diff(eyebrows, axis=0)
    eyebrow_movement_var = float(np.var(np.linalg.norm(eyebrow_vel, axis=1)))

    # Normalize prosody & face movements to comparable variance scales
    audio_scaled = min(1.0, audio_prosody_var * 1000.0)
    face_scaled = min(1.0, eyebrow_movement_var * 500.0)

    # Simple absolute difference mismatch
    mismatch = abs(audio_scaled - face_scaled)
    return float(mismatch)
