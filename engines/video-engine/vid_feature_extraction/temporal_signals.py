"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.temporal_signals
Layer   : Layer 2, Branch B — Temporal Consistency Analysis
Task    : Pure-NumPy temporal signal processing primitives

All functions are stateless, side-effect-free, and operate on plain
Python lists or numpy arrays. They carry no model weights and have
zero deep learning dependencies — NumPy + SciPy only.

These primitives are the forensic core of Branch B. Each one targets
a specific temporal signature of generative-AI-produced deepfakes:

Signal Function                  Deepfake Signature Targeted
────────────────────────────────────────────────────────────────────
rolling_std()                 → Local flickering / per-frame instability
linear_trend()                → Gradual swap blending / prob ramp
peak_count()                  → Isolated "hard" fake frames (sparse swaps)
cosine_drift_timeline()       → Frame-to-frame latent space jumps
heatmap_flux_timeline()       → Spatial manipulation region instability
detect_blink_rate()           → Unnatural blink cadence
pearson_correlation()         → Jaw-heatmap cross-modal mis-sync
scene_fake_prob_map()         → Scene-boundary fake prob transitions
derive_verdict()              → Weighted rule-based temporal verdict
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rolling / windowed statistics
# ---------------------------------------------------------------------------

def rolling_std(
    values:      List[float],
    window:      int  = 5,
) -> float:
    """
    Compute the mean of per-window standard deviations across a timeline.

    A high rolling_std on a fake_probability timeline indicates local
    flickering — the network confidence jumps frame-to-frame because the
    synthetic content is not temporally consistent.

    Args:
        values : Ordered list of scalar floats (e.g. fake_prob timeline).
        window : Rolling window size. Default: 5 frames.

    Returns:
        Mean of per-window std values. Returns 0.0 if len(values) < window.
    """
    n = len(values)
    if n < window:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    stds = [
        float(arr[i: i + window].std())
        for i in range(n - window + 1)
    ]
    return float(np.mean(stds)) if stds else 0.0


def rolling_mean(
    values: List[float],
    window: int = 5,
) -> List[float]:
    """
    Compute the rolling mean of a timeline.

    Returns a list of length (n - window + 1) — one value per window position.
    Returns an empty list if len(values) < window.
    """
    n = len(values)
    if n < window:
        return []
    arr = np.asarray(values, dtype=np.float64)
    return [float(arr[i: i + window].mean()) for i in range(n - window + 1)]


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------

def linear_trend(values: List[float]) -> float:
    """
    Fit a 1-D linear regression and return the slope.

    A rising slope on the fake_probability timeline indicates a gradual
    swap blending — the face becomes progressively more synthetic over time.
    A near-zero slope indicates temporal stability (either real or consistently fake).
    A falling slope may indicate a partial swap that is being corrected.

    Args:
        values : Ordered list of scalar floats.

    Returns:
        Slope of the least-squares linear fit. Returns 0.0 if n < 2.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x   = np.arange(n, dtype=np.float64)
    y   = np.asarray(values, dtype=np.float64)
    # Closed-form OLS slope: cov(x,y) / var(x)
    x_mean = x.mean()
    y_mean = y.mean()
    numerator   = float(np.dot(x - x_mean, y - y_mean))
    denominator = float(np.dot(x - x_mean, x - x_mean))
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

def peak_count(
    values:           List[float],
    high_threshold:   float = 0.70,
    low_threshold:    float = 0.50,
    min_gap:          int   = 2,
) -> int:
    """
    Count isolated high-confidence fake frames surrounded by lower-confidence frames.

    This targets sparse swap artifacts — individual frames where the generator
    fails to maintain consistency, producing a transient spike in fake_probability.
    Real faces have uniformly low probabilities; consistently-generated deepfakes
    have uniformly high probabilities. Spikes indicate per-frame rendering failures.

    A "peak" is defined as:
        values[i] > high_threshold
        AND at least one neighbour within `min_gap` frames is < low_threshold.

    Args:
        values          : Ordered fake_prob timeline.
        high_threshold  : Peak amplitude threshold. Default: 0.70.
        low_threshold   : Surrounding valley threshold. Default: 0.50.
        min_gap         : Neighbourhood radius to check for valley. Default: 2.

    Returns:
        Number of detected isolated peaks (int).
    """
    if len(values) < 3:
        return 0
    arr   = np.asarray(values, dtype=np.float64)
    n     = len(arr)
    count = 0
    for i in range(n):
        if arr[i] > high_threshold:
            # Check if any neighbour within min_gap is below low_threshold
            lo = max(0, i - min_gap)
            hi = min(n, i + min_gap + 1)
            neighbourhood = np.concatenate([arr[lo:i], arr[i+1:hi]])
            if len(neighbourhood) > 0 and np.any(neighbourhood < low_threshold):
                count += 1
    return count


# ---------------------------------------------------------------------------
# Embedding drift
# ---------------------------------------------------------------------------

def cosine_drift_timeline(
    embeddings: List[np.ndarray],
) -> List[float]:
    """
    Compute frame-to-frame cosine distance between consecutive (2560,) embeddings.

    Cosine distance ∈ [0, 1]:
        0 = identical direction (perfectly stable embedding)
        1 = orthogonal (maximum latent-space shift)
        2 = opposite direction (not common in practice)

    Real faces maintain very low cosine drift (<0.05) across consecutive frames.
    Deepfakes, especially those with rendering artefacts at cut boundaries,
    exhibit sharp jumps (>0.15) because the generator samples a different
    region of the latent space between frames.

    Args:
        embeddings : Ordered list of (D,) float32/64 numpy arrays.

    Returns:
        List of n-1 cosine distance values.
        Returns [] if len(embeddings) < 2.
    """
    n = len(embeddings)
    if n < 2:
        return []

    dists: List[float] = []
    for i in range(n - 1):
        a = embeddings[i].astype(np.float64)
        b = embeddings[i + 1].astype(np.float64)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            # Degenerate embedding (all-zero from failed IDS) — skip
            dists.append(0.0)
            continue
        cos_sim  = float(np.dot(a, b) / (na * nb))
        cos_sim  = float(np.clip(cos_sim, -1.0, 1.0))
        cos_dist = 1.0 - cos_sim
        dists.append(cos_dist)
    return dists


# ---------------------------------------------------------------------------
# Heatmap temporal flux
# ---------------------------------------------------------------------------

def heatmap_flux_timeline(
    heatmaps: List[np.ndarray],
) -> List[float]:
    """
    Compute mean absolute pixel-wise change between consecutive heatmaps.

    Each heatmap is a (224, 224) float32 array in [0, 1].

    A real, unmanipulated face region produces a temporally stable heatmap
    (low activation and smooth transitions). A synthetic region boundary
    shifts spatially frame-to-frame, producing high pixel-level flux.

    Args:
        heatmaps : Ordered list of (H, W) float32 numpy arrays.

    Returns:
        List of n-1 mean absolute difference values. Returns [] if n < 2.
    """
    n = len(heatmaps)
    if n < 2:
        return []

    flux: List[float] = []
    for i in range(n - 1):
        diff = np.abs(
            heatmaps[i].astype(np.float64) - heatmaps[i + 1].astype(np.float64)
        )
        flux.append(float(diff.mean()))
    return flux


# ---------------------------------------------------------------------------
# Landmark temporal signals
# ---------------------------------------------------------------------------

def detect_blink_rate(
    ear_left:    List[float],
    ear_right:   List[float],
    fps:         float,
    ear_threshold: float = 0.20,
    min_closed:    int   = 1,
) -> float:
    """
    Estimate blink rate (blinks per second) from Eye Aspect Ratio timelines.

    A blink is defined as a contiguous segment where both
    (ear_left[i] + ear_right[i]) / 2 < ear_threshold.

    Normal human blink rate: 0.2–0.4 blinks/second (12–24 per minute).
    Values significantly outside this range are forensically suspicious:
    - Too low (<0.05/s): synthetic face with frozen expression
    - Too high (>0.8/s): extreme blinking / poorly trained generator

    Args:
        ear_left      : Left eye EAR timeline.
        ear_right     : Right eye EAR timeline.
        fps           : Extraction FPS (needed to convert frames to seconds).
        ear_threshold : EAR value below which eye is considered 'closed'. Default: 0.20.
        min_closed    : Minimum consecutive frames for a valid blink. Default: 1.

    Returns:
        Blink rate in blinks per second. Returns 0.0 if not enough data.
    """
    n = min(len(ear_left), len(ear_right))
    if n < 3 or fps <= 0:
        return 0.0

    mean_ear    = [(l + r) / 2.0 for l, r in zip(ear_left[:n], ear_right[:n])]
    arr         = np.asarray(mean_ear)
    is_closed   = arr < ear_threshold

    # Count transitions from open→closed
    blink_count = 0
    in_blink    = False
    closed_run  = 0

    for closed in is_closed:
        if closed:
            closed_run += 1
            if not in_blink:
                in_blink = True
        else:
            if in_blink and closed_run >= min_closed:
                blink_count += 1
            in_blink   = False
            closed_run = 0

    # Close final blink if timeline ends while eye is closed
    if in_blink and closed_run >= min_closed:
        blink_count += 1

    duration_s = n / fps if fps > 0 else 1.0
    return float(blink_count / duration_s)


# ---------------------------------------------------------------------------
# Cross-modal correlation
# ---------------------------------------------------------------------------

def pearson_correlation(x: List[float], y: List[float]) -> float:
    """
    Compute the Pearson correlation coefficient between two scalar timelines.

    Used to correlate:
        - jaw_open_ratio vs. heatmap_activity_per_frame
          (detects lip-sync fakes — synthetic mouth is unlocked from heatmap)

    Returns:
        Pearson r in [-1, 1]. Returns 0.0 if n < 2 or either signal is constant.
    """
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    xa  = np.asarray(x[:n], dtype=np.float64)
    ya  = np.asarray(y[:n], dtype=np.float64)
    sx  = xa.std()
    sy  = ya.std()
    if sx < 1e-10 or sy < 1e-10:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


# ---------------------------------------------------------------------------
# Scene-level fake probability map
# ---------------------------------------------------------------------------

def compute_scene_fake_prob_map(
    frame_ids:    List[int],
    fake_probs:   List[float],
    scene_ids:    List[int],
) -> Tuple[Dict[int, float], float]:
    """
    Compute per-scene mean fake_probability and scene consistency score.

    Scene consistency score = std of per-scene means.
    High consistency_score → fake probability varies substantially across scenes
    → face swap was introduced only in certain scenes.

    Args:
        frame_ids  : Frame IDs of the valid Branch A results (same length).
        fake_probs : Fake probability values (same length as frame_ids).
        scene_ids  : Scene ID for each frame (same length as frame_ids).

    Returns:
        (scene_fake_prob_map, scene_consistency_score)
    """
    if not fake_probs:
        return {}, 0.0

    # Group per scene
    scene_buckets: Dict[int, List[float]] = {}
    for sid, prob in zip(scene_ids, fake_probs):
        if sid not in scene_buckets:
            scene_buckets[sid] = []
        scene_buckets[sid].append(prob)

    scene_map = {
        sid: float(np.mean(probs))
        for sid, probs in scene_buckets.items()
    }

    if len(scene_map) < 2:
        return scene_map, 0.0

    consistency_score = float(np.std(list(scene_map.values())))
    return scene_map, consistency_score


# ---------------------------------------------------------------------------
# Temporal verdict derivation
# ---------------------------------------------------------------------------

# Thresholds tuned against FaceForensics++ and DFDC validation sets.
# These are starting points for fine-tuning — each threshold maps directly
# to a known deepfake behavioral signature (see docstrings above).
_THRESHOLDS = {
    "fake_prob_mean":          0.55,   # Primary IDS signal
    "fake_prob_variance":      0.04,   # Flickering threshold
    "fake_prob_rolling_std":   0.08,   # Local instability threshold
    "fake_prob_trend_slope":   0.003,  # Rising trend threshold
    "fake_prob_peak_count":    3,      # Isolated spike threshold
    "embedding_drift_mean":    0.08,   # Latent space jump threshold (cosine)
    "embedding_drift_max":     0.25,   # Single extreme drift threshold
    "heatmap_flux_mean":       0.05,   # Spatial boundary shift threshold
    "heatmap_activity_mean":   0.35,   # High baseline manipulation
    "scene_consistency_score": 0.12,   # Scene-swap indicator
    "landmark_jaw_variance":   0.02,   # Abnormal jaw motion threshold
    "jaw_heatmap_correlation": 0.60,   # Lip-sync over-correlation threshold
}

# Signal weights — higher weight = stronger forensic evidence
_WEIGHTS = {
    "fake_prob_mean":          3.0,
    "fake_prob_variance":      1.5,
    "fake_prob_rolling_std":   1.5,
    "fake_prob_trend_slope":   1.0,
    "fake_prob_peak_count":    1.0,
    "embedding_drift_mean":    2.5,
    "embedding_drift_max":     1.5,
    "heatmap_flux_mean":       1.5,
    "heatmap_activity_mean":   1.0,
    "scene_consistency_score": 2.0,
    "landmark_jaw_variance":   0.5,
    "jaw_heatmap_correlation": 1.0,
}

# Maximum possible weighted score (sum of all weights)
_MAX_SCORE = sum(_WEIGHTS.values())


def derive_verdict(
    fake_prob_mean:          float,
    fake_prob_variance:      float,
    fake_prob_rolling_std:   float,
    fake_prob_trend_slope:   float,
    fake_prob_peak_count:    int,
    embedding_drift_mean:    float,
    embedding_drift_max:     float,
    heatmap_flux_mean:       float,
    heatmap_activity_mean:   float,
    scene_consistency_score: float,
    landmark_jaw_variance:   float   = 0.0,
    jaw_heatmap_correlation: float   = 0.0,
    landmark_available:      bool    = False,
) -> Tuple[str, float, List[str]]:
    """
    Derive a temporal deepfake verdict from all Branch B signals.

    Uses a weighted rule-based scoring system:
        score += weight[signal] when signal exceeds threshold[signal]

    The final score is normalised to [0, 1] (temporal_confidence).

    Verdict mapping:
        confidence > 0.55  → 'FAKE'
        confidence < 0.25  → 'REAL'
        otherwise          → 'UNCERTAIN'

    Args:
        All named temporal signal values.
        landmark_available: If False, landmark-based signals are excluded
                            from the score to avoid penalising missing data.

    Returns:
        Tuple of (verdict: str, confidence: float, signals_fired: List[str]).
        - verdict: 'FAKE' | 'REAL' | 'UNCERTAIN'
        - confidence: Normalised weighted score in [0, 1].
        - signals_fired: Names of signals that exceeded their thresholds.
    """
    t   = _THRESHOLDS
    w   = _WEIGHTS
    signals_fired: List[str] = []
    score = 0.0

    def _check(name: str, value: float, threshold: float, invert: bool = False) -> None:
        exceeded = (value < threshold) if invert else (value >= threshold)
        if exceeded:
            signals_fired.append(name)
            score_delta = w.get(name, 1.0)
            nonlocal score
            score += score_delta

    _check("fake_prob_mean",          fake_prob_mean,          t["fake_prob_mean"])
    _check("fake_prob_variance",       fake_prob_variance,      t["fake_prob_variance"])
    _check("fake_prob_rolling_std",    fake_prob_rolling_std,   t["fake_prob_rolling_std"])
    _check("fake_prob_trend_slope",    abs(fake_prob_trend_slope), t["fake_prob_trend_slope"])
    _check("fake_prob_peak_count",     float(fake_prob_peak_count), float(t["fake_prob_peak_count"]))
    _check("embedding_drift_mean",     embedding_drift_mean,    t["embedding_drift_mean"])
    _check("embedding_drift_max",      embedding_drift_max,     t["embedding_drift_max"])
    _check("heatmap_flux_mean",        heatmap_flux_mean,       t["heatmap_flux_mean"])
    _check("heatmap_activity_mean",    heatmap_activity_mean,   t["heatmap_activity_mean"])
    _check("scene_consistency_score",  scene_consistency_score, t["scene_consistency_score"])

    # Include landmark signals only when data is available
    if landmark_available:
        _check("landmark_jaw_variance",   landmark_jaw_variance,   t["landmark_jaw_variance"])
        _check("jaw_heatmap_correlation", jaw_heatmap_correlation, t["jaw_heatmap_correlation"])

    # Normalise score to [0, 1]
    effective_max = _MAX_SCORE
    if not landmark_available:
        effective_max -= w.get("landmark_jaw_variance", 0)
        effective_max -= w.get("jaw_heatmap_correlation", 0)
    effective_max = max(effective_max, 1.0)

    confidence = float(np.clip(score / effective_max, 0.0, 1.0))

    if confidence >= 0.55:
        verdict = "FAKE"
    elif confidence < 0.25:
        verdict = "REAL"
    else:
        verdict = "UNCERTAIN"

    return verdict, confidence, signals_fired


def risk_level(confidence: float) -> str:
    """Map temporal confidence to a human-readable risk level."""
    if confidence >= 0.70:
        return "HIGH"
    elif confidence >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"
