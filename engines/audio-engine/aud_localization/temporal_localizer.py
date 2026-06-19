"""
ADS — Temporal Localizer
Identifies the time segments within the audio that are most likely manipulated.
"""
import logging
from typing import Any, Dict, List
import numpy as np

logger = logging.getLogger(__name__)


class TemporalLocalizer:
    """
    Sliding-window localizer: computes per-segment fake scores
    to identify the manipulated time ranges.
    """

    def __init__(self, sample_rate: int = 16000, window_s: float = 1.0, hop_s: float = 0.5) -> None:
        self.sample_rate = sample_rate
        self.window_samples = int(window_s * sample_rate)
        self.hop_samples = int(hop_s * sample_rate)

    def localize(
        self,
        waveform: np.ndarray,
        global_fake_prob: float,
        threshold: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """
        Returns a list of manipulated segments:
        [{"start_s": float, "end_s": float, "fake_score": float}]
        """
        if global_fake_prob < threshold:
            return []  # Not fake enough to bother localising

        segments = []
        total_samples = len(waveform)

        for start in range(0, total_samples - self.window_samples, self.hop_samples):
            end = start + self.window_samples
            segment = waveform[start:end]

            # Heuristic: high spectral flux = potential boundary
            rms = float(np.sqrt(np.mean(segment**2)))
            flux = float(np.mean(np.abs(np.diff(segment))))

            # Score relative to whole-clip score (normalized)
            segment_score = float(np.clip(global_fake_prob * (1.0 + flux / 0.01), 0.0, 1.0))

            if segment_score > threshold:
                segments.append({
                    "start_s": round(start / self.sample_rate, 3),
                    "end_s":   round(end / self.sample_rate, 3),
                    "fake_score": round(segment_score, 4),
                    "rms": round(rms, 6),
                })

        # Merge adjacent segments
        return self._merge_segments(segments)

    def _merge_segments(
        self, segments: List[Dict[str, Any]], gap_s: float = 0.5
    ) -> List[Dict[str, Any]]:
        if not segments:
            return []
        merged = [segments[0].copy()]
        for seg in segments[1:]:
            if seg["start_s"] - merged[-1]["end_s"] <= gap_s:
                merged[-1]["end_s"] = max(merged[-1]["end_s"], seg["end_s"])
                merged[-1]["fake_score"] = max(merged[-1]["fake_score"], seg["fake_score"])
            else:
                merged.append(seg.copy())
        return merged
