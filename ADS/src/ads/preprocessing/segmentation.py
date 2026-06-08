"""Audio segmentation into chunks for processing.

Supports both fixed-window and VAD-based segmentation.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import torch
import torchaudio

from ads.config.settings import settings
from ads.domain.entities import SpeechSegment

logger = logging.getLogger(__name__)


class AudioSegmenter:
    """Segments audio into overlapping chunks for processing.

    Supports fixed window segmentation and VAD-guided segmentation.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        self.sample_rate = self.config.get("sample_rate", settings.audio.sample_rate)
        self.chunk_duration = self.config.get("chunk_duration", settings.audio.chunk_duration_seconds)
        self.hop_duration = self.config.get("hop_duration", settings.audio.hop_duration_seconds)
        self.min_duration = self.config.get("min_duration", settings.audio.min_duration_seconds)

        self.chunk_samples = int(self.chunk_duration * self.sample_rate)
        self.hop_samples = int(self.hop_duration * self.sample_rate)

    def fixed_window_segments(
        self,
        waveform: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> List[Tuple[torch.Tensor, float, float]]:
        """Split waveform into fixed-size overlapping windows.

        Args:
            waveform: Audio waveform (channels, samples)
            sample_rate: Optional sample rate override

        Returns:
            List of (chunk_tensor, start_time, end_time) tuples
        """
        sr = sample_rate or self.sample_rate
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        num_samples = waveform.shape[-1]
        chunks = []

        start = 0
        while start < num_samples:
            end = min(start + self.chunk_samples, num_samples)
            chunk = waveform[:, start:end]

            if end - start < int(self.min_duration * sr):
                if chunks:
                    last_chunk, last_start, _ = chunks.pop()
                    combined = torch.cat([last_chunk, chunk], dim=-1)
                    combined_duration = combined.shape[-1] / sr
                    chunks.append((combined, last_start, last_start + combined_duration))
                break

            start_time = start / sr
            end_time = end / sr
            chunks.append((chunk, start_time, end_time))
            start += self.hop_samples

        return chunks

    def vad_guided_segments(
        self,
        waveform: torch.Tensor,
        speech_segments: List[SpeechSegment],
        sample_rate: Optional[int] = None,
    ) -> List[Tuple[torch.Tensor, float, float]]:
        """Create chunks based on VAD speech segments.

        Args:
            waveform: Audio waveform (channels, samples)
            speech_segments: List of speech segments from VAD
            sample_rate: Optional sample rate override

        Returns:
            List of (chunk_tensor, start_time, end_time) tuples
        """
        sr = sample_rate or self.sample_rate
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        chunks = []
        for seg in speech_segments:
            start_sample = int(seg.start_time * sr)
            end_sample = int(seg.end_time * sr)
            chunk = waveform[:, start_sample:end_sample]
            chunks.append((chunk, seg.start_time, seg.end_time))

        if not chunks:
            return self.fixed_window_segments(waveform, sr)

        return chunks

    def merge_overlapping_windows(
        self,
        predictions: List[Tuple[float, float, float]],
        iou_threshold: float = 0.5,
    ) -> List[Tuple[float, float, float]]:
        """Merge overlapping prediction windows using NMS-style merging.

        Args:
            predictions: List of (start_time, end_time, score) tuples
            iou_threshold: IoU threshold for merging

        Returns:
            Merged list of (start_time, end_time, score) tuples
        """
        if not predictions:
            return []

        sorted_preds = sorted(predictions, key=lambda x: x[2], reverse=True)
        merged = []

        for pred in sorted_preds:
            is_merged = False
            for i, existing in enumerate(merged):
                iou = self._compute_iou(pred[:2], existing[:2])
                if iou > iou_threshold:
                    new_start = min(pred[0], existing[0])
                    new_end = max(pred[1], existing[1])
                    new_score = max(pred[2], existing[2])
                    merged[i] = (new_start, new_end, new_score)
                    is_merged = True
                    break
            if not is_merged:
                merged.append(pred)

        return merged

    @staticmethod
    def _compute_iou(seg1: Tuple[float, float], seg2: Tuple[float, float]) -> float:
        intersection = max(0, min(seg1[1], seg2[1]) - max(seg1[0], seg2[0]))
        union = max(seg1[1], seg2[1]) - min(seg1[0], seg2[0])
        return intersection / union if union > 0 else 0.0
