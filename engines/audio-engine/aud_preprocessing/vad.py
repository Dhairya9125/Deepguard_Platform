"""
ADS — Voice Activity Detection (VAD)
Separates speech segments from silence/noise using energy-based detection.
"""
import logging
from typing import List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """
    Simple energy-based VAD.
    Splits waveform into (start_sample, end_sample) speech regions.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 512,
        hop_length: int = 128,
        energy_threshold: float = 0.01,
        min_speech_frames: int = 8,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.energy_threshold = energy_threshold
        self.min_speech_frames = min_speech_frames

    def detect(self, waveform: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect speech regions in the waveform.

        Returns:
            List of (start_sample, end_sample) tuples for each speech segment.
        """
        # Compute RMS energy per frame
        frames = [
            waveform[i : i + self.frame_length]
            for i in range(0, len(waveform) - self.frame_length, self.hop_length)
        ]
        energies = np.array([np.sqrt(np.mean(f ** 2)) for f in frames])

        # Threshold
        is_speech = energies > self.energy_threshold

        # Merge consecutive speech frames into segments
        segments: List[Tuple[int, int]] = []
        in_speech = False
        start_frame = 0

        for i, speech in enumerate(is_speech):
            if speech and not in_speech:
                start_frame = i
                in_speech = True
            elif not speech and in_speech:
                if i - start_frame >= self.min_speech_frames:
                    start_sample = start_frame * self.hop_length
                    end_sample = min(i * self.hop_length + self.frame_length, len(waveform))
                    segments.append((start_sample, end_sample))
                in_speech = False

        if in_speech:
            start_sample = start_frame * self.hop_length
            segments.append((start_sample, len(waveform)))

        logger.debug("VAD found %d speech segments", len(segments))
        return segments

    def get_speech_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """Return only the speech portions concatenated together."""
        segments = self.detect(waveform)
        if not segments:
            return waveform  # no silence detected — return full audio
        return np.concatenate([waveform[s:e] for s, e in segments])
