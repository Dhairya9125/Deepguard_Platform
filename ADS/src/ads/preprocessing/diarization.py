"""Speaker Diarization using pyannote.audio.

Detects and segments different speakers in audio.
Optimized for CPU inference with fallback to oracle diarization.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from ads.config.settings import settings
from ads.domain.entities import SpeakerSegment

logger = logging.getLogger(__name__)


class SpeakerDiarization:
    """Speaker diarization engine.

    Uses pyannote.audio for multi-speaker detection and segmentation.
    Falls back to single-speaker assumption if model unavailable.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        self.model_id = self.config.get("model_id", settings.diarization.model_id)
        self.num_speakers = self.config.get("num_speakers", settings.diarization.num_speakers)
        self.threshold = self.config.get("threshold", settings.diarization.threshold)
        self.sample_rate = self.config.get("sample_rate", settings.audio.sample_rate)

        self._pipeline: Optional[Any] = None
        self._available = False

    def _load_pipeline(self) -> None:
        """Load pyannote pipeline lazily."""
        if self._pipeline is not None:
            return
        try:
            from pyannote.audio import Pipeline

            self._pipeline = Pipeline.from_pretrained(
                self.model_id,
                use_auth_token=None,
            )
            self._pipeline.to(torch.device("cpu"))
            self._available = True
            logger.info("Speaker diarization pipeline loaded")
        except Exception as e:
            logger.warning(f"pyannote pipeline unavailable: {e}. Using fallback.")
            self._available = False

    def diarize(
        self,
        waveform: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> List[SpeakerSegment]:
        """Perform speaker diarization on audio.

        Args:
            waveform: Audio waveform (channels, samples) or (samples,)
            sample_rate: Sample rate

        Returns:
            List of SpeakerSegment with speaker labels and time boundaries
        """
        self._load_pipeline()
        sr = sample_rate or self.sample_rate

        if not self._available:
            return self._fallback_diarization(waveform, sr)

        try:
            if waveform.dim() == 2:
                waveform = waveform.mean(dim=0, keepdim=True)

            from pyannote.core import Segment

            waveform_np = waveform.squeeze().numpy()
            duration = waveform_np.shape[-1] / sr

            import numpy as np
            from pyannote.audio import Audio

            audio = Audio(sample_rate=sr, mono=True)

            file = {"waveform": waveform, "sample_rate": sr}
            diarization = self._pipeline(file)

            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(
                    SpeakerSegment(
                        speaker_id=str(speaker),
                        start_time=turn.start,
                        end_time=turn.end,
                        confidence=1.0,
                    )
                )

            if not segments:
                return self._fallback_diarization(waveform, sr)

            return segments

        except Exception as e:
            logger.warning(f"Diarization failed: {e}. Using fallback.")
            return self._fallback_diarization(waveform, sr)

    def _fallback_diarization(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> List[SpeakerSegment]:
        """Fallback: treat entire audio as single speaker."""
        duration = waveform.shape[-1] / sample_rate
        return [
            SpeakerSegment(
                speaker_id="speaker_0",
                start_time=0.0,
                end_time=duration,
                confidence=1.0,
            )
        ]

    def count_speakers(self, segments: List[SpeakerSegment]) -> int:
        """Count unique speakers from segments."""
        return len({s.speaker_id for s in segments})
