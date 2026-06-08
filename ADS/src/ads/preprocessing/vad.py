"""Voice Activity Detection using Silero VAD.

Detects speech intervals in audio with configurable thresholds.
Optimized for CPU inference.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torchaudio

from ads.config.settings import settings
from ads.domain.entities import SpeechSegment

logger = logging.getLogger(__name__)


class SileroVAD:
    """Voice Activity Detection using Silero VAD model.

    Detects speech intervals in audio signals with configurable
    thresholds for speech/silence detection.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        self.model_id = self.config.get("model_id", settings.vad.model_id)
        self.threshold = self.config.get("threshold", settings.vad.threshold)
        self.min_speech_duration_ms = self.config.get(
            "min_speech_duration_ms", settings.vad.min_speech_duration_ms
        )
        self.min_silence_duration_ms = self.config.get(
            "min_silence_duration_ms", settings.vad.min_silence_duration_ms
        )
        self.speech_pad_ms = self.config.get("speech_pad_ms", settings.vad.speech_pad_ms)
        self.sample_rate = self.config.get("sample_rate", settings.audio.sample_rate)

        self._model: Optional[torch.nn.Module] = None
        self._utils = None

    def _load_model(self) -> None:
        """Load Silero VAD model. Lazy-loaded on first use."""
        if self._model is not None:
            return
        try:
            model, utils = torch.hub.load(
                repo_or_dir=self.model_id,
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._model = model
            self._utils = utils
            self._model.eval()
            logger.info("Silero VAD model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            raise

    def get_speech_timestamps(
        self,
        waveform: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> List[SpeechSegment]:
        """Detect speech segments in waveform.

        Args:
            waveform: Audio waveform tensor (channels, samples) or (samples,)
            sample_rate: Sample rate of the waveform

        Returns:
            List of SpeechSegment objects with start/end times
        """
        self._load_model()
        sr = sample_rate or self.sample_rate

        if waveform.dim() == 2:
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            waveform = waveform.squeeze(0)

        get_speech_ts = self._utils[0]
        speech_segments = get_speech_ts(
            waveform,
            self._model,
            sampling_rate=sr,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=True,
        )

        return [
            SpeechSegment(
                start_time=seg["start"],
                end_time=seg["end"],
                confidence=seg.get("confidence", self.threshold),
            )
            for seg in speech_segments
        ]

    def is_speech(self, waveform_chunk: torch.Tensor, sample_rate: Optional[int] = None) -> float:
        """Get speech probability for a single audio chunk.

        Args:
            waveform_chunk: Audio chunk tensor
            sample_rate: Sample rate

        Returns:
            Speech probability between 0 and 1
        """
        self._load_model()
        sr = sample_rate or self.sample_rate

        if waveform_chunk.dim() == 2:
            waveform_chunk = waveform_chunk.squeeze(0)

        with torch.no_grad():
            speech_prob = self._model(waveform_chunk, sr).item()
        return speech_prob

    def remove_silence(
        self,
        waveform: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[SpeechSegment]]:
        """Remove silence regions from audio.

        Args:
            waveform: Input audio waveform
            sample_rate: Sample rate

        Returns:
            Tuple of (cleaned_waveform, speech_segments)
        """
        segments = self.get_speech_timestamps(waveform, sample_rate)
        sr = sample_rate or self.sample_rate

        if not segments:
            return waveform, segments

        cleaned_parts = []
        for seg in segments:
            start_sample = int(seg.start_time * sr)
            end_sample = int(seg.end_time * sr)
            cleaned_parts.append(waveform[..., start_sample:end_sample])

        cleaned = torch.cat(cleaned_parts, dim=-1) if len(cleaned_parts) > 1 else cleaned_parts[0]
        return cleaned, segments
