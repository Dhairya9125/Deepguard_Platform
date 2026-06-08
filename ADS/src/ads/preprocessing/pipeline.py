"""Audio preprocessing pipeline combining VAD, diarization, and segmentation.

Orchestrates the complete preprocessing workflow for incoming audio.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio

from ads.config.settings import settings
from ads.domain.entities import AudioMetadata, SpeechSegment, SpeakerSegment
from ads.preprocessing.diarization import SpeakerDiarization
from ads.preprocessing.segmentation import AudioSegmenter
from ads.preprocessing.vad import SileroVAD

logger = logging.getLogger(__name__)


class AudioPreprocessingPipeline:
    """Complete audio preprocessing pipeline.

    Steps:
    1. Load and validate audio
    2. Resample to target sample rate
    3. Convert to mono
    4. Voice Activity Detection
    5. Speaker Diarization
    6. Audio Segmentation
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        self.sample_rate = self.config.get("sample_rate", settings.audio.sample_rate)
        self.mono = self.config.get("mono", settings.audio.mono)
        self.max_duration = self.config.get("max_duration", settings.audio.max_duration_seconds)
        self.normalize = self.config.get("normalize", settings.audio.normalize_audio)

        self.vad = SileroVAD(config)
        self.diarization = SpeakerDiarization(config)
        self.segmenter = AudioSegmenter(config)

    def process(
        self,
        audio_path: str,
    ) -> Dict[str, Any]:
        """Process audio file through the entire preprocessing pipeline.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with:
                - waveform: Processed waveform tensor
                - metadata: AudioMetadata
                - speech_segments: List of SpeechSegment
                - speaker_segments: List of SpeakerSegment
                - chunks: List of (chunk_tensor, start_time, end_time) tuples
        """
        start_time = time.time()

        waveform, orig_sr = torchaudio.load(audio_path)
        metadata = self._extract_metadata(audio_path, waveform, orig_sr)

        waveform = self._resample(waveform, orig_sr)
        if self.mono:
            waveform = self._to_mono(waveform)
        if self.normalize:
            waveform = self._normalize(waveform)

        if waveform.shape[-1] / self.sample_rate > self.max_duration:
            max_samples = int(self.max_duration * self.sample_rate)
            waveform = waveform[:, :max_samples]

        speech_segments = self.vad.get_speech_timestamps(waveform, self.sample_rate)
        speaker_segments = self.diarization.diarize(waveform, self.sample_rate)

        if speech_segments:
            chunks = self.segmenter.vad_guided_segments(waveform, speech_segments, self.sample_rate)
        else:
            chunks = self.segmenter.fixed_window_segments(waveform, self.sample_rate)

        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Preprocessing completed in {elapsed:.1f}ms: {len(chunks)} chunks from {len(speech_segments)} speech segments")

        return {
            "waveform": waveform,
            "metadata": metadata,
            "speech_segments": speech_segments,
            "speaker_segments": speaker_segments,
            "chunks": chunks,
            "preprocessing_time_ms": elapsed,
        }

    def _extract_metadata(
        self, path: str, waveform: torch.Tensor, sample_rate: int
    ) -> AudioMetadata:
        import hashlib, os

        sha256 = hashlib.sha256()
        wave_bytes = waveform.numpy().tobytes()
        sha256.update(wave_bytes)

        return AudioMetadata(
            filename=os.path.basename(path),
            format=AudioFormat[path.split(".")[-1].upper()] if path.split(".")[-1].upper()
                   in AudioFormat.__members__ else AudioFormat.WAV,
            duration_seconds=waveform.shape[-1] / sample_rate,
            sample_rate=sample_rate,
            channels=waveform.shape[0],
            file_size_bytes=os.path.getsize(path),
            hash_sha256=sha256.hexdigest(),
        )

    @staticmethod
    def _resample(waveform: torch.Tensor, orig_sr: int, target_sr: Optional[int] = None) -> torch.Tensor:
        target = target_sr or settings.audio.sample_rate
        if orig_sr != target:
            resampler = torchaudio.transforms.Resample(orig_sr, target)
            return resampler(waveform)
        return waveform

    @staticmethod
    def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.shape[0] > 1:
            return torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    @staticmethod
    def _normalize(waveform: torch.Tensor) -> torch.Tensor:
        max_val = waveform.abs().max()
        if max_val > 0:
            return waveform / max_val
        return waveform


from ads.domain.entities import AudioFormat
