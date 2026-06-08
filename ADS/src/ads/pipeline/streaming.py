"""Streaming audio processing for real-time deepfake detection.

Processes audio in streaming fashion for low-latency detection.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

import torch
import torchaudio

from ads.application.detector import DeepfakeDetector
from ads.config.settings import settings
from ads.domain.entities import AnalysisResult

logger = logging.getLogger(__name__)


class StreamingDetector:
    """Processes streaming audio chunks in real-time.

    Supports low-latency detection for live audio streams
    with rolling window analysis.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        self.window_duration = self.config.get("window_duration", 5.0)
        self.hop_duration = self.config.get("hop_duration", 2.5)
        self.sample_rate = self.config.get("sample_rate", settings.audio.sample_rate)
        self.window_samples = int(self.window_duration * self.sample_rate)
        self.hop_samples = int(self.hop_duration * self.sample_rate)

        self.detector = DeepfakeDetector(config)
        self.buffer = torch.zeros(0)
        self.results: List[Dict[str, Any]] = []

    async def process_chunk(
        self, chunk: torch.Tensor
    ) -> Optional[Dict[str, Any]]:
        """Process a single audio chunk and return detection result if ready.

        Args:
            chunk: Audio chunk tensor (samples,) or (1, samples)

        Returns:
            Detection result dict if window is complete, None otherwise
        """
        if chunk.dim() == 2:
            chunk = chunk.squeeze(0)

        self.buffer = torch.cat([self.buffer, chunk])

        if self.buffer.shape[0] >= self.window_samples:
            window = self.buffer[: self.window_samples]
            self.buffer = self.buffer[self.hop_samples :]

            return await self._analyze_window(window)

        return None

    async def _analyze_window(
        self, window: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze a single audio window."""
        import tempfile
        import soundfile as sf

        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                path = f.name
                sf.write(path, window.numpy(), self.sample_rate)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.detector.analyze, path, False
            )

            return {
                "is_deepfake": result.is_deepfake,
                "fake_probability": result.fake_probability,
                "confidence": result.confidence_score,
            }
        finally:
            if path:
                import os
                os.unlink(path)

    async def stream_analyze(
        self, audio_iterator: AsyncGenerator[torch.Tensor, None]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Analyze a streaming audio source.

        Args:
            audio_iterator: Async generator yielding audio chunks

        Yields:
            Detection results as windows complete
        """
        async for chunk in audio_iterator:
            result = await self.process_chunk(chunk)
            if result:
                yield result

    def reset(self) -> None:
        """Reset the streaming buffer."""
        self.buffer = torch.zeros(0)
        self.results.clear()
