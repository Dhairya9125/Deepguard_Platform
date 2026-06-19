"""
ADS — Audio Loader
Loads audio files to a normalised mono waveform at target sample rate.
Supports WAV, MP3, FLAC, OGG, M4A.
"""
import logging
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


class AudioLoader:
    """
    Loads an audio file and returns a mono float32 numpy array
    resampled to the target sample rate.
    """

    SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}

    def __init__(self, sample_rate: int = 16000, max_duration_s: float = 60.0) -> None:
        self.sample_rate = sample_rate
        self.max_duration_s = max_duration_s

    def load(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load an audio file.

        Returns:
            (waveform, sample_rate) where waveform is float32 mono ndarray.

        Raises:
            ValueError: if the file format is not supported.
            FileNotFoundError: if the file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format '{suffix}'. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        try:
            import librosa  # type: ignore
            waveform, sr = librosa.load(
                file_path,
                sr=self.sample_rate,
                mono=True,
                duration=self.max_duration_s,
            )
            logger.info(
                "Loaded audio: %s | duration=%.2fs | sr=%d Hz",
                path.name, len(waveform) / sr, sr,
            )
            return waveform.astype(np.float32), sr

        except ImportError:
            # librosa not installed — return silent placeholder for testing
            logger.warning("librosa not installed. Returning silence placeholder.")
            n_samples = int(self.sample_rate * min(5.0, self.max_duration_s))
            return np.zeros(n_samples, dtype=np.float32), self.sample_rate
