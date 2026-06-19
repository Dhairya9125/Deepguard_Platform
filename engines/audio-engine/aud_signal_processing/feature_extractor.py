"""
ADS — Spectral Feature Extractor
Extracts MFCC, delta-MFCC, spectral contrast, chroma, and zero-crossing rate
from an audio waveform. These features feed into detection branches.
"""
import logging
from typing import Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class SpectralFeatureExtractor:
    """
    Extracts a rich feature set from a waveform for deepfake detection.
    Falls back gracefully if librosa is not installed (returns zeros).
    """

    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 40) -> None:
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

    def extract(self, waveform: np.ndarray) -> Dict[str, Any]:
        """
        Extract all features from a mono float32 waveform.

        Returns a dict with keys:
          mfcc, delta_mfcc, delta2_mfcc, spectral_centroid,
          spectral_contrast, chroma, zcr, rms
        """
        try:
            import librosa  # type: ignore

            mfcc = librosa.feature.mfcc(
                y=waveform, sr=self.sample_rate, n_mfcc=self.n_mfcc
            )
            delta_mfcc = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)

            spec_centroid = librosa.feature.spectral_centroid(
                y=waveform, sr=self.sample_rate
            )
            spec_contrast = librosa.feature.spectral_contrast(
                y=waveform, sr=self.sample_rate
            )
            chroma = librosa.feature.chroma_stft(
                y=waveform, sr=self.sample_rate
            )
            zcr = librosa.feature.zero_crossing_rate(waveform)
            rms = librosa.feature.rms(y=waveform)

            return {
                "mfcc": mfcc,
                "delta_mfcc": delta_mfcc,
                "delta2_mfcc": delta2_mfcc,
                "spectral_centroid": spec_centroid,
                "spectral_contrast": spec_contrast,
                "chroma": chroma,
                "zcr": zcr,
                "rms": rms,
                "mfcc_mean": mfcc.mean(axis=1),
                "mfcc_std": mfcc.std(axis=1),
            }

        except ImportError:
            logger.warning("librosa not installed. Returning zero features.")
            z = np.zeros(self.n_mfcc)
            return {
                "mfcc": np.zeros((self.n_mfcc, 100)),
                "delta_mfcc": np.zeros((self.n_mfcc, 100)),
                "delta2_mfcc": np.zeros((self.n_mfcc, 100)),
                "spectral_centroid": np.zeros((1, 100)),
                "spectral_contrast": np.zeros((7, 100)),
                "chroma": np.zeros((12, 100)),
                "zcr": np.zeros((1, 100)),
                "rms": np.zeros((1, 100)),
                "mfcc_mean": z,
                "mfcc_std": z,
            }
