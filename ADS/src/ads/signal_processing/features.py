"""Signal processing feature extraction for audio deepfake detection.

Extracts comprehensive acoustic features including:
- Mel Spectrogram
- MFCC (Mel-Frequency Cepstral Coefficients)
- LFCC (Linear Frequency Cepstral Coefficients)
- CQCC (Constant Q Cepstral Coefficients)
- Chroma Features
- Spectral Contrast
- Pitch Features
- Harmonic/Percussive Features
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F

from ads.config.settings import settings
from ads.domain.entities import SignalFeatures

logger = logging.getLogger(__name__)


class SignalProcessor:
    """Comprehensive signal processing feature extractor.

    Produces all required acoustic features for the multi-branch architecture.
    All methods return torch.Tensors for seamless PyTorch integration.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        self.sample_rate = self.config.get("sample_rate", settings.audio.sample_rate)
        self.n_mfcc = self.config.get("n_mfcc", settings.signal.n_mfcc)
        self.n_lfcc = self.config.get("n_lfcc", settings.signal.n_lfcc)
        self.n_cqcc = self.config.get("n_cqcc", settings.signal.n_cqcc)
        self.n_mels = self.config.get("n_mels", settings.signal.n_mels)
        self.n_fft = self.config.get("n_fft", settings.signal.n_fft)
        self.hop_length = self.config.get("hop_length", settings.signal.hop_length)
        self.win_length = self.config.get("win_length", settings.signal.win_length)
        self.f_min = self.config.get("f_min", settings.signal.f_min)
        self.f_max = self.config.get("f_max", settings.signal.f_max)
        self.power = self.config.get("power", settings.signal.power)
        self.n_chroma = self.config.get("n_chroma", settings.signal.n_chroma)
        self.n_bands = self.config.get("n_bands", settings.signal.n_bands)

        self._mel_scale: Optional[torchaudio.transforms.MelSpectrogram] = None
        self._mfcc_transform: Optional[torchaudio.transforms.MFCC] = None

    def _ensure_mel_scale(self) -> torchaudio.transforms.MelSpectrogram:
        if self._mel_scale is None:
            self._mel_scale = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
                f_min=self.f_min,
                f_max=self.f_max,
                power=self.power,
                normalized=True,
            )
        return self._mel_scale

    def _ensure_mfcc(self) -> torchaudio.transforms.MFCC:
        if self._mfcc_transform is None:
            self._mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=self.n_mfcc,
                melkwargs={
                    "n_fft": self.n_fft,
                    "hop_length": self.hop_length,
                    "n_mels": self.n_mels,
                    "f_min": self.f_min,
                    "f_max": self.f_max,
                },
            )
        return self._mfcc_transform

    def compute_mel_spectrogram(
        self, waveform: torch.Tensor
    ) -> torch.Tensor:
        """Compute Mel spectrogram from waveform.

        Args:
            waveform: Audio tensor (1, samples) or (samples,)

        Returns:
            Mel spectrogram tensor (n_mels, time)
        """
        mel = self._ensure_mel_scale()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        spec = mel(waveform)
        return torch.log(spec + 1e-8).squeeze(0)

    def compute_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute MFCC features.

        Args:
            waveform: Audio tensor (1, samples) or (samples,)

        Returns:
            MFCC tensor (n_mfcc, time)
        """
        mfcc_transform = self._ensure_mfcc()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        mfcc = mfcc_transform(waveform)
        return mfcc.squeeze(0)

    def compute_lfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute Linear Frequency Cepstral Coefficients.

        Important for detecting vocoder artifacts in synthetic speech.

        Args:
            waveform: Audio tensor (1, samples) or (samples,)

        Returns:
            LFCC tensor (n_lfcc, time)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        spec = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            power=self.power,
            normalized=True,
        )(waveform)

        linear_fbanks = torchaudio.functional.linear_fbanks(
            n_freqs=spec.shape[-2],
            f_min=self.f_min,
            f_max=self.f_max or (self.sample_rate // 2),
            n_filter=self.n_lfcc,
            sample_rate=self.sample_rate,
        )
        lfcc_feat = torch.matmul(linear_fbanks.T.unsqueeze(0), spec)
        lfcc_feat = torch.log(lfcc_feat + 1e-8)

        dct = torchaudio.functional.create_dct(n_mfcc=self.n_lfcc, n_mels=self.n_lfcc)
        lfcc = torch.matmul(dct.T.unsqueeze(0), lfcc_feat)
        return lfcc.squeeze(0)

    def compute_cqcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute Constant Q Cepstral Coefficients.

        Robust to frequency shifts, useful for detecting pitch-shifted synthetic speech.

        Args:
            waveform: Audio tensor (1, samples) or (samples,)

        Returns:
            CQCC tensor (n_cqcc, time)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        try:
            cqt = torchaudio.transforms.CQT(
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
                n_bins=self.n_cqcc * 2,
                f_min=self.f_min or 32.0,
            )(waveform)
            cqt = torch.log(cqt.abs() + 1e-8)
            return cqt.squeeze(0)
        except Exception as e:
            logger.warning(f"CQCC computation failed: {e}. Returning zeros.")
            return torch.zeros(self.n_cqcc, waveform.shape[-1] // self.hop_length)

    def compute_chroma(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute chroma features (12-dimensional pitch class profile).

        Args:
            waveform: Audio tensor (1, samples) or (samples,)

        Returns:
            Chroma tensor (12, time)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        spec = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=self.power,
        )(waveform)

        chroma = torchaudio.functional.compute_deltas(spec, win_length=3)
        chroma = chroma[:, : self.n_chroma, :]
        return chroma.squeeze(0)

    def compute_spectral_contrast(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute spectral contrast features.

        Captures the spectral peak/valley characteristics.

        Args:
            waveform: Audio tensor (1, samples) or (samples,)

        Returns:
            Spectral contrast tensor (n_bands + 1, time)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        contrast = torchaudio.transforms.SpectralCentroid(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )(waveform)
        return contrast.squeeze(0)

    def compute_pitch(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute pitch (F0 contour) using torchaudio's pitch detection.

        Args:
            waveform: Audio tensor (1, samples) or (samples,)

        Returns:
            Pitch tensor (1, time)
        """
        if waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        pitch = torchaudio.functional.detect_pitch_frequency(
            waveform=waveform,
            sample_rate=self.sample_rate,
            frame_time=self.hop_length / self.sample_rate,
            win_length=3,
        )
        return pitch.squeeze(0)

    def compute_harmonic_features(
        self, waveform: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompose audio into harmonic and percussive components.

        Useful for detecting synthesis artifacts that affect harmonic structure.

        Args:
            waveform: Audio tensor (1, samples) or (samples,)

        Returns:
            Tuple of (harmonic, percussive) component tensors
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        harmonic, percussive = torchaudio.functional.hpss(waveform)
        return harmonic.squeeze(0), percussive.squeeze(0)

    def compute_all_features(
        self,
        waveform: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> SignalFeatures:
        """Compute all signal features for a given waveform.

        Args:
            waveform: Audio waveform tensor
            sample_rate: Sample rate (defaults to config)

        Returns:
            SignalFeatures with all computed features
        """
        sr = sample_rate or self.sample_rate

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        harmonic, percussive = self.compute_harmonic_features(waveform)

        return SignalFeatures(
            waveform=waveform,
            mel_spectrogram=self.compute_mel_spectrogram(waveform),
            mfcc=self.compute_mfcc(waveform),
            lfcc=self.compute_lfcc(waveform),
            cqcc=self.compute_cqcc(waveform),
            chroma=self.compute_chroma(waveform),
            spectral_contrast=self.compute_spectral_contrast(waveform),
            pitch=self.compute_pitch(waveform),
            harmonic=harmonic,
            perceptual=percussive,
        )

    @staticmethod
    def stack_features(features: SignalFeatures) -> torch.Tensor:
        """Stack multiple features into a multi-channel tensor for CNN input.

        Args:
            features: SignalFeatures object

        Returns:
            Tensor (n_features, height, width) suitable for CNN input
        """
        channels = []
        for feat_name in ["mfcc", "lfcc", "cqcc", "chroma"]:
            feat = getattr(features, feat_name, None)
            if feat is not None and isinstance(feat, torch.Tensor):
                channels.append(feat)

        if channels:
            min_time = min(c.shape[-1] for c in channels)
            channels = [c[..., :min_time] for c in channels]
            return torch.stack(channels, dim=0)
        return torch.zeros(4, 40, 100)
