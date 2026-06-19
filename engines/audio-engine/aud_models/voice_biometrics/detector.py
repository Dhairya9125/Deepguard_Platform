"""
ADS — Voice Biometrics Branch
Measures prosody and pitch consistency as a voice-authenticity signal.
Real voices have natural micro-variations; synthetic voices can be unnaturally smooth.
"""
import logging
from typing import Any, Dict
import numpy as np
from ..base_branch import BaseBranch
from ...domain.entities import BranchResult

logger = logging.getLogger(__name__)


class VoiceBiometricsBranch(BaseBranch):
    name = "voice_biometrics"

    def predict(self, waveform: np.ndarray, features: Dict[str, Any]) -> BranchResult:
        try:
            import librosa  # type: ignore
            # Extract fundamental frequency (F0) via piptrack
            pitches, magnitudes = librosa.piptrack(
                y=waveform,
                sr=16000,
                fmin=50.0,
                fmax=500.0,
            )
            # Pick dominant pitch per frame
            pitch_track = np.array([
                pitches[magnitudes[:, t].argmax(), t] if magnitudes[:, t].max() > 0 else 0.0
                for t in range(pitches.shape[1])
            ])
            voiced = pitch_track[pitch_track > 0]

            if len(voiced) < 10:
                return BranchResult(self.name, 0.5, 0.2)

            pitch_std  = float(np.std(voiced))
            pitch_mean = float(np.mean(voiced))

            # Very low std → unnaturally monotone (synthetic indicator)
            # Very high std → unlikely for normal speech
            norm_std = pitch_std / (pitch_mean + 1e-8)
            # Synthetic voices often have norm_std < 0.05
            fake_score = float(np.clip(1.0 - norm_std / 0.15, 0.0, 1.0))
            confidence = min(0.75, 0.4 + abs(fake_score - 0.5))

            return BranchResult(
                branch_name=self.name,
                fake_score=fake_score,
                confidence=confidence,
                features={
                    "pitch_mean_hz": round(pitch_mean, 2),
                    "pitch_std_hz":  round(pitch_std, 2),
                    "norm_std":      round(norm_std, 4),
                },
            )
        except Exception as exc:
            logger.error("VoiceBiometricsBranch error: %s", exc)
            return BranchResult(self.name, 0.5, 0.0, error=str(exc))
