"""
ADS Configuration
"""
from dataclasses import dataclass

@dataclass
class ADSConfig:
    # Audio loading
    sample_rate: int = 16000          # Target sample rate (Hz)
    max_duration_s: float = 60.0      # Max audio clip duration to analyse
    mono: bool = True                 # Convert to mono before analysis

    # Detection thresholds
    fake_threshold: float = 0.75      # Above this → FAKE
    uncertain_threshold: float = 0.45 # Above this → UNCERTAIN

    # Branch weights for ensemble fusion
    spectral_weight: float = 0.25
    temporal_weight: float = 0.20
    biometric_weight: float = 0.20
    wavlm_weight: float = 0.15
    xlsr_weight: float = 0.10
    diffusion_weight: float = 0.05
    adversarial_weight: float = 0.05

    # Device
    device: str = "cpu"

DEFAULT_CONFIG = ADSConfig()
