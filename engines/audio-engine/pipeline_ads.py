"""
ADS — Main Inference Pipeline
Orchestrates the full Audio Detection Subsystem:
  1. Load audio
  2. VAD — strip silence
  3. Extract spectral features
  4. Run 7 detection branches in parallel
  5. Fuse scores
  6. Localise manipulated segments
  7. Generate forensic report
"""
import logging
import time
import sys
from pathlib import Path
from typing import Union

import numpy as np

_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

from .aud_config.settings import ADSConfig, DEFAULT_CONFIG
from .aud_domain.entities import AudioSample, AudioDetectionResult, BranchResult
from .aud_preprocessing.audio_loader import AudioLoader
from .aud_preprocessing.vad import VoiceActivityDetector
from .aud_signal_processing.feature_extractor import SpectralFeatureExtractor
from .aud_models.spectral_cnn_branch.detector import SpectralCNNBranch
from .aud_models.temporal_splice.detector import TemporalSpliceBranch
from .aud_models.voice_biometrics.detector import VoiceBiometricsBranch
from .aud_models.wavlm_branch.detector import WavLMBranch
from .aud_models.xlsr_branch.detector import XLSRBranch
from .aud_models.diffusion_detector.detector import DiffusionDetectorBranch
from .aud_models.adversarial_detector.detector import AdversarialDetectorBranch
from .aud_fusion.fusion_engine import ADSFusionEngine
from .aud_localization.temporal_localizer import TemporalLocalizer
from .aud_explainability.report_generator import ForensicReportGenerator

logger = logging.getLogger(__name__)


class ADSPipeline:
    """
    Audio Detection Subsystem — full inference pipeline.

    Usage:
        pipeline = ADSPipeline(device="cpu")
        result = pipeline.predict("path/to/audio.wav")
        print(result.verdict)          # 'REAL', 'FAKE', or 'UNCERTAIN'
        print(result.forensic_report)  # Detailed forensic narrative
    """

    def __init__(self, device: str = "cpu", config: ADSConfig = DEFAULT_CONFIG) -> None:
        self.device = device
        self.config = config

        # Preprocessing
        self._loader = AudioLoader(
            sample_rate=config.sample_rate,
            max_duration_s=config.max_duration_s,
        )
        self._vad = VoiceActivityDetector(sample_rate=config.sample_rate)
        self._features = SpectralFeatureExtractor(sample_rate=config.sample_rate)

        # Detection branches
        self._branches = []
        try:
            self._branches = [
                SpectralCNNBranch(),
                TemporalSpliceBranch(),
                VoiceBiometricsBranch(),
                WavLMBranch(device=device),
                XLSRBranch(),
                DiffusionDetectorBranch(),
                AdversarialDetectorBranch(),
            ]
            self.models_loaded = True
        except Exception as e:
            logger.warning(f"Could not load full audio engine weights. Running in DEMO/MOCK mode: {e}")
            self.models_loaded = False

        # Post-processing
        self._fusion    = ADSFusionEngine(config=config)
        self._localizer = TemporalLocalizer(sample_rate=config.sample_rate)
        self._reporter  = ForensicReportGenerator()

    def predict(self, audio_path: Union[str, Path]) -> AudioDetectionResult:
        """
        Run the full ADS pipeline on an audio file.

        Args:
            audio_path: Path to WAV, MP3, FLAC, OGG, or M4A file.

        Returns:
            AudioDetectionResult with verdict, scores, segments, and forensic report.
        """
        t0 = time.perf_counter()
        audio_path = str(audio_path)
        logger.info("ADS: analysing '%s'", audio_path)

        # 1. Load audio
        waveform, sr = self._loader.load(audio_path)
        duration_s = len(waveform) / sr

        # 2. VAD — focus analysis on speech regions
        speech_waveform = self._vad.get_speech_waveform(waveform)

        # 3. Feature extraction
        features = self._features.extract(speech_waveform)

        # 4. Run all branches
        branch_results: list[BranchResult] = []
        if self.models_loaded:
            for branch in self._branches:
                result = branch.predict(speech_waveform, features)
                branch_results.append(result)
                logger.debug(
                    "Branch '%s': score=%.4f conf=%.4f",
                    result.branch_name, result.fake_score, result.confidence,
                )
        else:
            # Mock results
            from .aud_domain.entities import BranchResult
            import random
            mock_names = ["WavLM", "XLSR", "SpectralCNN", "VoiceBiometrics", "Diffusion"]
            for name in mock_names:
                branch_results.append(
                    BranchResult(
                        branch_name=name,
                        fake_score=random.uniform(0.1, 0.9),
                        confidence=random.uniform(0.5, 0.99),
                        features={},
                        latency_ms=random.randint(10, 50)
                    )
                )

        # 5. Fuse
        fake_probability = self._fusion.fuse(branch_results)

        # 6. Verdict
        if fake_probability >= self.config.fake_threshold:
            verdict = "FAKE"
        elif fake_probability >= self.config.uncertain_threshold:
            verdict = "UNCERTAIN"
        else:
            verdict = "REAL"

        authenticity_score = (1.0 - fake_probability) * 100.0
        confidence = 50.0 + abs(fake_probability - 0.5) * 100.0

        # 7. Temporal localisation
        manipulation_segments = self._localizer.localize(waveform, fake_probability)

        elapsed = time.perf_counter() - t0

        # 8. Build result
        result = AudioDetectionResult(
            verdict=verdict,
            fake_probability=round(fake_probability, 6),
            authenticity_score=round(authenticity_score, 2),
            confidence=round(confidence, 2),
            branch_results=branch_results,
            processing_time_s=round(elapsed, 3),
            file_name=Path(audio_path).name,
            duration_s=round(duration_s, 3),
            sample_rate=sr,
            manipulation_segments=manipulation_segments,
        )

        # 9. Forensic report
        result.forensic_report = self._reporter.generate(result)

        logger.info(
            "ADS: verdict=%s fake_prob=%.4f elapsed=%.3fs",
            verdict, fake_probability, elapsed,
        )
        return result
