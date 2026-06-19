"""Main Detector Service - Orchestrates the entire deepfake detection pipeline.

Coordinates:
1. Audio preprocessing (VAD, diarization, segmentation)
2. Signal processing (feature extraction)
3. Multi-branch inference (7 model branches)
4. Cross-branch fusion
5. Temporal localization
6. Explainability and report generation
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import torch

from ads.config.settings import settings
from ads.domain.entities import (
    AnalysisResult,
    AnalysisStage,
    AudioMetadata,
    BranchOutput,
    DetectionStatus,
    ManipulatedSegment,
)
from ads.explainability.engine import ExplainabilityEngine
from ads.fusion.engine import FusionTransformer
from ads.localization.engine import LocalizationEngine
from ads.models.adversarial_detector.model import AdversarialDetectorBranch
from ads.models.diffusion_detector.model import DiffusionDetectorBranch
from ads.models.spectral_cnn_branch.model import SpectralCNN
from ads.models.temporal_splice.model import TemporalSpliceBranch
from ads.models.voice_biometrics.model import VoiceBiometricsBranch
from ads.models.wavlm_branch.model import WavLMBranch
from ads.models.xlsr_branch.model import XLSRBranch
from ads.preprocessing.pipeline import AudioPreprocessingPipeline
from ads.signal_processing.features import SignalProcessor

logger = logging.getLogger(__name__)


class DeepfakeDetector:
    """Main orchestrator for deepfake detection.

    Manages the complete detection pipeline from audio input
    to forensic report generation.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        self.device = torch.device("cpu")

        self.preprocessor = AudioPreprocessingPipeline(config)
        self.signal_processor = SignalProcessor(config)

        self.wavlm_branch = WavLMBranch(config).to(self.device).eval()
        self.xlsr_branch = XLSRBranch(config).to(self.device).eval()
        self.spectral_cnn = SpectralCNN(config).to(self.device).eval()
        self.voice_biometrics = VoiceBiometricsBranch(config).to(self.device).eval()
        self.temporal_splice = TemporalSpliceBranch(config).to(self.device).eval()
        self.diffusion_detector = DiffusionDetectorBranch(config).to(self.device).eval()
        self.adversarial_detector = AdversarialDetectorBranch(config).to(self.device).eval()

        self.fusion_engine = FusionTransformer(config).to(self.device).eval()
        self.localization_engine = LocalizationEngine(config).to(self.device).eval()
        self.explainability = ExplainabilityEngine(config)

        logger.info("DeepfakeDetector initialized with all branches on CPU")

    def analyze(
        self,
        audio_path: str,
        return_report: bool = True,
        output_dir: Optional[str] = None,
    ) -> AnalysisResult:
        """Run complete deepfake analysis on audio file.

        Args:
            audio_path: Path to audio file
            return_report: Generate forensic report
            output_dir: Save report files to this directory

        Returns:
            AnalysisResult with detection outcome and optional report
        """
        start_time = time.time()
        result = AnalysisResult(
            audio_id=audio_path.split("/")[-1].split("\\")[-1],
            status=DetectionStatus.PROCESSING,
        )

        try:
            result.current_stage = AnalysisStage.PREPROCESSING
            preproc = self.preprocessor.process(audio_path)
            result.audio_metadata = preproc.get("metadata")
            result.speech_segments = preproc.get("speech_segments", [])
            result.speaker_segments = preproc.get("speaker_segments", [])
            waveform = preproc.get("waveform")
            chunks = preproc.get("chunks", [])

            result.current_stage = AnalysisStage.FEATURE_EXTRACTION
            signal_features = self.signal_processor.compute_all_features(waveform)
            result.signal_features = signal_features

            result.current_stage = AnalysisStage.BRANCH_INFERENCE
            branch_outputs = self._run_branches(waveform, signal_features)
            result.branch_outputs = {
                name: BranchOutput(
                    branch_name=name,
                    fake_probability=float(out.get("probability", out.get("confidence", 0.5))),
                    confidence=float(out.get("confidence", 0.5)),
                    processing_time_ms=out.get("processing_time_ms", 0),
                )
                for name, out in branch_outputs.items()
            }

            result.current_stage = AnalysisStage.FUSION
            fusion_result = self._run_fusion(branch_outputs)
            result.fake_probability = float(fusion_result.get("probability", 0.5))
            result.is_deepfake = result.fake_probability > 0.5
            result.confidence_score = float(fusion_result.get("confidence", 0.5))
            result.fusion_embedding = fusion_result.get("embedding")

            result.current_stage = AnalysisStage.LOCALIZATION
            if "temporal" in branch_outputs:
                temporal_out = branch_outputs["temporal"]
                loc_result = self.localization_engine.forward(
                    frame_features=temporal_out.get("seq_embedding", torch.zeros(1, 256)).unsqueeze(0),
                    frame_logits=temporal_out.get("frame_logits", torch.zeros(1, 10, 2)),
                )
                result.manipulated_segments = loc_result.get("manipulated_segments", [])
                result.timeline = loc_result.get("timeline")

            result.current_stage = AnalysisStage.EXPLAINABILITY
            if return_report:
                analysis_data = result.model_dump(mode="json")
                analysis_data.update({
                    "branch_outputs": branch_outputs,
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "model_version": "0.1.0",
                })
                report = self.explainability.generate_report(analysis_data, output_dir)
                result.explanations = report

            result.complete()
            elapsed = (time.time() - start_time) * 1000
            result.processing_time_ms = elapsed

            logger.info(
                f"Analysis completed in {elapsed:.0f}ms: "
                f"fake_prob={result.fake_probability:.3f}, "
                f"confidence={result.confidence_score:.3f}, "
                f"segments={len(result.manipulated_segments)}"
            )

        except Exception as e:
            result.status = DetectionStatus.FAILED
            result.error_message = str(e)
            logger.error(f"Analysis failed: {e}", exc_info=True)

        return result

    @torch.no_grad()
    def _run_branches(
        self,
        waveform: torch.Tensor,
        signal_features: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """Run all model branches and collect outputs."""
        if waveform.dim() == 2:
            waveform_1d = waveform.mean(dim=0)
        else:
            waveform_1d = waveform

        waveform_1d = waveform_1d.to(self.device)
        outputs = {}

        branches = {
            "wavlm": (self.wavlm_branch, {"waveform": waveform_1d}),
            "xlsr": (self.xlsr_branch, {"waveform": waveform_1d}),
            "spectral": (self.spectral_cnn, {"spectral_features": self._prepare_spectral_input(signal_features)}),
            "voice_biometrics": (self.voice_biometrics, {"features": self._prepare_biometric_input(signal_features)}),
            "diffusion": (self.diffusion_detector, {"waveform": waveform_1d}),
            "adversarial": (self.adversarial_detector, {"waveform": waveform_1d}),
        }

        for name, (model, kwargs) in branches.items():
            try:
                t0 = time.time()
                branch_out = model(**kwargs)
                elapsed = (time.time() - t0) * 1000
                output = {}
                for k, v in branch_out.items():
                    if isinstance(v, torch.Tensor):
                        output[k] = v.cpu()
                    else:
                        output[k] = v
                output["processing_time_ms"] = elapsed
                outputs[name] = output
            except Exception as e:
                logger.warning(f"Branch {name} failed: {e}")
                outputs[name] = {
                    "logits": torch.tensor([[0.0]]),
                    "confidence": torch.tensor([[0.5]]),
                    "probability": torch.tensor([[0.5]]),
                    "embedding": torch.zeros(1, 256),
                    "processing_time_ms": 0,
                }

        # Temporal branch needs sequential features
        try:
            t0 = time.time()
            feat_seq = self._prepare_temporal_input(waveform_1d)
            temporal_out = self.temporal_splice(feat_seq)
            elapsed = (time.time() - t0) * 1000
            temporal_out["processing_time_ms"] = elapsed
            outputs["temporal"] = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in temporal_out.items()}
        except Exception as e:
            logger.warning(f"Temporal branch failed: {e}")
            outputs["temporal"] = {"processing_time_ms": 0}

        return outputs

    def _run_fusion(
        self, branch_outputs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run fusion engine on branch outputs."""
        embeddings = {}
        confidences = {}

        for name, output in branch_outputs.items():
            emb = output.get("embedding")
            if isinstance(emb, torch.Tensor):
                embeddings[name] = emb
            conf = output.get("confidence")
            if isinstance(conf, torch.Tensor):
                confidences[name] = conf

        fusion_out = self.fusion_engine(embeddings, confidences)
        return {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in fusion_out.items()}

    def _prepare_spectral_input(self, signal_features: Any) -> torch.Tensor:
        """Prepare spectral features for CNN branch."""
        feat = SignalProcessor.stack_features(signal_features)
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)
        return feat.to(self.device)

    def _prepare_biometric_input(self, signal_features: Any) -> torch.Tensor:
        """Prepare features for voice biometrics branch."""
        mfcc = signal_features.mfcc
        if isinstance(mfcc, torch.Tensor):
            if mfcc.dim() == 2:
                mfcc = mfcc.T.unsqueeze(0)
            return mfcc.to(self.device)
        return torch.randn(1, 100, 40).to(self.device)

    def _prepare_temporal_input(self, waveform: torch.Tensor) -> torch.Tensor:
        """Prepare sequential features for temporal branch."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        mel = self.signal_processor.compute_mel_spectrogram(waveform)
        if mel.dim() == 2:
            mel = mel.T.unsqueeze(0)
        elif mel.dim() == 3:
            mel = mel.transpose(1, 2)

        proj = torch.nn.Linear(mel.shape[-1], 1024)
        return proj(mel).to(self.device)
