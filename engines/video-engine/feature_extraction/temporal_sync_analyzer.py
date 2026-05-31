"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.feature_extraction.temporal_sync_analyzer
Layer   : Layer 2 — Temporal Feature Extraction
Branch  : Branch D — Cross-Modal Synchronization

Orchestrates Branch D. Processes audio WAV + video frames to run SyncNet and
AV-HuBERT, correlating speech envelope/prosody with landmark motions.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import scipy.io.wavfile as wavfile
import torch

from .av_hubert_model import AVHuBERTActiveModel
from .result_types_l2d import BranchDResult, TrackSyncMetrics
from .sync_signals import (
    compute_audio_energy,
    compute_av_delay,
    compute_emotion_mismatch,
    compute_phoneme_viseme_mismatch,
)
from .syncnet_model import SyncNetMotionModel

logger = logging.getLogger(__name__)


class TemporalSyncAnalyzer:
    """
    VDS Layer 2, Branch D — Cross-Modal Synchronization.
    
    Orchestrates the evaluation of audio-visual lip sync and prosody/expression
    synchronization for video deepfake detection.
    """
    def __init__(
        self,
        device: str = "cpu",
        syncnet_checkpoint: str | Path | None = None,
        av_hubert_checkpoint: str | Path | None = None,
        thresholds: Dict[str, float] | None = None,
        weights: Dict[str, float] | None = None,
    ) -> None:
        self.device = device
        self.syncnet_checkpoint = syncnet_checkpoint
        self.av_hubert_checkpoint = av_hubert_checkpoint

        # ── Calibration System ─────────────────────────────────────────────
        self.thresholds = {
            "syncnet_confidence":          4.0,   # Confidence score from SyncNet (min required for real speech)
            "syncnet_offset_anomaly":      1,     # Max frame delay offset allowed before flagging delay anomaly
            "av_hubert_score":             0.60,  # Min probability for flagging as fake
            "phoneme_viseme_inconsistency": 0.55,  # Mismatch threshold between loudness & jaw opening
            "audio_video_delay":           0.15,  # Max AV delay in seconds allowed
            "emotion_mismatch":            0.60,  # Emotion prosody vs expression mismatch threshold
        }
        if thresholds:
            self.thresholds.update(thresholds)

        self.weights = {
            "syncnet_confidence":          3.0,   # High weight on SyncNet lip-sync mismatch
            "syncnet_offset_anomaly":      2.0,   # Lip-sync delay offset mismatch
            "av_hubert_score":             3.0,   # High weight on multimodal AV-HuBERT representation
            "phoneme_viseme_inconsistency": 2.0,   # Correlation viseme mismatch
            "audio_video_delay":           1.5,
            "emotion_mismatch":            1.5,
        }
        if weights:
            self.weights.update(weights)

        self.max_possible_score = sum(self.weights.values())

        # Lazy models loading container
        self._models_loaded = False
        self.syncnet: Optional[SyncNetMotionModel] = None
        self.av_hubert: Optional[AVHuBERTActiveModel] = None

        logger.info("TemporalSyncAnalyzer initialized | device=%s", device)

    def _lazy_load_models(self) -> None:
        if not self._models_loaded:
            logger.info("Lazily loading Branch D synchronization models ...")
            self.syncnet = SyncNetMotionModel(
                checkpoint_path=self.syncnet_checkpoint,
                device=self.device
            )
            self.av_hubert = AVHuBERTActiveModel(
                checkpoint_path=self.av_hubert_checkpoint,
                device=self.device
            )
            self._models_loaded = True
            logger.info("All Branch D models loaded.")

    def _extract_mouth_crop(self, face_crop: np.ndarray) -> np.ndarray:
        """Extract lower middle region of the aligned crop corresponding to the mouth."""
        if face_crop is None:
            # Return dummy empty crop
            return np.zeros((112, 112, 3), dtype=np.uint8)
        if face_crop.shape[:2] != (224, 224):
            face_crop = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        # Crop lower center region: y from 125 to 215, x from 56 to 168
        mouth = face_crop[125:215, 56:168]
        mouth_resized = cv2.resize(mouth, (112, 112), interpolation=cv2.INTER_LINEAR)
        return mouth_resized

    def _compute_numpy_mel_spec(self, audio_chunk: np.ndarray, num_mels: int = 13) -> np.ndarray:
        """
        Compute Mel-like spectrogram from 1D audio chunk entirely in NumPy.
        Produces a spectrogram of shape (13, 20) matching SyncNet input requirement.
        """
        if len(audio_chunk) < 256:
            # Pad
            audio_chunk = np.pad(audio_chunk, (0, 256 - len(audio_chunk)), mode="constant")
            
        segments = []
        step = len(audio_chunk) // 20
        for i in range(20):
            start = i * step
            end = start + 256
            segment = audio_chunk[start:end]
            if len(segment) < 256:
                segment = np.pad(segment, (0, 256 - len(segment)), mode="constant")
            
            # FFT Magnitude
            fft_mag = np.abs(np.fft.rfft(segment))[:64]
            group_size = max(1, len(fft_mag) // num_mels)
            mels = [np.mean(fft_mag[j*group_size : (j+1)*group_size]) for j in range(num_mels)]
            segments.append(mels)
            
        mel_spec = np.array(segments, dtype=np.float32).T  # (13, 20)
        mel_spec = np.log(mel_spec + 1e-6)
        return mel_spec

    def _run_syncnet_offsets(
        self, 
        crops: List[np.ndarray], 
        audio_signal: np.ndarray, 
        sample_rate: float, 
        fps: float
    ) -> Tuple[float, int]:
        """
        Evaluate SyncNet confidence and find the optimal offset by shifting 
        audio segments relative to the mouth crop sequence.
        """
        self._lazy_load_models()
        T = len(crops)
        if T < 5 or audio_signal is None or len(audio_signal) == 0:
            return 0.0, 0

        # We evaluate a clip of 5 frames from the center of the timeline
        center_frame = T // 2
        start_frame = max(0, center_frame - 2)
        end_frame = min(T, start_frame + 5)
        if end_frame - start_frame < 5:
            # Sequence too short
            return 0.0, 0

        # Extract 5 mouth crops and normalize to float32
        v_crops = [self._extract_mouth_crop(crops[f]) for f in range(start_frame, end_frame)]
        v_tensor = np.stack(v_crops, axis=0).astype(np.float32) / 255.0  # (5, 112, 112, 3)
        v_tensor = np.transpose(v_tensor, (3, 0, 1, 2))  # (3, 5, 112, 112)
        v_tensor = torch.from_numpy(v_tensor).unsqueeze(0).to(torch.device(self.device))  # (1, 3, 5, 112, 112)

        # Slide search window over audio shifts to find optimal sync (min distance / max confidence)
        best_conf = 0.0
        best_offset = 0
        
        # Search range of +/- 5 frames delay
        samples_per_frame = sample_rate / fps
        for offset in range(-5, 6):
            # Frame index mapped to audio sample index
            start_sample = int((start_frame + offset) * samples_per_frame)
            end_sample = start_sample + int(5 * samples_per_frame)
            
            # Pad if out of bounds
            if start_sample < 0 or end_sample > len(audio_signal):
                audio_chunk = np.zeros(int(5 * samples_per_frame), dtype=np.float32)
            else:
                audio_chunk = audio_signal[start_sample:end_sample]

            # Compute Mel spectrogram of shape (1, 1, 13, 20)
            mel_spec = self._compute_numpy_mel_spec(audio_chunk)
            a_tensor = torch.from_numpy(mel_spec).unsqueeze(0).unsqueeze(0).to(torch.device(self.device))

            with torch.no_grad():
                conf = float(self.syncnet.predict_sync_confidence(a_tensor, v_tensor).cpu().numpy()[0])

            if conf > best_conf:
                best_conf = conf
                best_offset = offset

        return best_conf, best_offset

    def _analyze_track(
        self,
        track_id: str,
        layer1_result,
        crops: List[np.ndarray],
        audio_signal: np.ndarray,
        sample_rate: float
    ) -> TrackSyncMetrics:
        """Evaluate Branch D synchronization signals for a single identity track."""
        n_frames = len(crops)
        valid_crops = [c for c in crops if c is not None]
        n_valid = len(valid_crops)

        # Retrieve landmarks
        landmarks_list = layer1_result.landmarks_for_track(track_id)
        
        coords = []
        jaw_open = []
        for lm in landmarks_list:
            if lm.detection_success and lm.landmarks_478 is not None:
                coords.append(lm.smoothed_landmarks)
                jaw_open.append(lm.jaw_open_ratio)

        coords_arr = np.array(coords) if coords else None
        jaw_open_arr = np.array(jaw_open) if jaw_open else None
        fps = layer1_result.metadata.target_fps

        # ── 1. NumPy Forensic Signal Calculations ──────────────────────────
        audio_energy = compute_audio_energy(audio_signal, sample_rate, n_frames, fps)
        delay_s, delay_conf = compute_av_delay(audio_energy, jaw_open_arr, fps)
        phoneme_viseme_mismatch = compute_phoneme_viseme_mismatch(audio_energy, jaw_open_arr)
        emotion_mismatch = compute_emotion_mismatch(audio_signal, coords_arr, fps)

        # ── 2. SyncNet Offset & Lip-Sync Evaluation ───────────────────────
        syncnet_conf, syncnet_offset = self._run_syncnet_offsets(
            crops, audio_signal, sample_rate, fps
        )

        # ── 3. AV-HuBERT Multimodal Fusion ────────────────────────────────
        self._lazy_load_models()
        
        # Prepare inputs for AV-HuBERT: resample mouth crops sequence to 16 frames
        # Select first 16 mouth crops
        av_crops = []
        for c in crops:
            if c is not None:
                av_crops.append(self._extract_mouth_crop(c))
        
        if not av_crops:
            av_crops = [np.zeros((112, 112, 3), dtype=np.uint8)] * 16
        while len(av_crops) < 16:
            av_crops.append(av_crops[-1])
        av_crops = av_crops[:16]
        
        v_input = np.stack(av_crops, axis=0).astype(np.float32) / 255.0  # (16, 112, 112, 3)
        v_input = np.transpose(v_input, (0, 3, 1, 2))  # (16, 3, 112, 112)
        v_tensor = torch.from_numpy(v_input).unsqueeze(0).to(torch.device(self.device))  # (1, 16, 3, 112, 112)

        # Resample matching audio segment: 16 frames * sample_rate / fps
        audio_samples_len = int(16 * sample_rate / fps)
        if audio_signal is not None and len(audio_signal) > 0:
            if len(audio_signal) >= audio_samples_len:
                a_input = audio_signal[:audio_samples_len]
            else:
                a_input = np.pad(audio_signal, (0, audio_samples_len - len(audio_signal)), mode="constant")
        else:
            a_input = np.zeros(audio_samples_len, dtype=np.float32)

        a_tensor = torch.from_numpy(a_input).unsqueeze(0).to(torch.device(self.device))  # (1, A)

        with torch.no_grad():
            av_hubert_prob = float(self.av_hubert.predict_probability(v_tensor, a_tensor).cpu().numpy()[0])

        # ── 4. Weighted Verdict System ─────────────────────────────────────
        signals_fired = []
        weighted_score = 0.0

        # Check SyncNet
        if syncnet_conf < self.thresholds["syncnet_confidence"]:
            signals_fired.append("syncnet_confidence")
            weighted_score += self.weights["syncnet_confidence"]

        if abs(syncnet_offset) > self.thresholds["syncnet_offset_anomaly"]:
            signals_fired.append("syncnet_offset_anomaly")
            weighted_score += self.weights["syncnet_offset_anomaly"]

        # Check AV-HuBERT
        if av_hubert_prob >= self.thresholds["av_hubert_score"]:
            signals_fired.append("av_hubert_score")
            weighted_score += self.weights["av_hubert_score"]

        # Check signals
        if phoneme_viseme_mismatch >= self.thresholds["phoneme_viseme_inconsistency"]:
            signals_fired.append("phoneme_viseme_inconsistency")
            weighted_score += self.weights["phoneme_viseme_inconsistency"]

        if abs(delay_s) >= self.thresholds["audio_video_delay"]:
            signals_fired.append("audio_video_delay")
            weighted_score += self.weights["audio_video_delay"]

        if emotion_mismatch >= self.thresholds["emotion_mismatch"]:
            signals_fired.append("emotion_mismatch")
            weighted_score += self.weights["emotion_mismatch"]

        # Calculate final confidence
        confidence = weighted_score / self.max_possible_score

        # Verdict
        if confidence >= 0.50:
            verdict = "FAKE"
            risk_level = "HIGH"
        elif confidence < 0.25:
            verdict = "REAL"
            risk_level = "LOW"
        else:
            verdict = "UNCERTAIN"
            risk_level = "MEDIUM"

        return TrackSyncMetrics(
            track_id=track_id,
            n_frames=n_frames,
            n_valid_frames=n_valid,
            syncnet_confidence=syncnet_conf,
            syncnet_offset=syncnet_offset,
            av_hubert_score=av_hubert_prob,
            phoneme_viseme_inconsistency=phoneme_viseme_mismatch,
            audio_video_delay=delay_s,
            emotion_mismatch_score=emotion_mismatch,
            sync_verdict=verdict,
            sync_confidence=confidence,
            sync_signals_fired=signals_fired,
            sync_risk_level=risk_level,
        )

    def analyze(self, layer1_result) -> BranchDResult:
        """
        Execute Branch D analysis on the Layer 1 Preprocessing results.
        """
        logger.info("VDS Layer 2 Branch D START | tracks=%s", layer1_result.unique_track_ids)
        t_start = time.perf_counter()
        warnings: List[str] = []

        track_metrics: Dict[str, TrackSyncMetrics] = {}

        if not layer1_result.unique_track_ids:
            warnings.append("No tracked faces in Layer 1 preprocessing result.")
            logger.warning("No tracks to process in Branch D.")
            return BranchDResult(processing_warnings=warnings)

        # 1. Load wav audio signal
        audio_signal = None
        sample_rate = 16000
        if layer1_result.metadata.has_audio and layer1_result.metadata.audio_path:
            try:
                sample_rate, audio_data = wavfile.read(str(layer1_result.metadata.audio_path))
                # Normalize float values
                if audio_data.dtype == np.int16:
                    audio_signal = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int8:
                    audio_signal = (audio_data.astype(np.float32) - 128.0) / 128.0
                else:
                    audio_signal = audio_data.astype(np.float32)
                
                # Convert multi-channel (stereo) to mono
                if audio_signal.ndim > 1:
                    audio_signal = np.mean(audio_signal, axis=1)
            except Exception as exc:
                warnings.append(f"Failed to read WAV audio: {exc}")
                logger.warning("WAV read failed (non-fatal): %s", exc)

        if audio_signal is None:
            warnings.append("No audio signal loaded — running with dummy audio data.")
            logger.warning("Audio unavailable. Using zero-padded audio stub.")
            audio_signal = np.zeros(16000, dtype=np.float32)

        # Map track crops
        track_crops: Dict[str, List[np.ndarray]] = {tid: [] for tid in layer1_result.unique_track_ids}
        for fp in layer1_result.frames:
            for tf in fp.tracked_faces:
                if tf.track_id in track_crops:
                    track_crops[tf.track_id].append(tf.aligned_crop)

        # Process each track
        for tid in layer1_result.unique_track_ids:
            crops = track_crops[tid]
            metrics = self._analyze_track(tid, layer1_result, crops, audio_signal, sample_rate)
            track_metrics[tid] = metrics
            logger.info(
                "  %s  →  syncnet_conf=%.2f (offset=%d)  av_hubert=%.4f  verdict=%s (conf=%.4f)",
                tid, metrics.syncnet_confidence, metrics.syncnet_offset,
                metrics.av_hubert_score, metrics.sync_verdict, metrics.sync_confidence
            )

        # Global aggregations
        syncnet_confs = [m.syncnet_confidence for m in track_metrics.values()]
        av_huberts = [m.av_hubert_score for m in track_metrics.values()]
        delays = [m.audio_video_delay for m in track_metrics.values()]
        mismatches = [m.emotion_mismatch_score for m in track_metrics.values()]

        fake_tracks = [m for m in track_metrics.values() if m.sync_verdict == "FAKE"]
        real_tracks = [m for m in track_metrics.values() if m.sync_verdict == "REAL"]

        n_fake = len(fake_tracks)
        n_real = len(real_tracks)
        n_uncertain = len(track_metrics) - n_fake - n_real

        # Majority vote
        if n_fake > n_real:
            overall_verdict = "FAKE"
        elif n_real > n_fake:
            overall_verdict = "REAL"
        else:
            overall_verdict = "UNCERTAIN"

        overall_conf = float(np.mean([m.sync_confidence for m in track_metrics.values()]))

        elapsed = time.perf_counter() - t_start
        logger.info(
            "VDS Layer 2 Branch D COMPLETE | elapsed=%.2fs | overall_verdict=%s (conf=%.4f)",
            elapsed, overall_verdict, overall_conf
        )

        return BranchDResult(
            track_metrics=track_metrics,
            overall_verdict=overall_verdict,
            overall_confidence=overall_conf,
            n_fake_tracks=n_fake,
            n_real_tracks=n_real,
            n_uncertain_tracks=n_uncertain,
            video_syncnet_conf_mean=float(np.mean(syncnet_confs)),
            video_av_hubert_mean=float(np.mean(av_huberts)),
            video_delay_mean=float(np.mean(delays)),
            video_emotion_mismatch_mean=float(np.mean(mismatches)),
            processing_warnings=warnings,
        )

    def unload(self) -> None:
        """Unload models to release GPU/CPU memory."""
        if self._models_loaded:
            try:
                del self.syncnet
                del self.av_hubert
                self.syncnet = None
                self.av_hubert = None
                self._models_loaded = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Branch D model weights successfully unloaded.")
            except Exception as e:
                logger.warning("Error unloading Branch D models: %s", e)
