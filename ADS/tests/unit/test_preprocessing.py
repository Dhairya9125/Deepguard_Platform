"""Unit tests for preprocessing pipeline."""

import pytest
import torch


class TestVAD:
    def test_silero_vad_initialization(self):
        from ads.preprocessing.vad import SileroVAD
        vad = SileroVAD()
        assert vad.threshold == 0.5
        assert vad.sample_rate == 16000

    def test_silero_vad_speech_detection(self):
        from ads.preprocessing.vad import SileroVAD
        vad = SileroVAD()
        waveform = torch.randn(16000)
        segments = vad.get_speech_timestamps(waveform)
        assert isinstance(segments, list)


class TestDiarization:
    def test_diarization_fallback(self):
        from ads.preprocessing.diarization import SpeakerDiarization
        dia = SpeakerDiarization()
        waveform = torch.randn(1, 16000)
        segments = dia.diarize(waveform, 16000)
        assert len(segments) > 0
        assert segments[0].speaker_id == "speaker_0"

    def test_count_speakers(self):
        from ads.preprocessing.diarization import SpeakerDiarization
        from ads.domain.entities import SpeakerSegment
        dia = SpeakerDiarization()
        segments = [
            SpeakerSegment(speaker_id="spk1", start_time=0.0, end_time=1.0, confidence=1.0),
            SpeakerSegment(speaker_id="spk2", start_time=1.0, end_time=2.0, confidence=1.0),
        ]
        assert dia.count_speakers(segments) == 2


class TestSegmentation:
    def test_fixed_window_segments(self):
        from ads.preprocessing.segmentation import AudioSegmenter
        segmenter = AudioSegmenter()
        waveform = torch.randn(1, 16000 * 10)
        chunks = segmenter.fixed_window_segments(waveform)
        assert len(chunks) > 0
        assert len(chunks[0]) == 3

    def test_merge_overlapping_windows(self):
        from ads.preprocessing.segmentation import AudioSegmenter
        segmenter = AudioSegmenter()
        preds = [(0.0, 1.0, 0.9), (0.8, 1.8, 0.85), (3.0, 4.0, 0.7)]
        merged = segmenter.merge_overlapping_windows(preds)
        assert len(merged) <= len(preds)

    def test_iou(self):
        from ads.preprocessing.segmentation import AudioSegmenter
        segmenter = AudioSegmenter()
        iou = segmenter._compute_iou((0.0, 1.0), (0.5, 1.5))
        assert 0 < iou < 1.0


class TestPreprocessingPipeline:
    def test_pipeline_initialization(self):
        from ads.preprocessing.pipeline import AudioPreprocessingPipeline
        pipeline = AudioPreprocessingPipeline()
        assert pipeline.vad is not None
        assert pipeline.diarization is not None
        assert pipeline.segmenter is not None

    def test_pipeline_process_synthetic(self, tmp_path):
        from ads.preprocessing.pipeline import AudioPreprocessingPipeline
        import torchaudio
        audio_path = tmp_path / "test.wav"
        waveform = torch.randn(1, 16000 * 3)
        torchaudio.save(str(audio_path), waveform, 16000)
        pipeline = AudioPreprocessingPipeline()
        result = pipeline.process(str(audio_path))
        assert "waveform" in result
        assert "metadata" in result
        assert "speech_segments" in result
        assert "chunks" in result
        assert result["preprocessing_time_ms"] > 0
