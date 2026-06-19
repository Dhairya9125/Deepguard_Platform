"""Unit tests for domain entities."""

import pytest
from datetime import datetime
from ads.domain.entities import (
    AudioMetadata,
    AudioFormat,
    SpeechSegment,
    SpeakerSegment,
    ManipulatedSegment,
    ManipulationType,
    RiskLevel,
    ForensicReport,
    AnalysisResult,
    DetectionStatus,
    AnalysisStage,
    SignalFeatures,
    BranchOutput,
)


class TestAudioMetadata:
    def test_valid_metadata(self):
        meta = AudioMetadata(
            filename="test.wav",
            format=AudioFormat.WAV,
            duration_seconds=10.5,
            sample_rate=16000,
            channels=1,
            file_size_bytes=320000,
        )
        assert meta.duration_seconds == 10.5
        assert meta.format == AudioFormat.WAV

    def test_invalid_duration(self):
        with pytest.raises(ValueError):
            AudioMetadata(
                filename="test.wav",
                format=AudioFormat.WAV,
                duration_seconds=-1,
                sample_rate=16000,
                channels=1,
                file_size_bytes=320000,
            )


class TestSpeechSegment:
    def test_duration(self):
        seg = SpeechSegment(start_time=1.0, end_time=3.0, confidence=0.9)
        assert seg.duration == 2.0

    def test_defaults(self):
        seg = SpeechSegment(start_time=0.0, end_time=1.0)
        assert seg.confidence == 0.0
        assert seg.is_speech is True


class TestManipulatedSegment:
    def test_duration(self):
        seg = ManipulatedSegment(start_time=1.0, end_time=4.0, confidence=0.95)
        assert seg.duration == 3.0

    def test_default_manipulation_type(self):
        seg = ManipulatedSegment(start_time=0.0, end_time=1.0, confidence=0.5)
        assert seg.manipulation_type == ManipulationType.UNKNOWN


class TestForensicReport:
    def test_default_risk(self):
        report = ForensicReport(analysis_id="test-123")
        assert report.risk_assessment == RiskLevel.SAFE

    def test_markdown_generation(self):
        report = ForensicReport(
            analysis_id="test-123",
            executive_summary="Test summary",
            manipulated_segments=[
                ManipulatedSegment(start_time=1.0, end_time=2.0, confidence=0.9)
            ],
        )
        md = report.to_markdown()
        assert "Forensic Analysis Report" in md
        assert "Test summary" in md
        assert "1.00s - 2.00s" in md

    def test_html_generation(self):
        report = ForensicReport(
            analysis_id="test-123",
            risk_assessment=RiskLevel.HIGH,
            executive_summary="Test",
        )
        html = report.to_html()
        assert "Forensic Analysis Report" in html
        assert "red" in html


class TestAnalysisResult:
    def test_initial_state(self):
        result = AnalysisResult(audio_id="test.wav")
        assert result.status == DetectionStatus.PENDING
        assert result.current_stage == AnalysisStage.PREPROCESSING
        assert result.is_deepfake is False

    def test_risk_level_calculation(self):
        result = AnalysisResult(audio_id="test.wav")
        result.fake_probability = 0.95
        assert result.risk_level == RiskLevel.CRITICAL

        result.fake_probability = 0.05
        assert result.risk_level == RiskLevel.SAFE

        result.fake_probability = 0.45
        assert result.risk_level == RiskLevel.MEDIUM

    def test_complete(self):
        result = AnalysisResult(audio_id="test.wav")
        result.complete()
        assert result.status == DetectionStatus.COMPLETED
        assert result.completed_at is not None


class TestBranchOutput:
    def test_valid(self):
        output = BranchOutput(
            branch_name="wavlm",
            fake_probability=0.85,
            confidence=0.92,
        )
        assert output.fake_probability == 0.85
