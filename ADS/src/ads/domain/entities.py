"""Domain entities for the Audio Deepfake Detection System.

Follows Domain-Driven Design with rich domain models,
value objects, and aggregates.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


class AudioFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"
    AAC = "aac"
    OPUS = "opus"


class ManipulationType(str, Enum):
    AI_VOICE_CLONE = "ai_voice_clone"
    SYNTHETIC_SPEECH = "synthetic_speech"
    VOICE_CONVERSION = "voice_conversion"
    DIFFUSION_ATTACK = "diffusion_attack"
    AUDIO_SPLICING = "audio_splicing"
    PARTIAL_MANIPULATION = "partial_manipulation"
    SPEAKER_IMPERSONATION = "speaker_impersonation"
    ADVERSARIAL_ATTACK = "adversarial_attack"
    PHRASE_INSERTION = "phrase_insertion"
    AUDIO_REPLACEMENT = "audio_replacement"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DetectionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AnalysisStage(str, Enum):
    PREPROCESSING = "preprocessing"
    FEATURE_EXTRACTION = "feature_extraction"
    BRANCH_INFERENCE = "branch_inference"
    FUSION = "fusion"
    LOCALIZATION = "localization"
    EXPLAINABILITY = "explainability"
    COMPLETE = "complete"


class AudioMetadata(BaseModel):
    filename: str
    format: AudioFormat
    duration_seconds: float
    sample_rate: int
    channels: int
    bit_depth: Optional[int] = None
    bit_rate: Optional[int] = None
    file_size_bytes: int
    hash_md5: Optional[str] = None
    hash_sha256: Optional[str] = None
    language: Optional[str] = None
    creation_date: Optional[datetime] = None

    @field_validator("duration_seconds")
    @classmethod
    def validate_duration(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Duration must be positive")
        return v


class SpeechSegment(BaseModel):
    start_time: float
    end_time: float
    confidence: float = 0.0
    speaker_id: Optional[str] = None
    is_speech: bool = True
    label: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class SpeakerSegment(BaseModel):
    speaker_id: str
    start_time: float
    end_time: float
    confidence: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class SignalFeatures(BaseModel):
    waveform: Optional[Any] = None
    mel_spectrogram: Optional[Any] = None
    mfcc: Optional[Any] = None
    lfcc: Optional[Any] = None
    cqcc: Optional[Any] = None
    chroma: Optional[Any] = None
    spectral_contrast: Optional[Any] = None
    pitch: Optional[Any] = None
    harmonic: Optional[Any] = None
    perceptual: Optional[Any] = None


class BranchOutput(BaseModel):
    branch_name: str
    fake_probability: float = 0.0
    confidence: float = 0.0
    embedding: Optional[Any] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: float = 0.0


class ManipulatedSegment(BaseModel):
    start_time: float
    end_time: float
    confidence: float = 0.0
    manipulation_type: ManipulationType = ManipulationType.UNKNOWN
    severity: RiskLevel = RiskLevel.MEDIUM

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class ForensicTimeline(BaseModel):
    segments: List[ManipulatedSegment] = Field(default_factory=list)
    overall_start: float = 0.0
    overall_end: float = 0.0

    @property
    def total_manipulated_duration(self) -> float:
        return sum(s.duration for s in self.segments)

    @property
    def manipulation_ratio(self) -> float:
        total = self.overall_end - self.overall_start
        if total <= 0:
            return 0.0
        return self.total_manipulated_duration / total


class ShapExplanation(BaseModel):
    feature_name: str
    importance: float
    direction: str  # "increases_fake" or "increases_real"


class AttentionMap(BaseModel):
    timestep: float
    attention_weight: float
    branch_name: str


class ForensicReport(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    analysis_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    executive_summary: str = ""
    risk_assessment: RiskLevel = RiskLevel.SAFE
    risk_score: float = 0.0
    technical_findings: Dict[str, Any] = Field(default_factory=dict)
    branch_summaries: Dict[str, BranchOutput] = Field(default_factory=dict)
    manipulated_segments: List[ManipulatedSegment] = Field(default_factory=list)
    timeline: Optional[ForensicTimeline] = None
    shap_explanations: List[ShapExplanation] = Field(default_factory=list)
    attention_maps: List[AttentionMap] = Field(default_factory=list)
    confidence_score: float = 0.0
    processing_time_ms: float = 0.0
    model_version: str = ""
    warnings: List[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            f"# Forensic Analysis Report",
            f"**Report ID:** {self.id}",
            f"**Analysis ID:** {self.analysis_id}",
            f"**Created:** {self.created_at.isoformat()}",
            f"**Confidence:** {self.confidence_score:.2%}",
            f"**Risk Level:** {self.risk_assessment.value}",
            f"**Risk Score:** {self.risk_score:.2%}",
            f"",
            f"## Executive Summary",
            f"{self.executive_summary}",
            f"",
            f"## Technical Findings",
            f"- Manipulated segments detected: {len(self.manipulated_segments)}",
            f"- Total processing time: {self.processing_time_ms:.1f}ms",
            f"- Model version: {self.model_version}",
            f"",
        ]
        for seg in self.manipulated_segments:
            lines.append(
                f"- [{seg.start_time:.2f}s - {seg.end_time:.2f}s] "
                f"Type: {seg.manipulation_type.value}, "
                f"Confidence: {seg.confidence:.2%}, "
                f"Severity: {seg.severity.value}"
            )
        return "\n".join(lines)

    def to_html(self) -> str:
        html = "<html><body>"
        html += f"<h1>Forensic Analysis Report</h1>"
        html += f"<p><strong>Report ID:</strong> {self.id}</p>"
        html += f"<p><strong>Risk Level:</strong> <span style='color:{self._risk_color()}'>{self.risk_assessment.value}</span></p>"
        html += f"<h2>Executive Summary</h2><p>{self.executive_summary}</p>"
        html += f"<h2>Manipulated Segments</h2><ul>"
        for seg in self.manipulated_segments:
            html += f"<li>{seg.start_time:.2f}s - {seg.end_time:.2f}s: {seg.manipulation_type.value} ({seg.confidence:.2%})</li>"
        html += "</ul></body></html>"
        return html

    def _risk_color(self) -> str:
        return {
            RiskLevel.SAFE: "green",
            RiskLevel.LOW: "yellow",
            RiskLevel.MEDIUM: "orange",
            RiskLevel.HIGH: "red",
            RiskLevel.CRITICAL: "darkred",
        }.get(self.risk_assessment, "gray")


class AnalysisResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    audio_id: str
    status: DetectionStatus = DetectionStatus.PENDING
    current_stage: AnalysisStage = AnalysisStage.PREPROCESSING
    audio_metadata: Optional[AudioMetadata] = None
    speech_segments: List[SpeechSegment] = Field(default_factory=list)
    speaker_segments: List[SpeakerSegment] = Field(default_factory=list)
    signal_features: Optional[SignalFeatures] = None
    branch_outputs: Dict[str, BranchOutput] = Field(default_factory=dict)
    fake_probability: float = 0.0
    is_deepfake: bool = False
    confidence_score: float = 0.0
    speaker_consistency_score: float = 1.0
    fusion_embedding: Optional[Any] = None
    manipulated_segments: List[ManipulatedSegment] = Field(default_factory=list)
    timeline: Optional[ForensicTimeline] = None
    explanations: Optional[ForensicReport] = None
    error_message: Optional[str] = None
    processing_time_ms: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    @property
    def risk_level(self) -> RiskLevel:
        if self.fake_probability < 0.1:
            return RiskLevel.SAFE
        if self.fake_probability < 0.3:
            return RiskLevel.LOW
        if self.fake_probability < 0.6:
            return RiskLevel.MEDIUM
        if self.fake_probability < 0.85:
            return RiskLevel.HIGH
        return RiskLevel.CRITICAL

    @property
    def duration_seconds(self) -> float:
        if self.audio_metadata:
            return self.audio_metadata.duration_seconds
        return 0.0

    def complete(self) -> None:
        self.status = DetectionStatus.COMPLETED
        self.current_stage = AnalysisStage.COMPLETE
        self.completed_at = datetime.utcnow()


class AudioRecording(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    file_path: str
    metadata: AudioMetadata
    upload_time: datetime = Field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    is_sample: bool = False

    @property
    def duration(self) -> float:
        return self.metadata.duration_seconds


class ModelVersion(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    version: str
    stage: str = "None"
    run_id: str = ""
    source: str = ""
    status: str = "active"
    metrics: Dict[str, float] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    artifact_uri: Optional[str] = None
    signature: Optional[str] = None
    description: str = ""


class DatasetInfo(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    version: str
    source: str
    description: str = ""
    num_samples: int = 0
    num_real: int = 0
    num_fake: int = 0
    languages: List[str] = Field(default_factory=list)
    manipulation_types: List[ManipulationType] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    checksum: Optional[str] = None
