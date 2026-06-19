"""Hierarchical configuration management for ADS.

Uses YAML base configs with environment variable overrides via pydantic-settings.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelBackend(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


class AudioConfig(BaseSettings):
    sample_rate: int = 16000
    mono: bool = True
    max_duration_seconds: float = 30.0
    min_duration_seconds: float = 0.5
    chunk_duration_seconds: float = 5.0
    hop_duration_seconds: float = 2.5
    normalize_audio: bool = True
    remove_dc_offset: bool = True
    pre_emphasis: float = 0.97


class VadConfig(BaseSettings):
    model_id: str = "snakers4/silero-vad"
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    window_size_samples: int = 512
    speech_pad_ms: int = 30


class DiarizationConfig(BaseSettings):
    model_id: str = "pyannote/speaker-diarization-3.1"
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    threshold: float = 0.5
    onset: float = 0.7
    offset: float = 0.4


class SignalProcessingConfig(BaseSettings):
    n_mfcc: int = 40
    n_lfcc: int = 40
    n_cqcc: int = 40
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    f_min: float = 0.0
    f_max: Optional[float] = None
    power: float = 2.0
    n_chroma: int = 12
    n_bands: int = 7
    f0_min: float = 50.0
    f0_max: float = 500.0


class WavLMConfig(BaseSettings):
    model_id: str = "microsoft/wavlm-large"
    freeze_encoder: bool = True
    trainable_layers: int = 4
    dropout: float = 0.1
    embedding_dim: int = 1024
    projection_dim: int = 256
    use_cache: bool = True


class XLSRConfig(BaseSettings):
    model_id: str = "facebook/wav2vec2-xls-r-300m"
    freeze_encoder: bool = True
    trainable_layers: int = 4
    dropout: float = 0.1
    embedding_dim: int = 1024
    projection_dim: int = 256
    use_cache: bool = True


class SpectralCNNConfig(BaseSettings):
    backbone: str = "efficientnet-b0"
    input_channels: int = 3
    input_height: int = 128
    input_width: int = 128
    dropout: float = 0.2
    embedding_dim: int = 256
    use_pretrained: bool = True


class VoiceBiometricsConfig(BaseSettings):
    model_id: str = "speechbrain/spkrec-ecapa-voxceleb"
    embedding_dim: int = 192
    threshold: float = 0.5
    score_normalization: bool = True
    use_adaptive_threshold: bool = True


class TemporalSpliceConfig(BaseSettings):
    model_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    max_seq_len: int = 512
    use_crf: bool = True
    num_classes: int = 2


class DiffusionDetectorConfig(BaseSettings):
    model_dim: int = 128
    num_res_blocks: int = 4
    use_noise_residual: bool = True
    use_freq_fingerprint: bool = True
    embedding_dim: int = 128


class AdversarialDetectorConfig(BaseSettings):
    use_lid: bool = True
    use_mahalanobis: bool = True
    use_odin: bool = True
    temperature: float = 1.0
    noise_magnitude: float = 0.001
    epsilon: float = 0.01
    embedding_dim: int = 128


class FusionConfig(BaseSettings):
    fusion_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    use_adaptive_fusion: bool = True
    use_gating: bool = True
    output_dim: int = 1


class LocalizationConfig(BaseSettings):
    use_conformer: bool = True
    conformer_dim: int = 256
    conformer_num_layers: int = 4
    conformer_num_heads: int = 8
    use_crf: bool = True
    max_segments: int = 20
    min_segment_duration: float = 0.1
    threshold: float = 0.5


class ExplainabilityConfig(BaseSettings):
    use_shap: bool = True
    use_attention: bool = True
    use_spectrogram_heatmap: bool = True
    max_features_shap: int = 20
    background_samples: int = 50
    generate_pdf: bool = True
    generate_html: bool = True
    generate_json: bool = True


class PipelineConfig(BaseSettings):
    batch_size: int = 1
    num_workers: int = 0
    prefetch_factor: int = 2
    pin_memory: bool = False
    use_amp: bool = False
    compile_model: bool = False
    max_retries: int = 3
    timeout_seconds: int = 120


class MLflowConfig(BaseSettings):
    tracking_uri: str = Field(default="http://localhost:5000")
    experiment_name: str = "ads"
    registry_uri: str = "models:/ads-detector"
    artifact_location: Optional[str] = None
    run_name_prefix: str = "ads-run"


class DatabaseConfig(BaseSettings):
    url: str = Field(default="postgresql+asyncpg://ads:ads@localhost:5432/ads")
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20
    pool_pre_ping: bool = True
    pool_recycle: int = 3600


class RedisConfig(BaseSettings):
    url: str = Field(default="redis://localhost:6379/0")
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    max_connections: int = 20


class KafkaConfig(BaseSettings):
    bootstrap_servers: str = "localhost:9092"
    topic_prefix: str = "ads"
    consumer_group: str = "ads-consumer"
    max_poll_records: int = 100
    session_timeout: int = 30000


class SecurityConfig(BaseSettings):
    secret_key: str = Field(default="change-me-in-production")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    cors_origins: List[str] = ["*"]
    rate_limit_per_minute: int = 60
    max_upload_size_mb: int = 100
    enable_audit_log: bool = True
    allowed_audio_formats: List[str] = ["wav", "mp3", "flac", "ogg", "m4a", "aac", "opus"]
    bcrypt_rounds: int = 12


class TrainingConfig(BaseSettings):
    batch_size: int = 8
    eval_batch_size: int = 16
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 4
    num_workers: int = 0
    pin_memory: bool = False
    use_amp: bool = False
    compile_model: bool = False
    early_stop_patience: int = 10
    early_stop_metric: str = "val_eer"
    save_top_k: int = 3
    checkpoint_dir: str = Field(default="./models/checkpoints")
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 500
    use_weighted_sampler: bool = True
    freeze_encoder_epochs: int = 3
    lr_scheduler: str = "cosine"
    optimizer: str = "adamw"
    mixup_alpha: float = 0.2
    spec_augment: bool = True
    max_audio_duration: float = 8.0
    min_audio_duration: float = 1.0
    data_augmentation: List[str] = Field(default_factory=lambda: ["noise", "reverb", "speed"])

    # Model branches to train (subset of all 7)
    train_branches: List[str] = Field(
        default_factory=lambda: ["wavlm", "spectral", "voice_biometrics", "temporal"]
    )

    # Loss weights
    loss_bce_weight: float = 1.0
    loss_focal_gamma: float = 2.0
    loss_contrastive_weight: float = 0.1


class ModelServingConfig(BaseSettings):
    max_batch_size: int = 32
    max_wait_ms: int = 100
    use_dynamic_batching: bool = True
    warmup_models: bool = True
    cache_embeddings: bool = True
    cache_ttl_seconds: int = 3600


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ADS_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"
    project_root: str = str(Path(__file__).parent.parent.parent.parent)
    data_dir: str = Field(default="./data")
    model_dir: str = Field(default="./models")
    cache_dir: str = Field(default="./.cache")

    # Model backend
    model_backend: ModelBackend = ModelBackend.CPU
    device: str = "cpu"

    # Sub-configs
    audio: AudioConfig = AudioConfig()
    vad: VadConfig = VadConfig()
    diarization: DiarizationConfig = DiarizationConfig()
    signal: SignalProcessingConfig = SignalProcessingConfig()
    wavlm: WavLMConfig = WavLMConfig()
    xlsr: XLSRConfig = XLSRConfig()
    spectral_cnn: SpectralCNNConfig = SpectralCNNConfig()
    voice_biometrics: VoiceBiometricsConfig = VoiceBiometricsConfig()
    temporal_splice: TemporalSpliceConfig = TemporalSpliceConfig()
    diffusion_detector: DiffusionDetectorConfig = DiffusionDetectorConfig()
    adversarial_detector: AdversarialDetectorConfig = AdversarialDetectorConfig()
    fusion: FusionConfig = FusionConfig()
    localization: LocalizationConfig = LocalizationConfig()
    explainability: ExplainabilityConfig = ExplainabilityConfig()
    pipeline: PipelineConfig = PipelineConfig()
    training: TrainingConfig = TrainingConfig()
    mlflow: MLflowConfig = MLflowConfig()
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    kafka: KafkaConfig = KafkaConfig()
    security: SecurityConfig = SecurityConfig()
    serving: ModelServingConfig = ModelServingConfig()

    @field_validator("device", mode="before")
    @classmethod
    def resolve_device(cls, v: str, info: Any) -> str:
        if v and v != "auto":
            return v
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @classmethod
    def from_yaml(cls, path: str) -> "Settings":
        with open(path) as f:
            yaml_config = yaml.safe_load(f)
        env_config = cls().model_dump()
        merged = _deep_merge(yaml_config, env_config)
        return cls(**merged)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Settings":
        if config_path and os.path.exists(config_path):
            return cls.from_yaml(config_path)
        return cls()


def _deep_merge(source: dict, destination: dict) -> dict:
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            _deep_merge(value, node)
        else:
            destination[key] = value
    return destination


settings = Settings.load()
