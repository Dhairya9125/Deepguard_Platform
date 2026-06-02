"""Database models and connection management using SQLAlchemy async.

Supports PostgreSQL for production and SQLite for development/testing.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import AsyncGenerator, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from ads.config.settings import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class AudioRecordingModel(Base):
    __tablename__ = "audio_recordings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    sample_rate: Mapped[int] = mapped_column(Integer, nullable=False)
    channels: Mapped[int] = mapped_column(Integer, default=1)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    audio_format: Mapped[str] = mapped_column(String(10), nullable=False)
    hash_sha256: Mapped[Optional[str]] = mapped_column(String(64))
    user_id: Mapped[Optional[str]] = mapped_column(String(36))
    session_id: Mapped[Optional[str]] = mapped_column(String(36))
    upload_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    is_sample: Mapped[bool] = mapped_column(Boolean, default=False)

    analyses = relationship("AnalysisResultModel", back_populates="audio_recording")


class AnalysisResultModel(Base):
    __tablename__ = "analysis_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    audio_id: Mapped[str] = mapped_column(String(36), ForeignKey("audio_recordings.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    fake_probability: Mapped[float] = mapped_column(Float, default=0.0)
    is_deepfake: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    speaker_consistency_score: Mapped[float] = mapped_column(Float, default=1.0)
    processing_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    audio_recording = relationship("AudioRecordingModel", back_populates="analyses")
    report = relationship("ForensicReportModel", uselist=False, back_populates="analysis")


class ForensicReportModel(Base):
    __tablename__ = "forensic_reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    analysis_id: Mapped[str] = mapped_column(String(36), ForeignKey("analysis_results.id"), nullable=False)
    risk_level: Mapped[str] = mapped_column(String(20), default="safe")
    executive_summary: Mapped[str] = mapped_column(Text, default="")
    technical_findings: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    report_json: Mapped[Optional[str]] = mapped_column(Text)

    analysis = relationship("AnalysisResultModel", back_populates="report")


class UserModel(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), default="viewer")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime)


class AuditLogModel(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(36))
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(50))
    resource_id: Mapped[Optional[str]] = mapped_column(String(36))
    details: Mapped[Optional[str]] = mapped_column(Text)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ModelRegistryModel(Base):
    __tablename__ = "model_registry"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(20), nullable=False)
    stage: Mapped[str] = mapped_column(String(20), default="None")
    run_id: Mapped[str] = mapped_column(String(36))
    source: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(20), default="active")
    metrics: Mapped[Optional[str]] = mapped_column(Text)
    params: Mapped[Optional[str]] = mapped_column(Text)
    artifact_uri: Mapped[Optional[str]] = mapped_column(String(1024))
    signature: Mapped[Optional[str]] = mapped_column(Text)
    description: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AsyncDatabaseManager:
    """Manages async database connections and sessions."""

    def __init__(self, url: Optional[str] = None) -> None:
        self.url = url or settings.database.url
        self._engine = None
        self._session_factory = None

    async def initialize(self) -> None:
        self._engine = create_async_engine(self.url, echo=settings.database.echo)
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialized")

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        if self._session_factory is None:
            await self.initialize()
        async with self._session_factory() as session:
            yield session

    async def close(self) -> None:
        if self._engine:
            await self._engine.dispose()


db_manager = AsyncDatabaseManager()
