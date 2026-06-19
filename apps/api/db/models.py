"""
DeepGuard Platform — Database Models
Module  : apps.api.db.models
Layer   : Infrastructure / ORM

SQLModel table definitions.  Each class here maps directly to a PostgreSQL table.

Tables:
  - User            : accounts that can log in and use the API
  - AnalysisJob     : one row per uploaded file (video / image / audio)
  - ForensicResult  : one-to-one with AnalysisJob, holds the final detection output
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlmodel import Column, Field, JSON, Relationship, SQLModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC time (timezone-naive)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _new_uuid() -> str:
    """Return a new UUID4 string — used as primary key default."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------

class User(SQLModel, table=True):
    """
    Represents a registered account on the DeepGuard platform.
    Passwords are never stored in plain text — only the bcrypt hash.
    """
    __tablename__ = "users"

    id: str = Field(default_factory=_new_uuid, primary_key=True, index=True)
    username: str = Field(unique=True, index=True, min_length=3, max_length=50)
    first_name: Optional[str] = Field(default=None, max_length=50)
    email: str = Field(unique=True, index=True, max_length=255)
    hashed_password: str = Field(max_length=255)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=_utcnow)

    # Reverse relationship — one user has many jobs and api keys
    jobs: list["AnalysisJob"] = Relationship(back_populates="user")
    api_keys: list["APIKey"] = Relationship(back_populates="user")


# ---------------------------------------------------------------------------
# AnalysisJob
# ---------------------------------------------------------------------------

class JobStatus(str):
    """String constants for AnalysisJob.status — not an Enum so DB stores plain text."""
    PENDING    = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED  = "COMPLETED"
    FAILED     = "FAILED"


class AnalysisJob(SQLModel, table=True):
    """
    One row per file uploaded for analysis.
    Replaces the fragile in-memory JOB_STORE dict in vds_pipeline.py.

    Status lifecycle:
        PENDING → PROCESSING → COMPLETED
                            → FAILED
    """
    __tablename__ = "analysis_jobs"

    id: str = Field(default_factory=_new_uuid, primary_key=True, index=True)
    modality: str = Field(max_length=10, description="'image', 'video', or 'audio'")
    status: str = Field(default=JobStatus.PENDING, max_length=20, index=True)
    message: Optional[str] = Field(default=None, max_length=500)
    file_name: Optional[str] = Field(default=None, max_length=255)

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    # Foreign key to User (nullable — allows anonymous submissions)
    user_id: Optional[str] = Field(default=None, foreign_key="users.id", index=True)

    # Relationships
    user: Optional[User] = Relationship(back_populates="jobs")
    result: Optional["ForensicResult"] = Relationship(back_populates="job")


# ---------------------------------------------------------------------------
# ForensicResult
# ---------------------------------------------------------------------------

class ForensicResult(SQLModel, table=True):
    """
    Stores the final detection output for a completed AnalysisJob.
    One-to-one relationship with AnalysisJob.

    raw_results holds the full JSON payload returned by the detection engine
    so nothing is ever lost — you can re-render any report from history.
    """
    __tablename__ = "forensic_results"

    id: str = Field(default_factory=_new_uuid, primary_key=True, index=True)
    job_id: str = Field(foreign_key="analysis_jobs.id", unique=True, index=True)

    verdict: str = Field(max_length=20, description="'REAL', 'FAKE', or 'UNCERTAIN'")
    fake_probability: float = Field(ge=0.0, le=1.0)
    authenticity_score: float = Field(ge=0.0, le=100.0)
    confidence: float = Field(ge=0.0, le=100.0)

    # Full JSON payload — stored as JSONB in PostgreSQL for fast querying
    raw_results: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    created_at: datetime = Field(default_factory=_utcnow)

    # Relationship
    job: Optional[AnalysisJob] = Relationship(back_populates="result")


# ---------------------------------------------------------------------------
# APIKey
# ---------------------------------------------------------------------------

class APIKey(SQLModel, table=True):
    """
    API Keys for programmatic access to the platform.
    """
    __tablename__ = "api_keys"

    id: str = Field(default_factory=_new_uuid, primary_key=True, index=True)
    key: str = Field(unique=True, index=True, max_length=100)
    name: str = Field(max_length=100, default="API Key")
    
    created_at: datetime = Field(default_factory=_utcnow)
    last_used_at: Optional[datetime] = Field(default=None)

    # Foreign key to User
    user_id: str = Field(foreign_key="users.id", index=True)

    # Relationship
    user: Optional[User] = Relationship(back_populates="api_keys")
