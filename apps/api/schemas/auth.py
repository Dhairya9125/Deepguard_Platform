"""
DeepGuard Platform — Auth Pydantic Schemas
Module  : apps.api.schemas.auth
Layer   : API / Validation

Request and response models for the /api/v1/auth/* endpoints.
These are separate from the SQLModel DB models (apps.api.db.models)
to keep the API surface decoupled from the database schema.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class UserCreate(BaseModel):
    """Body for POST /api/v1/auth/register"""
    username: str = Field(..., min_length=3, max_length=50, examples=["alice"])
    first_name: str = Field(..., min_length=1, max_length=50, examples=["Alice"])
    email: EmailStr = Field(..., examples=["alice@example.com"])
    password: str = Field(..., min_length=8, max_length=128, examples=["str0ngP@ssw0rd"])


class UserLogin(BaseModel):
    """Body for POST /api/v1/auth/login"""
    email: str = Field(..., examples=["alice@example.com"])
    password: str = Field(..., examples=["str0ngP@ssw0rd"])

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8)


# ---------------------------------------------------------------------------
# Response bodies
# ---------------------------------------------------------------------------

class UserResponse(BaseModel):
    """Safe user representation — never includes hashed_password."""
    id: str
    username: str
    first_name: Optional[str] = None
    email: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True  # allows building from SQLModel ORM objects


class TokenResponse(BaseModel):
    """Response for a successful login — returns a JWT bearer token."""
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    user: UserResponse
