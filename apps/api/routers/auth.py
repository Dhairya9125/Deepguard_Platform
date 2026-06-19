"""
DeepGuard Platform — Authentication Router
Module  : apps.api.routers.auth
Layer   : API

Endpoints:
  POST /api/v1/auth/register  — Create a new user account
  POST /api/v1/auth/login     — Get a JWT bearer token
  GET  /api/v1/auth/me        — Return current user info (requires JWT)

JWT tokens are passed in the Authorization header:
  Authorization: Bearer <token>
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Header
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from apps.api.core.config import settings
from apps.api.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from datetime import timedelta
from apps.api.db.database import get_session
from apps.api.db.models import User, APIKey
from apps.api.schemas.auth import (
    TokenResponse, UserCreate, UserLogin, UserResponse,
    ForgotPasswordRequest, ResetPasswordRequest
)
from apps.api.services.email import send_reset_password_email
import os

router = APIRouter()
_bearer = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Helper — get current user from JWT
# ---------------------------------------------------------------------------

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_session),
) -> User:
    """
    FastAPI dependency that extracts and validates the JWT from the
    Authorization header, then returns the corresponding User from the DB.

    Raises HTTP 401 if the token is missing, invalid, or expired.
    Raises HTTP 403 if the account is disabled.

    Usage in any protected endpoint:
        async def my_endpoint(current_user: User = Depends(get_current_user)):
            ...
    """
    # Check API key first
    if x_api_key:
        result = await db.execute(select(APIKey).where(APIKey.key == x_api_key))
        api_key_obj = result.scalar_one_or_none()
        if not api_key_obj:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key.",
            )
        # Update last used
        from datetime import datetime, timezone
        api_key_obj.last_used_at = datetime.now(timezone.utc).replace(tzinfo=None)
        db.add(api_key_obj)
        await db.commit()
        
        user_id = api_key_obj.user_id
    elif credentials:
        user_id = decode_access_token(credentials.credentials)
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token.",
                headers={"WWW-Authenticate": "Bearer"},
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Provide a Bearer token or X-API-Key header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled.")

    return user


# ---------------------------------------------------------------------------
# POST /register
# ---------------------------------------------------------------------------

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
async def register(body: UserCreate, db: AsyncSession = Depends(get_session)):
    """
    Create a new DeepGuard account.

    - **username**: 3-50 characters, must be unique
    - **email**: valid email address, must be unique
    - **password**: minimum 8 characters (stored as bcrypt hash — never in plain text)
    """
    # Check for duplicate username
    existing = await db.execute(select(User).where(User.username == body.username))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username '{body.username}' is already taken.",
        )

    # Check for duplicate email
    existing_email = await db.execute(select(User).where(User.email == body.email))
    if existing_email.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Email '{body.email}' is already registered.",
        )

    user = User(
        username=body.username,
        first_name=body.first_name,
        email=body.email,
        hashed_password=hash_password(body.password),
    )
    db.add(user)
    await db.flush()   # assigns user.id before commit
    await db.refresh(user)

    return UserResponse.model_validate(user)


# ---------------------------------------------------------------------------
# POST /login
# ---------------------------------------------------------------------------

@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Log in and receive a JWT access token",
)
async def login(body: UserLogin, db: AsyncSession = Depends(get_session)):
    """
    Authenticate with email + password.
    Returns a JWT bearer token valid for **7 days** by default.

    Pass the token in all subsequent requests:
    ```
    Authorization: Bearer <your_token_here>
    ```
    """
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()

    # Use a single error message for both "user not found" and "wrong password"
    # to prevent email enumeration attacks
    invalid_creds = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid email or password.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not user or not verify_password(body.password, user.hashed_password):
        raise invalid_creds

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled.")

    token = create_access_token(subject=user.id)

    return TokenResponse(
        access_token=token,
        expires_in_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        user=UserResponse.model_validate(user),
    )


# ---------------------------------------------------------------------------
# GET /me
# ---------------------------------------------------------------------------

@router.get(
    "/me",
    response_model=UserResponse,
    summary="Return info about the currently logged-in user",
)
async def me(current_user: User = Depends(get_current_user)):
    """
    Requires a valid JWT bearer token.
    Returns the profile of the currently authenticated user.
    """
    return UserResponse.model_validate(current_user)


# ---------------------------------------------------------------------------
# POST /forgot-password
# ---------------------------------------------------------------------------

@router.post(
    "/forgot-password",
    status_code=status.HTTP_200_OK,
    summary="Request a password reset email",
)
async def forgot_password(body: ForgotPasswordRequest, db: AsyncSession = Depends(get_session)):
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()

    if user and user.is_active:
        # Create a token valid for 15 minutes, with subject prefix 'reset:'
        token = create_access_token(
            subject=f"reset:{user.id}",
            expires_delta=timedelta(minutes=15)
        )
        app_url = os.environ.get("NEXT_PUBLIC_APP_URL", "http://localhost:3000")
        reset_link = f"{app_url}/reset-password?token={token}"
        
        # We don't await this because it's synchronous in our email.py
        # For production, we'd use a background task
        send_reset_password_email(user.email, reset_link)

    # Always return a success message so we don't leak which emails exist
    return {"message": "If an account with that email exists, we sent a password reset link."}


# ---------------------------------------------------------------------------
# POST /reset-password
# ---------------------------------------------------------------------------

@router.post(
    "/reset-password",
    status_code=status.HTTP_200_OK,
    summary="Reset password using a token",
)
async def reset_password(body: ResetPasswordRequest, db: AsyncSession = Depends(get_session)):
    subject = decode_access_token(body.token)
    
    if not subject or not subject.startswith("reset:"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired password reset token."
        )
        
    user_id = subject.split("reset:")[1]
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user."
        )
        
    user.hashed_password = hash_password(body.new_password)
    db.add(user)
    await db.commit()
    
    return {"message": "Password has been successfully reset."}

