import secrets
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.db.database import get_session
from apps.api.db.models import APIKey, User
from apps.api.routers.auth import get_current_user
from apps.api.schemas.users import (
    APIKeyCreateRequest,
    APIKeyResponse,
    UserProfileResponse,
    UserProfileUpdateRequest,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# Profile Endpoints
# ---------------------------------------------------------------------------

@router.get("/me", response_model=UserProfileResponse)
async def get_my_profile(current_user: User = Depends(get_current_user)):
    """Get the currently logged-in user's profile."""
    return UserProfileResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        created_at=current_user.created_at,
    )


@router.patch("/me", response_model=UserProfileResponse)
async def update_my_profile(
    req: UserProfileUpdateRequest,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Update user profile information."""
    if req.username is not None:
        # Check if username is taken
        if req.username != current_user.username:
            existing = await db.execute(select(User).where(User.username == req.username))
            if existing.scalar_one_or_none():
                raise HTTPException(status_code=400, detail="Username already taken")
        current_user.username = req.username

    if req.email is not None:
        # Check if email is taken
        if req.email != current_user.email:
            existing = await db.execute(select(User).where(User.email == req.email))
            if existing.scalar_one_or_none():
                raise HTTPException(status_code=400, detail="Email already taken")
        current_user.email = req.email

    db.add(current_user)
    await db.commit()
    await db.refresh(current_user)

    return UserProfileResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        created_at=current_user.created_at,
    )


# ---------------------------------------------------------------------------
# API Key Endpoints
# ---------------------------------------------------------------------------

@router.get("/me/keys", response_model=List[APIKeyResponse])
async def list_my_keys(
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """List all API keys for the current user."""
    stmt = select(APIKey).where(APIKey.user_id == current_user.id).order_by(APIKey.created_at.desc())
    result = await db.execute(stmt)
    keys = result.scalars().all()
    
    return [
        APIKeyResponse(
            id=k.id,
            key=k.key,
            name=k.name,
            created_at=k.created_at,
            last_used_at=k.last_used_at
        )
        for k in keys
    ]


@router.post("/me/keys", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    req: APIKeyCreateRequest,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Generate a new API key."""
    # Enforce a limit (e.g. 10 keys per user) to prevent abuse
    stmt = select(APIKey).where(APIKey.user_id == current_user.id)
    result = await db.execute(stmt)
    if len(result.scalars().all()) >= 10:
        raise HTTPException(status_code=400, detail="Maximum number of API keys reached (10).")

    new_key_value = "dg_live_" + secrets.token_urlsafe(32)
    new_key = APIKey(
        key=new_key_value,
        name=req.name,
        user_id=current_user.id
    )
    
    db.add(new_key)
    await db.commit()
    await db.refresh(new_key)

    return APIKeyResponse(
        id=new_key.id,
        key=new_key.key,
        name=new_key.name,
        created_at=new_key.created_at,
        last_used_at=new_key.last_used_at
    )


@router.delete("/me/keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: str,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Revoke and delete an API key."""
    stmt = select(APIKey).where(APIKey.id == key_id).where(APIKey.user_id == current_user.id)
    result = await db.execute(stmt)
    key_obj = result.scalar_one_or_none()
    
    if not key_obj:
        raise HTTPException(status_code=404, detail="API key not found")
        
    await db.delete(key_obj)
    await db.commit()
    return None
