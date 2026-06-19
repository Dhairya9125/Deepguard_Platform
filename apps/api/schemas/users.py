from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field

class APIKeyResponse(BaseModel):
    id: str
    key: str
    name: str
    created_at: datetime
    last_used_at: Optional[datetime]

class APIKeyCreateRequest(BaseModel):
    name: str = Field(..., max_length=100)

class UserProfileResponse(BaseModel):
    id: str
    username: str
    email: str
    created_at: datetime

class UserProfileUpdateRequest(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[str] = Field(None, max_length=255)
