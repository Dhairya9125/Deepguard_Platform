"""
Configuration settings for the DeepGuard VDS API.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "DeepGuard VDS API"
    API_V1_STR: str = "/api/v1"
    
    # Storage settings
    BASE_DIR: Path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    STORAGE_DIR: Path = BASE_DIR / "storage"
    UPLOAD_DIR: Path = STORAGE_DIR / "uploads"
    ARTIFACTS_DIR: Path = STORAGE_DIR / "artifacts"
    
    class Config:
        case_sensitive = True

settings = Settings()

# Ensure directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
