"""
Configuration settings for the DeepGuard Platform API.

All values are driven by environment variables or a .env file.
For local development, copy .env.example → .env and fill in your Neon DB URL.

Database: Neon Serverless PostgreSQL (https://neon.tech)
  - Free forever tier, no Docker required
  - Requires SSL (handled automatically via connect_args in database.py)

See: https://docs.pydantic.dev/latest/concepts/pydantic_settings/"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "DeepGuard VDS API"
    API_V1_STR: str = "/api/v1"

    # ------------------------------------------------------------------
    # Storage settings
    # ------------------------------------------------------------------
    BASE_DIR: Path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    STORAGE_DIR: Path = BASE_DIR / "storage"
    UPLOAD_DIR: Path = STORAGE_DIR / "uploads"
    ARTIFACTS_DIR: Path = STORAGE_DIR / "artifacts"

    # ------------------------------------------------------------------
    # Database — Neon Serverless PostgreSQL
    # ------------------------------------------------------------------
    # Set DATABASE_URL in your .env file to your Neon connection string.
    # Get it from: https://console.neon.tech → your project → Connection Details
    # Format: postgresql+asyncpg://<user>:<password>@<host>.neon.tech/<dbname>?sslmode=require
    #
    # To use AWS RDS in production, just change this URL — zero code change needed.
    BACKEND_DATABASE_URL: str = "postgresql+asyncpg://user:password@host.neon.tech/dbname?sslmode=require"

    # Neon (and most cloud Postgres) requires SSL — set to False only for local plain Postgres
    DB_SSL_REQUIRED: bool = True

    # ------------------------------------------------------------------
    # Security / JWT
    # ------------------------------------------------------------------
    # IMPORTANT: Change this to a long random string in production!
    # Generate one with: python -c "import secrets; print(secrets.token_hex(32))"
    SECRET_KEY: str = "deepguard-dev-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7   # 7 days

    class Config:
        case_sensitive = True
        env_file = ".env"   # optional .env file in the project root
        extra = "ignore"


settings = Settings()

# Ensure storage directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
