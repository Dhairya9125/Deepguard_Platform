"""
DeepGuard Platform — Database Connection
Module  : apps.api.db.database
Layer   : Infrastructure

Provides:
  - async SQLAlchemy engine (asyncpg driver)
  - async_session factory for database operations
  - get_session FastAPI dependency (injected into routers)
  - init_db() called at app startup to create all tables

Connection string is read from the DATABASE_URL environment variable.
For local dev, set DATABASE_URL in your .env file to your Neon connection string.
See .env.example for the correct format.
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from apps.api.core.config import settings

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
# asyncpg is the async PostgreSQL driver — works with Neon, AWS RDS, and local Postgres.
# Neon requires SSL; DB_SSL_REQUIRED controls this (True by default).
# Pool kept small to stay within Neon free tier's 100-connection limit.
_connect_args: dict = {"ssl": "require"} if settings.DB_SSL_REQUIRED else {}

engine = create_async_engine(
    settings.BACKEND_DATABASE_URL,
    echo=False,
    pool_pre_ping=True,          # verifies connection is alive before use
    pool_size=3,                 # keep small for Neon free tier (100 connection limit)
    max_overflow=5,
    connect_args=_connect_args,  # SSL for Neon/cloud Postgres
)

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------
async_session = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,      # avoids "DetachedInstanceError" after commit
)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield a database session for the duration of a single HTTP request.
    The session is automatically closed (and rolled back on error) when
    the request handler exits.

    Usage in a router:
        async def my_endpoint(db: AsyncSession = Depends(get_session)):
            ...
    """
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ---------------------------------------------------------------------------
# Table creation (called at startup)
# ---------------------------------------------------------------------------
async def init_db() -> None:
    """
    Create all SQLModel tables that do not yet exist in the database.
    Safe to call every time the app starts — it is a no-op if tables already exist.

    In production, replace this with Alembic migrations for proper schema versioning.
    """
    # Import models so SQLModel is aware of them before creating tables
    import apps.api.db.models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
