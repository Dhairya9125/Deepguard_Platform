import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlmodel import SQLModel

from apps.api.main import app
from apps.api.db.database import get_session
from apps.api.core.config import settings

# --- Setup In-Memory SQLite for tests ---
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestingSessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)

async def override_get_session():
    async with TestingSessionLocal() as session:
        yield session

app.dependency_overrides[get_session] = override_get_session

@pytest.fixture(autouse=True, scope="function")
async def setup_db():
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield
    # Drop tables
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)

@pytest.fixture
async def async_client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

# --- Tests ---

@pytest.mark.asyncio
async def test_read_main(async_client: AsyncClient):
    response = await async_client.get("/")
    assert response.status_code == 200
    assert "Welcome to DeepGuard Platform API" in response.json()["message"]

@pytest.mark.asyncio
async def test_get_job_not_found(async_client: AsyncClient):
    # Without auth
    response = await async_client.get(f"{settings.API_V1_STR}/jobs/non-existent-job-id")
    assert response.status_code == 401

    # Need auth to test 404 properly, handled in test_jobs.py
