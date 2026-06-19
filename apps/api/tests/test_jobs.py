import pytest
from httpx import AsyncClient, ASGITransport
from apps.api.main import app
from apps.api.core.config import settings

@pytest.mark.asyncio
async def test_jobs_empty(async_client: AsyncClient):
    # Register and login
    await async_client.post(f"{settings.API_V1_STR}/auth/register", json={"username": "jobuser", "email": "job@example.com", "password": "123"})
    login_resp = await async_client.post(f"{settings.API_V1_STR}/auth/login", json={"username": "jobuser", "password": "123"})
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Fetch jobs
    resp = await async_client.get(f"{settings.API_V1_STR}/jobs", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert len(data["items"]) == 0

@pytest.mark.asyncio
async def test_jobs_stats_empty(async_client: AsyncClient):
    login_resp = await async_client.post(f"{settings.API_V1_STR}/auth/login", json={"username": "jobuser", "password": "123"})
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    resp = await async_client.get(f"{settings.API_V1_STR}/stats", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_jobs"] == 0
    assert data["total_fake"] == 0
    assert data["total_real"] == 0
