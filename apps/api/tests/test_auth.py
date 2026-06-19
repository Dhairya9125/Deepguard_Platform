import pytest
from httpx import AsyncClient, ASGITransport
from apps.api.main import app
from apps.api.core.config import settings

@pytest.mark.asyncio
async def test_register_and_login(async_client: AsyncClient):
    # 1. Register User
    payload = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "securepassword123"
    }
    resp = await async_client.post(f"{settings.API_V1_STR}/auth/register", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["username"] == "testuser"
    assert data["email"] == "test@example.com"
    
    # 2. Duplicate Registration should fail
    resp2 = await async_client.post(f"{settings.API_V1_STR}/auth/register", json=payload)
    assert resp2.status_code == 409
    
    # 3. Login
    login_payload = {
        "username": "testuser",
        "password": "securepassword123"
    }
    login_resp = await async_client.post(f"{settings.API_V1_STR}/auth/login", json=login_payload)
    assert login_resp.status_code == 200
    token_data = login_resp.json()
    assert "access_token" in token_data
    
    # 4. Fetch /me
    headers = {"Authorization": f"Bearer {token_data['access_token']}"}
    me_resp = await async_client.get(f"{settings.API_V1_STR}/auth/me", headers=headers)
    assert me_resp.status_code == 200
    assert me_resp.json()["username"] == "testuser"

@pytest.mark.asyncio
async def test_api_keys(async_client: AsyncClient):
    # Setup user and get token
    payload = {"username": "keyuser", "email": "key@example.com", "password": "password123"}
    await async_client.post(f"{settings.API_V1_STR}/auth/register", json=payload)
    login_resp = await async_client.post(f"{settings.API_V1_STR}/auth/login", json={"username": "keyuser", "password": "password123"})
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create key
    resp = await async_client.post(f"{settings.API_V1_STR}/users/me/keys", json={"name": "Test Key"}, headers=headers)
    assert resp.status_code == 201
    key_val = resp.json()["key"]
    key_id = resp.json()["id"]
    
    # Use key to access /auth/me
    key_headers = {"X-API-Key": key_val}
    me_resp = await async_client.get(f"{settings.API_V1_STR}/auth/me", headers=key_headers)
    assert me_resp.status_code == 200
    assert me_resp.json()["username"] == "keyuser"
    
    # Revoke key
    del_resp = await async_client.delete(f"{settings.API_V1_STR}/users/me/keys/{key_id}", headers=headers)
    assert del_resp.status_code == 204
    
    # Verify key revoked
    me_resp2 = await async_client.get(f"{settings.API_V1_STR}/auth/me", headers=key_headers)
    assert me_resp2.status_code == 401
