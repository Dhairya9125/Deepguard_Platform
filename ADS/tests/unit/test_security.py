"""Unit tests for security module."""

import pytest
from ads.security.auth import SecurityManager, AuditLogger, RateLimiter


class TestSecurityManager:
    def test_password_hashing(self):
        sm = SecurityManager()
        hashed = sm.hash_password("test_password")
        assert hashed != "test_password"
        assert sm.verify_password("test_password", hashed)

    def test_token_creation_and_validation(self):
        sm = SecurityManager()
        data = {"sub": "test_user", "role": "admin"}
        token = sm.create_access_token(data)
        assert sm.validate_token(token)

    def test_invalid_token(self):
        sm = SecurityManager()
        assert not sm.validate_token("invalid_token")

    def test_role_check(self):
        sm = SecurityManager()
        admin_token = sm.create_access_token({"sub": "admin", "role": "admin"})
        viewer_token = sm.create_access_token({"sub": "viewer", "role": "viewer"})

        from ads.security.auth import Role
        assert sm.check_role(admin_token, Role.ADMIN)
        assert sm.check_role(admin_token, Role.VIEWER)
        assert not sm.check_role(viewer_token, Role.ADMIN)

    def test_token_decode(self):
        sm = SecurityManager()
        data = {"sub": "user1", "role": "analyst"}
        token = sm.create_access_token(data)
        decoded = sm.decode_token(token)
        assert decoded["sub"] == "user1"
        assert decoded["role"] == "analyst"
        assert decoded["type"] == "access"


class TestRateLimiter:
    def test_rate_limiting(self):
        rl = RateLimiter(max_requests=3, window_seconds=60)
        assert rl.check("test_key")
        assert rl.check("test_key")
        assert rl.check("test_key")
        assert not rl.check("test_key")

    def test_remaining(self):
        rl = RateLimiter(max_requests=5, window_seconds=60)
        assert rl.remaining("key") == 5
        rl.check("key")
        assert rl.remaining("key") == 4

    def test_different_keys(self):
        rl = RateLimiter(max_requests=2, window_seconds=60)
        assert rl.check("key1")
        assert rl.check("key2")
        assert rl.check("key2")
        assert rl.check("key1")
        assert not rl.check("key2")
