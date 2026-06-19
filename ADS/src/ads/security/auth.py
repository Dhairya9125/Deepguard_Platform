"""Security layer: JWT authentication, RBAC, audit logging, encryption.

Implements enterprise-grade security patterns for the ADS API.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from jose import JWTError, jwt

from ads.config.settings import settings

logger = logging.getLogger(__name__)


class Role(str, Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API = "api"


def _hash_password(password: str) -> str:
    """Hash password using SHA-256 with salt."""
    salt = uuid.uuid4().hex
    return f"{salt}:{hashlib.sha256((salt + password).encode()).hexdigest()}"


def _verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash."""
    try:
        salt, h = hashed.split(":", 1)
        return h == hashlib.sha256((salt + password).encode()).hexdigest()
    except (ValueError, AttributeError):
        return False


class SecurityManager:
    """Enterprise security manager for authentication and authorization."""

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        self.secret_key = self.config.get("secret_key", settings.security.secret_key)
        self.algorithm = self.config.get("algorithm", settings.security.algorithm)
        self.access_token_expire = self.config.get(
            "access_token_expire_minutes", settings.security.access_token_expire_minutes
        )
        self.refresh_token_expire = self.config.get(
            "refresh_token_expire_days", settings.security.refresh_token_expire_days
        )

    def hash_password(self, password: str) -> str:
        return _hash_password(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return _verify_password(plain_password, hashed_password)

    def create_access_token(
        self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None
    ) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + (
            expires_delta or timedelta(minutes=self.access_token_expire)
        )
        to_encode.update({"exp": expire, "type": "access"})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire)
        to_encode.update({"exp": expire, "type": "refresh"})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def decode_token(self, token: str) -> Dict[str, Any]:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError as e:
            logger.warning(f"Token decode failed: {e}")
            raise

    def validate_token(self, token: str) -> bool:
        try:
            payload = self.decode_token(token)
            return payload.get("type") in ("access", "refresh")
        except JWTError:
            return False

    def check_role(self, token: str, required_role: Role) -> bool:
        payload = self.decode_token(token)
        user_role = payload.get("role", "viewer")
        role_hierarchy = {"admin": 4, "analyst": 3, "api": 2, "viewer": 1}
        return role_hierarchy.get(user_role, 0) >= role_hierarchy.get(required_role.value, 0)


class AuditLogger:
    """Structured audit logging for compliance and security monitoring."""

    def __init__(self) -> None:
        self.enabled = settings.security.enable_audit_log

    def log(
        self,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
    ) -> None:
        """Record an auditable event."""
        if not self.enabled:
            return

        entry = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "user_id": user_id or "anonymous",
            "details": details or {},
            "ip_address": ip_address or "0.0.0.0",
        }

        logger.info(f"AUDIT: {json.dumps(entry)}")


class EncryptionService:
    """Handles encryption/decryption of sensitive data at rest."""

    def __init__(self, key: Optional[str] = None) -> None:
        from cryptography.fernet import Fernet
        import base64

        raw_key = key or settings.security.secret_key
        if len(raw_key) < 32:
            raw_key = hashlib.sha256(raw_key.encode()).digest()
        else:
            raw_key = raw_key[:32].encode() if isinstance(raw_key, str) else raw_key[:32]
        self.cipher = Fernet(base64.urlsafe_b64encode(raw_key))

    def encrypt(self, data: str) -> str:
        return self.cipher.encrypt(data.encode()).decode()

    def decrypt(self, encrypted_data: str) -> str:
        return self.cipher.decrypt(encrypted_data.encode()).decode()


class RateLimiter:
    """Simple in-memory rate limiter using token bucket algorithm."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[datetime]] = {}

    def check(self, key: str) -> bool:
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.window_seconds)

        if key not in self.requests:
            self.requests[key] = []

        self.requests[key] = [t for t in self.requests[key] if t > cutoff]

        if len(self.requests[key]) >= self.max_requests:
            return False

        self.requests[key].append(now)
        return True

    def remaining(self, key: str) -> int:
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.window_seconds)
        if key not in self.requests:
            return self.max_requests
        self.requests[key] = [t for t in self.requests[key] if t > cutoff]
        return max(0, self.max_requests - len(self.requests[key]))


security_manager = SecurityManager()
audit_logger = AuditLogger()
rate_limiter = RateLimiter()
