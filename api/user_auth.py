"""JWT-based authentication middleware for OptiStock user endpoints.

Provides FastAPI dependencies for Bearer token authentication.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import users as user_manager

# ── Configuration ──────────────────────────────────────────────────────────
JWT_SECRET = os.environ.get("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24

security_scheme = HTTPBearer(auto_error=False)


# ── JWT token utilities ───────────────────────────────────────────────────
def create_access_token(user_id: str, email: str) -> str:
    """Create a JWT access token for a user."""
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    payload = {
        "sub": user_id,
        "email": email,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Decode and validate a JWT token. Raises HTTPException on failure."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Token has expired", "code": "TOKEN_EXPIRED"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Invalid token", "code": "INVALID_TOKEN"},
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
) -> dict:
    """FastAPI dependency: extract and validate the Bearer token, return user dict."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Missing authentication token", "code": "MISSING_TOKEN"},
        )

    payload = decode_access_token(credentials.credentials)
    user_id = payload.get("sub")

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Invalid token payload", "code": "INVALID_TOKEN"},
        )

    user = await user_manager.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "User not found", "code": "USER_NOT_FOUND"},
        )

    if not user.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "Account is deactivated", "code": "ACCOUNT_DISABLED"},
        )

    # Remove sensitive fields before returning
    user.pop("password_hash", None)
    user.pop("reset_token", None)
    user.pop("reset_token_expiry", None)
    user.pop("stripe_customer_id", None)
    user.pop("stripe_subscription_id", None)

    return user


async def get_current_user_from_api_key(
    api_key: str = Depends(lambda request: request.headers.get("X-API-Key")),
) -> dict:
    """FastAPI dependency: authenticate via X-API-Key header."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Missing API key", "code": "MISSING_API_KEY"},
        )

    user = await user_manager.validate_api_key(api_key)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Invalid API key", "code": "UNAUTHORIZED"},
        )

    if not user.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "Account is deactivated", "code": "ACCOUNT_DISABLED"},
        )

    return user
