"""User authentication and API key management routes for OptiStock SaaS.

Endpoints:
- POST /v1/auth/signup
- POST /v1/auth/login
- POST /v1/auth/verify-email
- POST /v1/auth/request-reset
- POST /v1/auth/reset-password
- POST /v1/auth/change-password
- GET  /v1/keys
- POST /v1/keys
- DELETE /v1/keys/{key_id}
- GET  /v1/me
- PUT  /v1/me
- GET  /v1/me/usage
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, field_validator

from . import user_auth
from . import users as user_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["auth", "users", "api-keys"])

# ── Request / Response Schemas ─────────────────────────────────────────────

class SignupRequest(BaseModel):
    email: str
    password: str
    name: str = ""

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        # Basic email format check
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("Invalid email format")
        if len(v) > 254:
            raise ValueError("Email too long")
        return v.strip().lower()

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if len(v) > 128:
            raise ValueError("Password too long")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str


class VerifyEmailRequest(BaseModel):
    token: str


class RequestResetRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class CreateApiKeyRequest(BaseModel):
    name: str = ""


class UpdateUserRequest(BaseModel):
    name: Optional[str] = None


# ── Auth Endpoints ─────────────────────────────────────────────────────────

@router.post("/auth/signup", status_code=status.HTTP_201_CREATED)
async def signup(request: SignupRequest):
    """Register a new user account."""
    try:
        user = await user_manager.create_user(
            email=request.email,
            password=request.password,
            name=request.name,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": str(e), "code": "EMAIL_EXISTS"},
        )

    # Generate email verification token (don't await — fire and forget)
    try:
        await user_manager.generate_verification_token(user["id"])
    except Exception as e:
        logger.warning("Failed to generate verification token: %s", e)

    return {
        "user_id": user["id"],
        "message": "Account created successfully. Please check your email to verify.",
    }


@router.post("/auth/login")
async def login(request: LoginRequest):
    """Authenticate user and return a JWT token."""
    user = await user_manager.verify_user_login(request.email, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Invalid email or password", "code": "INVALID_CREDENTIALS"},
        )

    access_token = user_auth.create_access_token(user["id"], user["email"])

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "is_verified": bool(user["is_verified"]),
            "subscription_tier": user["subscription_tier"],
        },
    }


@router.post("/auth/verify-email")
async def verify_email(request: VerifyEmailRequest):
    """Verify user's email with token."""
    # We need to find the user by token
    import aiosqlite
    from . import storage
    db = await storage.init_db()
    cursor = await db.execute(
        "SELECT user_id FROM email_verifications WHERE token = ? AND is_used = 0",
        (request.token,),
    )
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid or expired verification token", "code": "INVALID_TOKEN"},
        )

    success = await user_manager.verify_email_token(request.token, row["user_id"])
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Token already used or expired", "code": "TOKEN_EXPIRED"},
        )

    return {"message": "Email verified successfully"}


@router.post("/auth/request-reset")
async def request_reset(request: RequestResetRequest):
    """Request a password reset token. Always returns 200 to prevent email enumeration."""
    user = await user_manager.get_user_by_email(request.email)
    if user:
        token = user_manager._generate_reset_token_plaintext()
        await user_manager.store_reset_token(request.email, token)
        # In production, send email with token here
        logger.info("Reset token generated for %s", request.email)

    # Always return success to prevent email enumeration
    return {"message": "If the email exists, a reset link has been sent"}


@router.post("/auth/reset-password")
async def reset_password(request: ResetPasswordRequest):
    """Reset password using a valid token."""
    import aiosqlite
    from . import storage
    import hashlib

    db = await storage.init_db()
    token_hash = hashlib.sha256(request.token.encode("utf-8")).hexdigest()
    cursor = await db.execute(
        "SELECT u.email FROM password_resets pr JOIN users u ON pr.user_id = u.id WHERE pr.token = ? AND pr.is_used = 0",
        (token_hash,),
    )
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid or expired reset token", "code": "INVALID_TOKEN"},
        )

    success = await user_manager.reset_password(request.token, row["email"], request.new_password)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Failed to reset password", "code": "RESET_FAILED"},
        )

    return {"message": "Password reset successfully"}


@router.post("/auth/change-password")
async def change_password(
    request: ChangePasswordRequest,
    current_user: dict = Depends(user_auth.get_current_user),
):
    """Change the current user's password."""
    success = await user_manager.change_password(
        current_user["id"], request.current_password, request.new_password
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Current password is incorrect", "code": "INVALID_PASSWORD"},
        )

    return {"message": "Password changed successfully"}


# ── API Key Endpoints ──────────────────────────────────────────────────────

@router.get("/keys")
async def list_keys(current_user: dict = Depends(user_auth.get_current_user)):
    """List all API keys for the current user."""
    keys = await user_manager.list_api_keys(current_user["id"])
    return {"keys": keys}


@router.post("/keys", status_code=status.HTTP_201_CREATED)
async def create_key(
    request: CreateApiKeyRequest,
    current_user: dict = Depends(user_auth.get_current_user),
):
    """Create a new API key. The plaintext key is returned only ONCE."""
    key_id, plaintext_key = await user_manager.create_api_key(
        current_user["id"], request.name
    )
    return {
        "key_id": key_id,
        "api_key": plaintext_key,
        "message": "Save this key now — it will never be shown again.",
    }


@router.delete("/keys/{key_id}")
async def revoke_key(
    key_id: str,
    current_user: dict = Depends(user_auth.get_current_user),
):
    """Revoke an API key."""
    success = await user_manager.revoke_api_key(current_user["id"], key_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "API key not found", "code": "KEY_NOT_FOUND"},
        )
    return {"message": "API key revoked"}


# ── User Endpoints ─────────────────────────────────────────────────────────

@router.get("/me")
async def get_me(current_user: dict = Depends(user_auth.get_current_user)):
    """Get the current user's profile."""
    return current_user


@router.put("/me")
async def update_me(
    request: UpdateUserRequest,
    current_user: dict = Depends(user_auth.get_current_user),
):
    """Update the current user's profile."""
    updates = {}
    if request.name is not None:
        updates["name"] = request.name

    if updates:
        user = await user_manager.update_user(current_user["id"], **updates)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "User not found", "code": "USER_NOT_FOUND"},
            )
        user.pop("password_hash", None)
        return user

    return current_user


@router.get("/me/usage")
async def get_usage(current_user: dict = Depends(user_auth.get_current_user)):
    """Get the current user's API usage stats."""
    user = await user_manager.get_user_by_id(current_user["id"])
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "User not found", "code": "USER_NOT_FOUND"},
        )

    return {
        "subscription_tier": user["subscription_tier"],
        "api_usage_count": user["api_usage_count"],
        "api_usage_limit": user["api_usage_limit"],
        "remaining": max(0, user["api_usage_limit"] - user["api_usage_count"]),
    }
