"""User management and API key operations for OptiStock SaaS.

All database operations are async (aiosqlite).
- bcrypt for password hashing
- SHA-256 with constant-time comparison for API keys
- JWT-compatible tokens for email verification and password resets
"""

import hashlib
import hmac
import logging
import os
import secrets
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

import aiosqlite
import bcrypt

from . import storage  # reuses the shared DB connection pattern

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
_TOKEN_EXPIRY_HOURS = 24
_VERIFICATION_TOKEN_EXPIRY_HOURS = 48
_API_KEY_PREFIX = "osk"

# Subscription tier limits (monthly API request caps)
_TIER_LIMITS = {
    "free": 100,
    "starter": 1_000,
    "professional": 10_000,
    "enterprise": 100_000,
}

# ── Schema DDL (appended to storage.py SCHEMA) ────────────────────────────
USER_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    is_verified INTEGER NOT NULL DEFAULT 0,
    is_active INTEGER NOT NULL DEFAULT 1,
    subscription_tier TEXT NOT NULL DEFAULT 'free',
    stripe_customer_id TEXT,
    stripe_subscription_id TEXT,
    api_usage_count INTEGER NOT NULL DEFAULT 0,
    api_usage_limit INTEGER NOT NULL DEFAULT 100,
    reset_token TEXT,
    reset_token_expiry TEXT
);

CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    name TEXT NOT NULL DEFAULT '',
    key_hash TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS email_verifications (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    token TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    is_used INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS password_resets (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    token TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    is_used INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_email_verifications_token ON email_verifications(token);
CREATE INDEX IF NOT EXISTS idx_password_resets_token ON password_resets(token);
"""


# ── Database connection (shared with storage.py) ──────────────────────────
async def _get_db() -> aiosqlite.Connection:
    """Get the shared database connection, initializing if needed."""
    return await storage.init_db()


# ── Password utilities ────────────────────────────────────────────────────
def hash_password(plaintext: str) -> str:
    """Hash a plaintext password with bcrypt."""
    return bcrypt.hashpw(plaintext.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plaintext: str, password_hash: str) -> bool:
    """Verify a plaintext password against a bcrypt hash (constant-time)."""
    try:
        return bcrypt.checkpw(plaintext.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def _generate_secure_token(length: int = 32) -> str:
    """Generate a URL-safe secure random token."""
    return secrets.token_urlsafe(length)


# ── Token generation ──────────────────────────────────────────────────────
def _generate_reset_token_plaintext() -> str:
    """Generate a password reset token (plaintext only, no DB side effects).

    The caller should store the hashed token via the route handler.
    Returns the plaintext token to be sent to the user.
    """
    return _generate_secure_token(48)


async def store_reset_token(email: str, token: str) -> bool:
    """Store a password reset token in the database.

    Call this after _generate_reset_token_plaintext().
    Returns True if the user was found and the token was stored.
    """
    import hashlib as _hashlib

    user = await get_user_by_email(email)
    if not user:
        return False

    token_hash = _hashlib.sha256(token.encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc)
    expires = now + timedelta(hours=_TOKEN_EXPIRY_HOURS)

    db = await _get_db()
    reset_id = str(uuid.uuid4())
    await db.execute(
        """
        INSERT INTO password_resets (id, user_id, token, created_at, expires_at, is_used)
        VALUES (?, ?, ?, ?, ?, 0)
        """,
        (reset_id, user["id"], token_hash, now.isoformat(), expires.isoformat()),
    )
    await db.execute(
        "UPDATE users SET reset_token = ?, reset_token_expiry = ? WHERE id = ?",
        (token_hash, expires.isoformat(), user["id"]),
    )
    await db.commit()
    return True


async def verify_reset_token(token: str, email: str) -> bool:
    """Verify a password reset token. Returns True if valid and not expired."""
    import hashlib as _hashlib

    token_hash = _hashlib.sha256(token.encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc).isoformat()

    db = await _get_db()
    cursor = await db.execute(
        """
        SELECT pr.id, pr.expires_at, pr.is_used
        FROM password_resets pr
        JOIN users u ON pr.user_id = u.id
        WHERE pr.token = ? AND u.email = ?
        """,
        (token_hash, email),
    )
    row = await cursor.fetchone()
    if not row:
        return False

    if row["expires_at"] < now or row["is_used"]:
        return False

    # Mark as used
    await db.execute(
        "UPDATE password_resets SET is_used = 1 WHERE id = ?", (row["id"],)
    )
    await db.execute(
        "UPDATE users SET reset_token = NULL, reset_token_expiry = NULL WHERE id = ?",
        (row["id"],),
    )
    await db.commit()
    return True


async def generate_verification_token(user_id: str) -> str:
    """Generate an email verification token for a user."""
    token = _generate_secure_token(48)
    now = datetime.now(timezone.utc)
    expires = now + timedelta(hours=_VERIFICATION_TOKEN_EXPIRY_HOURS)

    db = await _get_db()
    await db.execute(
        """
        INSERT INTO email_verifications (id, user_id, token, created_at, expires_at, is_used)
        VALUES (?, ?, ?, ?, ?, 0)
        """,
        (str(uuid.uuid4()), user_id, token, now.isoformat(), expires.isoformat()),
    )
    await db.commit()
    return token


async def verify_email_token(token: str, user_id: str) -> bool:
    """Verify an email token. Returns True if valid and not expired."""
    now = datetime.now(timezone.utc).isoformat()

    db = await _get_db()
    cursor = await db.execute(
        """
        SELECT id, expires_at, is_used
        FROM email_verifications
        WHERE token = ? AND user_id = ?
        """,
        (token, user_id),
    )
    row = await cursor.fetchone()
    if not row:
        return False

    if row["expires_at"] < now or row["is_used"]:
        return False

    await db.execute(
        "UPDATE email_verifications SET is_used = 1 WHERE id = ?", (row["id"],)
    )
    await db.execute(
        "UPDATE users SET is_verified = 1 WHERE id = ?", (user_id,)
    )
    await db.commit()
    return True


# ── API Key management ────────────────────────────────────────────────────
def _hash_api_key(plaintext: str) -> str:
    """SHA-256 hash of an API key."""
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def generate_api_key() -> str:
    """Generate a new API key with the osk- prefix."""
    unique = secrets.token_hex(16)
    chunks = [unique[i:i+5] for i in range(0, len(unique), 5)]
    return f"{_API_KEY_PREFIX}-{'-'.join(chunks[:3])}"


def verify_api_key_constant_time(plaintext: str, stored_hash: str) -> bool:
    """Constant-time comparison of a plaintext key against a stored hash."""
    key_hash = _hash_api_key(plaintext)
    return hmac.compare_digest(key_hash, stored_hash)


async def create_api_key(user_id: str, name: str = "") -> tuple[str, str]:
    """Create a new API key for a user.

    Returns (key_id, plaintext_key) — the plaintext is only returned here.
    """
    plaintext_key = generate_api_key()
    key_hash = _hash_api_key(plaintext_key)
    key_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    db = await _get_db()
    await db.execute(
        """
        INSERT INTO api_keys (id, user_id, name, key_hash, created_at, is_active)
        VALUES (?, ?, ?, ?, ?, 1)
        """,
        (key_id, user_id, name, key_hash, now),
    )
    await db.commit()
    return key_id, plaintext_key


async def list_api_keys(user_id: str) -> list[dict]:
    """List all active API keys for a user (without the hash)."""
    db = await _get_db()
    cursor = await db.execute(
        "SELECT id, user_id, name, created_at, is_active FROM api_keys WHERE user_id = ?",
        (user_id,),
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def revoke_api_key(user_id: str, key_id: str) -> bool:
    """Revoke (deactivate) an API key. Returns True if the key existed and belonged to the user."""
    db = await _get_db()
    cursor = await db.execute(
        "SELECT id FROM api_keys WHERE id = ? AND user_id = ?",
        (key_id, user_id),
    )
    row = await cursor.fetchone()
    if not row:
        return False

    await db.execute(
        "UPDATE api_keys SET is_active = 0 WHERE id = ?", (key_id,)
    )
    await db.commit()
    return True


async def validate_api_key(key: str) -> Optional[dict]:
    """Validate an API key and return the associated user dict, or None."""
    if not key or not key.strip():
        return None

    key_hash = _hash_api_key(key)

    db = await _get_db()
    cursor = await db.execute(
        """
        SELECT u.id, u.email, u.name, u.is_verified, u.is_active,
               u.subscription_tier, u.api_usage_count, u.api_usage_limit
        FROM api_keys ak
        JOIN users u ON ak.user_id = u.id
        WHERE ak.key_hash = ? AND ak.is_active = 1 AND u.is_active = 1
        """,
        (key_hash,),
    )
    row = await cursor.fetchone()
    if not row:
        return None

    return dict(row)


# ── User CRUD ─────────────────────────────────────────────────────────────
async def create_user(email: str, password: str, name: str = "") -> dict:
    """Create a new user. Raises ValueError on duplicate email."""
    existing = await get_user_by_email(email)
    if existing:
        raise ValueError(f"User with email {email} already exists")

    user_id = str(uuid.uuid4())
    pw_hash = hash_password(password)
    now = datetime.now(timezone.utc).isoformat()
    tier = "free"
    limit = _TIER_LIMITS[tier]

    db = await _get_db()
    await db.execute(
        """
        INSERT INTO users (id, email, password_hash, name, created_at,
                           is_verified, is_active, subscription_tier,
                           api_usage_count, api_usage_limit)
        VALUES (?, ?, ?, ?, ?, 0, 1, ?, 0, ?)
        """,
        (user_id, email, pw_hash, name, now, tier, limit),
    )
    await db.commit()

    return await get_user_by_id(user_id)


async def get_user_by_email(email: str) -> Optional[dict]:
    """Get a user by email (includes password_hash — don't expose to client)."""
    db = await _get_db()
    cursor = await db.execute(
        "SELECT * FROM users WHERE email = ?", (email,)
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


async def get_user_by_id(user_id: str) -> Optional[dict]:
    """Get a user by ID."""
    db = await _get_db()
    cursor = await db.execute(
        "SELECT * FROM users WHERE id = ?", (user_id,)
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


async def update_user(user_id: str, **fields) -> Optional[dict]:
    """Update user fields. Only allows safe fields."""
    allowed = {
        "name", "subscription_tier", "stripe_customer_id",
        "stripe_subscription_id", "api_usage_limit", "is_active",
    }
    safe_fields = {k: v for k, v in fields.items() if k in allowed}
    if not safe_fields:
        return await get_user_by_id(user_id)

    set_clause = ", ".join(f"{k} = ?" for k in safe_fields)
    values = list(safe_fields.values()) + [user_id]

    db = await _get_db()
    await db.execute(f"UPDATE users SET {set_clause} WHERE id = ?", values)
    await db.commit()
    return await get_user_by_id(user_id)


async def delete_user(user_id: str) -> bool:
    """Delete a user and all associated data."""
    db = await _get_db()
    await db.execute("DELETE FROM users WHERE id = ?", (user_id,))
    await db.execute("DELETE FROM api_keys WHERE user_id = ?", (user_id,))
    await db.execute("DELETE FROM email_verifications WHERE user_id = ?", (user_id,))
    await db.execute("DELETE FROM password_resets WHERE user_id = ?", (user_id,))
    await db.commit()
    return True


async def list_users(limit: int = 50, offset: int = 0) -> list[dict]:
    """List users (admin use — excludes password_hash)."""
    db = await _get_db()
    limit = min(max(1, limit), 100)
    offset = max(0, offset)

    cursor = await db.execute(
        """
        SELECT id, email, name, created_at, is_verified, is_active,
               subscription_tier, api_usage_count, api_usage_limit
        FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?
        """,
        (limit, offset),
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def verify_user_login(email: str, password: str) -> Optional[dict]:
    """Verify login credentials. Returns user dict (without password_hash) or None."""
    user = await get_user_by_email(email)
    if not user:
        return None
    if not user["is_active"]:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    # Strip password hash from returned dict
    user.pop("password_hash", None)
    return user


async def change_password(user_id: str, current_password: str, new_password: str) -> bool:
    """Change a user's password. Returns True on success."""
    user = await get_user_by_id(user_id)
    if not user:
        return False
    if not verify_password(current_password, user["password_hash"]):
        return False
    new_hash = hash_password(new_password)
    db = await _get_db()
    await db.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user_id)
    )
    await db.commit()
    return True


async def reset_password(token: str, email: str, new_password: str) -> bool:
    """Reset a user's password using a valid reset token."""
    valid = await verify_reset_token(token, email)
    if not valid:
        return False

    new_hash = hash_password(new_password)
    db = await _get_db()
    await db.execute(
        "UPDATE users SET password_hash = ? WHERE email = ?", (new_hash, email)
    )
    await db.commit()
    return True


async def increment_api_usage(user_id: str) -> None:
    """Increment the user's monthly API usage count."""
    db = await _get_db()
    await db.execute(
        "UPDATE users SET api_usage_count = api_usage_count + 1 WHERE id = ?",
        (user_id,),
    )
    await db.commit()


async def check_api_usage(user_id: str) -> bool:
    """Check if user has remaining API quota. Returns True if within limits."""
    user = await get_user_by_id(user_id)
    if not user:
        return False
    return user["api_usage_count"] < user["api_usage_limit"]
