"""API key validation and rate limiting for OptiStock.

- API keys are SHA-256 hashed before storage/comparison.
- Constant-time comparison via hmac.compare_digest prevents timing attacks.
- Simple in-memory sliding-window rate limiter per API key.
- Supports both env var keys (legacy) and user database keys.
"""

import hashlib
import hmac
import os
import time
from collections import defaultdict
from fastapi import Request, HTTPException

# ── API key store (hashed) ─────────────────────────────────────────────────
# We store SHA-256 hashes, never raw keys.
_VALID_KEY_HASHES: set[str] = set()

_raw_keys = os.environ.get("OPTISTOCK_API_KEYS", "")
for k in _raw_keys.split(","):
    k = k.strip()
    if k:
        _VALID_KEY_HASHES.add(hashlib.sha256(k.encode("utf-8")).hexdigest())


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


# ── Rate limiting ──────────────────────────────────────────────────────────
# Sliding window: track timestamps of requests per key hash.
# Default: 100 requests per 60-second window.
_RATE_WINDOW_SECONDS = 60
_RATE_MAX_REQUESTS = 100
_request_log: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(key_hash: str) -> None:
    """Raise HTTP 429 if the key has exceeded the rate limit."""
    now = time.monotonic()
    window_start = now - _RATE_WINDOW_SECONDS

    # Prune old entries
    timestamps = _request_log[key_hash]
    # Remove entries outside the window
    while timestamps and timestamps[0] < window_start:
        timestamps.pop(0)

    if len(timestamps) >= _RATE_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail={"error": "Rate limit exceeded", "code": "RATE_LIMITED"},
        )

    timestamps.append(now)


def _prune_rate_log() -> None:
    """Remove expired entries from the rate limiter to prevent memory leaks."""
    now = time.monotonic()
    cutoff = now - _RATE_WINDOW_SECONDS * 2
    keys_to_check = list(_request_log.keys())
    for key in keys_to_check:
        timestamps = _request_log[key]
        while timestamps and timestamps[0] < cutoff:
            timestamps.pop(0)
        if not timestamps:
            del _request_log[key]


# Counter for periodic pruning (reset after each full prune)
_rate_prune_counter: int = 0
_RATE_PRUNE_INTERVAL = 100


# ── Key validation ─────────────────────────────────────────────────────────
def validate_api_key_format(api_key: str | None) -> str:
    """Validate that an API key looks reasonable before hashing/comparing.

    Raises 401 on empty / None.
    Returns the (possibly valid) key string for hashing.
    """
    if not api_key or not api_key.strip():
        raise HTTPException(
            status_code=401,
            detail={"error": "Missing API key", "code": "MISSING_API_KEY"},
        )
    # Format: must be printable ASCII, no control chars, no null bytes
    if any(ord(c) < 32 or ord(c) == 127 for c in api_key):
        raise HTTPException(
            status_code=401,
            detail={"error": "Invalid API key format", "code": "INVALID_API_KEY_FORMAT"},
        )
    # Reasonable length: 8–512 chars
    if len(api_key) < 8 or len(api_key) > 512:
        raise HTTPException(
            status_code=401,
            detail={"error": "API key length invalid", "code": "INVALID_API_KEY_LENGTH"},
        )
    return api_key


async def validate_api_key(request: Request) -> str:
    """FastAPI dependency: validate and rate-limit the API key.

    Checks in order:
    1. No key provided → allow as demo/unauthenticated access
    2. Env var keys (legacy deployment mode)
    3. User database keys (SaaS mode)

    Returns the raw API key from the header, or "demo-unauthenticated".
    """
    global _rate_prune_counter

    raw_key = request.headers.get("X-API-Key")

    # No key provided → demo access
    if not raw_key:
        return "demo-unauthenticated"

    # Validate format first
    try:
        validate_api_key_format(raw_key)
    except HTTPException:
        raise  # Format is invalid, reject immediately

    key_hash = _hash_key(raw_key)
    is_valid = False

    # Check env var keys first (legacy mode)
    if _VALID_KEY_HASHES:
        for stored_hash in _VALID_KEY_HASHES:
            if hmac.compare_digest(key_hash, stored_hash):
                is_valid = True
                break

    # Check user database keys (SaaS mode)
    if not is_valid:
        from . import users as user_manager
        user = await user_manager.validate_api_key(raw_key)
        if user:
            is_valid = True

    # Allow if either check passed
    if not is_valid:
        raise HTTPException(
            status_code=401,
            detail={"error": "Invalid API key", "code": "UNAUTHORIZED"},
        )

    _check_rate_limit(key_hash)

    # Periodic prune to prevent memory leak in the rate log
    _rate_prune_counter += 1
    if _rate_prune_counter >= _RATE_PRUNE_INTERVAL:
        _prune_rate_log()
        _rate_prune_counter = 0

    return raw_key
