"""API key validation and rate limiting for OptiStock.

- API keys are SHA-256 hashed before storage/comparison.
- Constant-time comparison via hmac.compare_digest prevents timing attacks.
- Simple in-memory sliding-window rate limiter per API key.
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

    In dev mode (no keys configured), passes through after rate-limit check.
    Returns the **raw** API key from the header (never stored).
    """
    raw_key = request.headers.get("X-API-Key")

    if not _VALID_KEY_HASHES:
        # Dev mode: no keys configured — still validate format and rate-limit
        if raw_key:
            validate_api_key_format(raw_key)
            key_hash = _hash_key(raw_key)
            _check_rate_limit(key_hash)
        return raw_key or "dev-unauthenticated"

    key = validate_api_key_format(raw_key)
    key_hash = _hash_key(key)

    # Constant-time comparison against all stored hashes
    is_valid = False
    for stored_hash in _VALID_KEY_HASHES:
        if hmac.compare_digest(key_hash, stored_hash):
            is_valid = True
            break

    if not is_valid:
        raise HTTPException(
            status_code=401,
            detail={"error": "Invalid API key", "code": "UNAUTHORIZED"},
        )

    _check_rate_limit(key_hash)
    return key
