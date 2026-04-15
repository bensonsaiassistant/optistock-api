"""API key validation for OptiStock."""

import os
from fastapi import Request, HTTPException

VALID_API_KEYS = [
    k.strip() for k in os.environ.get("OPTISTOCK_API_KEYS", "").split(",") if k.strip()
]


async def validate_api_key(request: Request) -> str:
    api_key = request.headers.get("X-API-Key")
    if not VALID_API_KEYS:
        return api_key  # Allow all if no keys configured (dev mode)
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key
