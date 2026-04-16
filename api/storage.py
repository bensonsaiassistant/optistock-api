"""Request data storage layer for OptiStock API.

Stores every API request and its results in SQLite for later analysis
and model retraining. Uses aiosqlite for async compatibility with FastAPI.

Security:
- All queries use parameterized placeholders (?) — no string interpolation.
- Input validation before storage (NaN/Inf rejection, string length caps, null-byte checks).
- API keys are SHA-256 hashed before storage.
- Storage failures are caught and logged — they never cause a request to fail.
"""

import hashlib
import json
import logging
import math
import os
import re
from datetime import datetime, timezone
from typing import Optional

import aiosqlite

logger = logging.getLogger(__name__)

# ── Security constants ─────────────────────────────────────────────────────
_MAX_STRING_LEN = 4096          # cap any text field stored in the DB
_MAX_RAW_REQUEST_BYTES = 5 * 1024 * 1024  # 5 MB cap on raw_request JSON
_NULL_BYTE_RE = re.compile(r"\x00")

# ── Connection globals ─────────────────────────────────────────────────────
_db: Optional[aiosqlite.Connection] = None
_db_path: Optional[str] = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT UNIQUE NOT NULL,
    api_key_hash TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    tier TEXT,
    cost_of_capital REAL,
    item_count INTEGER,
    timestamp TEXT NOT NULL,
    compute_time_ms REAL,
    raw_request TEXT
);

CREATE TABLE IF NOT EXISTS request_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    current_available INTEGER,
    on_order_qty INTEGER,
    back_order_qty INTEGER,
    order_frequency_days INTEGER,
    cost REAL,
    sale_price REAL,
    lead_time_days REAL,
    historical_data_count INTEGER,
    -- Results
    optimal_outp INTEGER,
    recommended_order_qty INTEGER,
    expected_profit REAL,
    expected_daily_sales REAL,
    expected_avg_inventory REAL,
    cube_usage REAL,
    profit_per_cube REAL,
    demand_source TEXT,
    ads REAL,
    variance REAL,
    warnings TEXT
);

CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON api_requests(timestamp);
CREATE INDEX IF NOT EXISTS idx_items_request_id ON request_items(request_id);
CREATE INDEX IF NOT EXISTS idx_items_item_id ON request_items(item_id);

-- User management tables (OptiStock SaaS)
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


# ── Validation helpers ─────────────────────────────────────────────────────
def _check_null_bytes(value: str, field: str) -> str:
    if _NULL_BYTE_RE.search(value):
        raise ValueError(f"{field} must not contain null bytes")
    return value


def _cap_string(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    value = str(value)[:_MAX_STRING_LEN]
    return _check_null_bytes(value, field)


def _check_finite(value, field: str) -> float | None:
    """Reject NaN/Inf before storing."""
    if value is None:
        return None
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        raise ValueError(f"{field} must be a finite number (got {value})")
    return f


def _cap_raw_request(data: dict) -> str:
    """Serialize to JSON, truncating if it exceeds the max size.

    Truncation can produce invalid JSON, so we validate the result
    and return a safe placeholder if it fails.
    """
    try:
        raw = json.dumps(data, default=str)
    except (TypeError, ValueError, OverflowError) as e:
        # Cannot serialize the request data — return minimal placeholder
        return json.dumps({"error": "serialization_failed", "reason": str(e)})

    if len(raw) > _MAX_RAW_REQUEST_BYTES:
        raw = raw[:_MAX_RAW_REQUEST_BYTES]
        # Try to keep valid JSON; if truncation breaks it, return placeholder
        try:
            json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raw = json.dumps({"error": "truncated", "note": "request exceeded size limit"})
    return raw


# ── Database management ────────────────────────────────────────────────────
def _hash_api_key(api_key: str) -> str:
    """SHA-256 hash of the API key (stored, never the raw key)."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


async def init_db(db_path: Optional[str] = None) -> aiosqlite.Connection:
    """Open the database and ensure schema exists.

    Safe to call multiple times — idempotent.
    Returns the active connection.
    """
    global _db, _db_path

    if db_path is not None:
        _db_path = db_path
    if _db_path is None:
        _db_path = os.environ.get(
            "OPTISTOCK_DB_PATH",
            os.path.join(os.path.dirname(__file__), "..", "optistock.db"),
        )

    if _db is not None:
        return _db

    parent = os.path.dirname(_db_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    _db = await aiosqlite.connect(_db_path)
    _db.row_factory = aiosqlite.Row
    await _db.executescript(SCHEMA)
    await _db.commit()
    logger.info(f"Database initialized at {_db_path}")
    return _db


# ── Write operations ───────────────────────────────────────────────────────
async def store_request(
    request_id: str,
    api_key: str,
    endpoint: str,
    request_data: dict,
    response_data: dict,
    compute_time_ms: float,
) -> None:
    """Store a request and its results.

    Never raises — failures are logged and swallowed so the API keeps serving.
    All data is validated before storage.
    """
    try:
        conn = await init_db()
    except (ValueError, OSError, aiosqlite.Error) as e:
        logger.error("Failed to connect to storage DB: %s", e)
        return
    except Exception as e:
        logger.exception("Unexpected error connecting to storage DB: %s", e)
        return

    try:
        # --- Validate inputs before any SQL ---
        request_id = _cap_string(request_id, "request_id")
        if not request_id:
            raise ValueError("request_id is required")
        _check_null_bytes(request_id, "request_id")

        api_key_hash = _hash_api_key(api_key)

        endpoint = _cap_string(endpoint, "endpoint")
        if not endpoint:
            raise ValueError("endpoint is required")
        _check_null_bytes(endpoint, "endpoint")

        tier = _cap_string(request_data.get("tier"), "tier")
        cost_of_capital = _check_finite(request_data.get("cost_of_capital"), "cost_of_capital")
        compute_time_ms = _check_finite(compute_time_ms, "compute_time_ms")

        now = datetime.now(timezone.utc).isoformat()

        items_data = request_data.get("items", [])
        if not isinstance(items_data, list):
            raise ValueError("items must be a list")

        raw_request_json = _cap_raw_request(request_data)

        # --- All INSERTs use parameterized queries (?) ---
        await conn.execute(
            """
            INSERT INTO api_requests
                (request_id, api_key_hash, endpoint, tier, cost_of_capital,
                 item_count, timestamp, compute_time_ms, raw_request)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                api_key_hash,
                endpoint,
                tier,
                cost_of_capital,
                len(items_data),
                now,
                compute_time_ms,
                raw_request_json,
            ),
        )

        response_items = response_data.get("items", [])

        for i, item_input in enumerate(items_data):
            if not isinstance(item_input, dict):
                continue

            item_result = response_items[i] if i < len(response_items) else {}

            historical_data_count = 0
            hist = item_input.get("historical_data")
            if hist is not None:
                historical_data_count = len(hist) if isinstance(hist, list) else 0

            # Validate each field before parameterized insert
            item_id = _cap_string(item_input.get("item_id"), "item_id")
            current_available = int(item_input.get("current_available", 0))
            on_order_qty = int(item_input.get("on_order_qty", 0))
            back_order_qty = int(item_input.get("back_order_qty", 0))
            order_frequency_days = int(item_input.get("order_frequency_days", 7))
            cost = _check_finite(item_input.get("cost"), "cost")
            sale_price = _check_finite(item_input.get("sale_price"), "sale_price")
            lead_time_days = _check_finite(item_input.get("lead_time_days"), "lead_time_days")

            optimal_outp = item_result.get("optimal_outp")
            if optimal_outp is not None:
                optimal_outp = int(optimal_outp)
            recommended_order_qty = item_result.get("recommended_order_qty")
            if recommended_order_qty is not None:
                recommended_order_qty = int(recommended_order_qty)
            expected_profit = _check_finite(item_result.get("expected_profit"), "expected_profit")
            expected_daily_sales = _check_finite(item_result.get("expected_daily_sales"), "expected_daily_sales")
            expected_avg_inventory = _check_finite(item_result.get("expected_avg_inventory"), "expected_avg_inventory")
            cube_usage = _check_finite(item_result.get("cube_usage"), "cube_usage")
            profit_per_cube = _check_finite(item_result.get("profit_per_cube"), "profit_per_cube")
            demand_source = _cap_string(item_result.get("demand_source"), "demand_source")
            ads = _check_finite(item_result.get("ads"), "ads")
            variance = _check_finite(item_result.get("variance"), "variance")
            warnings_json = json.dumps(item_result.get("warnings", []))

            await conn.execute(
                """
                INSERT INTO request_items
                    (request_id, item_id, current_available, on_order_qty,
                     back_order_qty, order_frequency_days, cost, sale_price,
                     lead_time_days, historical_data_count,
                     optimal_outp, recommended_order_qty, expected_profit,
                     expected_daily_sales, expected_avg_inventory,
                     cube_usage, profit_per_cube, demand_source, ads,
                     variance, warnings)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    item_id,
                    current_available,
                    on_order_qty,
                    back_order_qty,
                    order_frequency_days,
                    cost,
                    sale_price,
                    lead_time_days,
                    historical_data_count,
                    optimal_outp,
                    recommended_order_qty,
                    expected_profit,
                    expected_daily_sales,
                    expected_avg_inventory,
                    cube_usage,
                    profit_per_cube,
                    demand_source,
                    ads,
                    variance,
                    warnings_json,
                ),
            )

        await conn.commit()

    except (ValueError, TypeError, OSError) as e:
        # Expected errors: validation failures, I/O issues
        logger.error("Failed to store request %s: %s", request_id, e)
        try:
            await conn.rollback()
        except (aiosqlite.Error, ValueError, OSError):
            pass
    except Exception as e:
        # Unexpected errors: still log and continue serving
        logger.exception("Unexpected error storing request %s", request_id)
        try:
            await conn.rollback()
        except (aiosqlite.Error, ValueError, OSError):
            pass


# ── Read operations ────────────────────────────────────────────────────────
async def get_requests(
    start_date: str,
    end_date: str,
    item_id: Optional[str] = None,
    api_key_hash: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    """Query stored requests with optional filters.

    Dates are ISO 8601 strings (UTC).
    Returns a list of dicts with request + items merged.
    All query parameters are passed as positional args (parameterized).
    """
    try:
        conn = await init_db()
    except (ValueError, OSError, aiosqlite.Error) as e:
        logger.error("Failed to connect to storage DB for query: %s", e)
        return []
    except Exception as e:
        logger.exception("Unexpected error connecting to storage DB for query: %s", e)
        return []

    try:
        start_date = _cap_string(start_date, "start_date") or ""
        end_date = _cap_string(end_date, "end_date") or ""

        query = """
            SELECT ar.*, ri.item_id as ri_item_id, ri.current_available,
                   ri.on_order_qty, ri.back_order_qty, ri.order_frequency_days,
                   ri.cost, ri.sale_price, ri.lead_time_days,
                   ri.historical_data_count, ri.optimal_outp,
                   ri.recommended_order_qty, ri.expected_profit,
                   ri.expected_daily_sales, ri.expected_avg_inventory,
                   ri.cube_usage, ri.profit_per_cube, ri.demand_source,
                   ri.ads, ri.variance, ri.warnings
            FROM api_requests ar
            LEFT JOIN request_items ri ON ar.request_id = ri.request_id
            WHERE ar.timestamp >= ? AND ar.timestamp <= ?
        """
        params: list = [start_date, end_date]

        if item_id is not None:
            query += " AND ri.item_id = ?"
            params.append(_cap_string(item_id, "item_id") or "")

        if api_key_hash is not None:
            query += " AND ar.api_key_hash = ?"
            params.append(_cap_string(api_key_hash, "api_key_hash") or "")

        limit = min(max(1, limit), 1000)  # clamp 1–1000
        query += " ORDER BY ar.timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()

        requests_map: dict[str, dict] = {}
        for row in rows:
            rid = row["request_id"]
            if rid not in requests_map:
                raw_req = row["raw_request"]
                requests_map[rid] = {
                    "request_id": rid,
                    "api_key_hash": row["api_key_hash"],
                    "endpoint": row["endpoint"],
                    "tier": row["tier"],
                    "cost_of_capital": row["cost_of_capital"],
                    "item_count": row["item_count"],
                    "timestamp": row["timestamp"],
                    "compute_time_ms": row["compute_time_ms"],
                    "raw_request": json.loads(raw_req) if raw_req else None,
                    "items": [],
                }
            if row["ri_item_id"] is not None:
                warnings_raw = row["warnings"]
                requests_map[rid]["items"].append({
                    "item_id": row["ri_item_id"],
                    "current_available": row["current_available"],
                    "on_order_qty": row["on_order_qty"],
                    "back_order_qty": row["back_order_qty"],
                    "order_frequency_days": row["order_frequency_days"],
                    "cost": row["cost"],
                    "sale_price": row["sale_price"],
                    "lead_time_days": row["lead_time_days"],
                    "historical_data_count": row["historical_data_count"],
                    "optimal_outp": row["optimal_outp"],
                    "recommended_order_qty": row["recommended_order_qty"],
                    "expected_profit": row["expected_profit"],
                    "expected_daily_sales": row["expected_daily_sales"],
                    "expected_avg_inventory": row["expected_avg_inventory"],
                    "cube_usage": row["cube_usage"],
                    "profit_per_cube": row["profit_per_cube"],
                    "demand_source": row["demand_source"],
                    "ads": row["ads"],
                    "variance": row["variance"],
                    "warnings": json.loads(warnings_raw) if warnings_raw else [],
                })

        return list(requests_map.values())

    except (ValueError, TypeError, OSError, aiosqlite.Error) as e:
        # Expected errors: validation, I/O, database
        logger.error("Failed to query requests: %s", e)
        return []
    except Exception as e:
        # Unexpected errors: still log and continue serving
        logger.exception("Unexpected error querying requests: %s", e)
        return []


async def get_request(request_id: str) -> Optional[dict]:
    """Get a single request by its ID, with all items.

    Uses parameterized query.
    """
    try:
        conn = await init_db()
    except (ValueError, OSError, aiosqlite.Error) as e:
        logger.error("Failed to connect to storage DB for get_request: %s", e)
        return None
    except Exception as e:
        logger.exception("Unexpected error connecting to storage DB for get_request: %s", e)
        return None

    try:
        request_id = _cap_string(request_id, "request_id")
        if not request_id:
            return None

        cursor = await conn.execute(
            "SELECT * FROM api_requests WHERE request_id = ?", (request_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        result = dict(row)
        raw_req = row["raw_request"]
        result["raw_request"] = json.loads(raw_req) if raw_req else None

        cursor = await conn.execute(
            "SELECT * FROM request_items WHERE request_id = ?", (request_id,)
        )
        items = []
        async for item_row in cursor:
            item_dict = dict(item_row)
            warnings_raw = item_dict.get("warnings")
            item_dict["warnings"] = (
                json.loads(warnings_raw) if warnings_raw else []
            )
            item_dict.pop("id", None)
            item_dict.pop("request_id", None)
            items.append(item_dict)

        result["items"] = items
        return result

    except (ValueError, TypeError, OSError, aiosqlite.Error) as e:
        # Expected errors: validation, I/O, database
        logger.error("Failed to get request %s: %s", request_id, e)
        return None
    except Exception as e:
        # Unexpected errors: still log and continue serving
        logger.exception("Unexpected error getting request %s", request_id)
        return None


async def close_db() -> None:
    """Close the database connection (for shutdown)."""
    global _db
    if _db is not None:
        await _db.close()
        _db = None
        logger.info("Database connection closed")
