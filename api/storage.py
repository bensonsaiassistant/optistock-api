"""Request data storage layer for OptiStock API.

Stores every API request and its results in SQLite for later analysis
and model retraining. Uses aiosqlite for async compatibility with FastAPI.

Storage failures are caught and logged — they never cause a request to fail.
"""

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import aiosqlite

logger = logging.getLogger(__name__)

# Global connection reference — opened once per process, reused across requests
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
    optimal_psl INTEGER,
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
"""


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
        _db_path = os.path.join(os.path.dirname(__file__), "..", "optistock.db")

    if _db is not None:
        return _db

    # Ensure parent directory exists
    parent = os.path.dirname(_db_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    _db = await aiosqlite.connect(_db_path)
    _db.row_factory = aiosqlite.Row
    await _db.executescript(SCHEMA)
    await _db.commit()
    logger.info(f"Database initialized at {_db_path}")
    return _db


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
    """
    try:
        conn = await init_db()
    except Exception as e:
        logger.error(f"Failed to connect to storage DB: {e}")
        return

    try:
        api_key_hash = _hash_api_key(api_key)
        now = datetime.now(timezone.utc).isoformat()

        # Extract top-level request metadata
        items_data = request_data.get("items", [])
        tier = request_data.get("tier")
        cost_of_capital = request_data.get("cost_of_capital")

        # Insert request record
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
                json.dumps(request_data, default=str),
            ),
        )

        # Insert item records with results
        response_items = response_data.get("items", [])
        # Match by index (items are in the same order)
        for i, item_input in enumerate(items_data):
            item_result = response_items[i] if i < len(response_items) else {}

            historical_data_count = 0
            hist = item_input.get("historical_data")
            if hist is not None:
                historical_data_count = len(hist) if isinstance(hist, list) else 0

            await conn.execute(
                """
                INSERT INTO request_items
                    (request_id, item_id, current_available, on_order_qty,
                     back_order_qty, order_frequency_days, cost, sale_price,
                     lead_time_days, historical_data_count,
                     optimal_psl, recommended_order_qty, expected_profit,
                     expected_daily_sales, expected_avg_inventory,
                     cube_usage, profit_per_cube, demand_source, ads,
                     variance, warnings)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    item_input.get("item_id"),
                    item_input.get("current_available", 0),
                    item_input.get("on_order_qty", 0),
                    item_input.get("back_order_qty", 0),
                    item_input.get("order_frequency_days", 7),
                    item_input.get("cost"),
                    item_input.get("sale_price"),
                    item_input.get("lead_time_days"),
                    historical_data_count,
                    item_result.get("optimal_psl"),
                    item_result.get("recommended_order_qty"),
                    item_result.get("expected_profit"),
                    item_result.get("expected_daily_sales"),
                    item_result.get("expected_avg_inventory"),
                    item_result.get("cube_usage"),
                    item_result.get("profit_per_cube"),
                    item_result.get("demand_source"),
                    item_result.get("ads"),
                    item_result.get("variance"),
                    json.dumps(item_result.get("warnings", [])),
                ),
            )

        await conn.commit()

    except Exception as e:
        logger.error(f"Failed to store request {request_id}: {e}")
        try:
            await conn.rollback()
        except Exception:
            pass


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
    """
    try:
        conn = await init_db()
    except Exception as e:
        logger.error(f"Failed to connect to storage DB for query: {e}")
        return []

    try:
        # Build query with optional filters
        query = """
            SELECT ar.*, ri.item_id as ri_item_id, ri.current_available,
                   ri.on_order_qty, ri.back_order_qty, ri.order_frequency_days,
                   ri.cost, ri.sale_price, ri.lead_time_days,
                   ri.historical_data_count, ri.optimal_psl,
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
            params.append(item_id)

        if api_key_hash is not None:
            query += " AND ar.api_key_hash = ?"
            params.append(api_key_hash)

        query += " ORDER BY ar.timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()

        # Merge rows: group items under their request
        requests_map: dict[str, dict] = {}
        for row in rows:
            rid = row["request_id"]
            if rid not in requests_map:
                requests_map[rid] = {
                    "request_id": rid,
                    "api_key_hash": row["api_key_hash"],
                    "endpoint": row["endpoint"],
                    "tier": row["tier"],
                    "cost_of_capital": row["cost_of_capital"],
                    "item_count": row["item_count"],
                    "timestamp": row["timestamp"],
                    "compute_time_ms": row["compute_time_ms"],
                    "raw_request": json.loads(row["raw_request"]) if row["raw_request"] else None,
                    "items": [],
                }
            if row["ri_item_id"] is not None:
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
                    "optimal_psl": row["optimal_psl"],
                    "recommended_order_qty": row["recommended_order_qty"],
                    "expected_profit": row["expected_profit"],
                    "expected_daily_sales": row["expected_daily_sales"],
                    "expected_avg_inventory": row["expected_avg_inventory"],
                    "cube_usage": row["cube_usage"],
                    "profit_per_cube": row["profit_per_cube"],
                    "demand_source": row["demand_source"],
                    "ads": row["ads"],
                    "variance": row["variance"],
                    "warnings": json.loads(row["warnings"]) if row["warnings"] else [],
                })

        return list(requests_map.values())

    except Exception as e:
        logger.error(f"Failed to query requests: {e}")
        return []


async def get_request(request_id: str) -> Optional[dict]:
    """Get a single request by its ID, with all items."""
    try:
        conn = await init_db()
    except Exception as e:
        logger.error(f"Failed to connect to storage DB for get_request: {e}")
        return None

    try:
        cursor = await conn.execute(
            "SELECT * FROM api_requests WHERE request_id = ?", (request_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        result = dict(row)
        result["raw_request"] = json.loads(row["raw_request"]) if row["raw_request"] else None

        # Fetch items
        cursor = await conn.execute(
            "SELECT * FROM request_items WHERE request_id = ?", (request_id,)
        )
        items = []
        async for item_row in cursor:
            item_dict = dict(item_row)
            item_dict["warnings"] = (
                json.loads(item_dict["warnings"]) if item_dict["warnings"] else []
            )
            item_dict.pop("id", None)
            item_dict.pop("request_id", None)
            items.append(item_dict)

        result["items"] = items
        return result

    except Exception as e:
        logger.error(f"Failed to get request {request_id}: {e}")
        return None


async def close_db() -> None:
    """Close the database connection (for shutdown)."""
    global _db
    if _db is not None:
        await _db.close()
        _db = None
        logger.info("Database connection closed")
