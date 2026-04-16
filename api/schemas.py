"""Pydantic models for OptiStock API requests and responses.

All fields are validated for:
- Numeric bounds (no negative costs, no astronomical quantities, no NaN/Inf)
- String length limits (item_id max 128 chars, no null bytes)
- Date format and reasonableness (not year 9999 or negative)
- List size caps (max 10,000 historical data points, max 50 items per request)
- SQL injection pattern rejection
"""

import math
import re
from datetime import datetime, date, timedelta
from typing import Optional

from pydantic import BaseModel, field_validator, model_validator

# ── constants ──────────────────────────────────────────────────────────────
MAX_ITEM_ID_LENGTH = 128
MAX_ITEMS_PER_REQUEST = 50
MAX_HISTORICAL_DATA_POINTS = 10_000
MAX_STRING_FIELD_LENGTH = 1024          # generic string field cap
MIN_REALISTIC_COST = 0.0
MAX_REALISTIC_COST = 1_000_000.0        # $1M per unit is absurd enough
MAX_REALISTIC_QUANTITY = 10_000_000     # 10M units
MAX_REALISTIC_DIMENSION = 10_000.0      # metres — anything bigger is wrong
MAX_REALISTIC_DAYS = 365 * 50           # 50 years in days
MIN_DATE = date(2020, 1, 1)
MAX_DATE = date(2035, 12, 31)
MIN_COST_OF_CAPITAL = -0.5              # allow negative (deflation) but bounded
MAX_COST_OF_CAPITAL = 5.0               # 500 % is insane but we cap
# SQL injection keywords (case-insensitive) that should never appear in item_id
_SQLI_PATTERNS = re.compile(
    r"(\b(union|select|insert|update|delete|drop|alter|create|exec|execute)\b"
    r"|;|--|/\*|\*/|'|\"|\bOR\b\s+\d+\s*=\s*\d+)",
    re.IGNORECASE,
)


# ── helpers ────────────────────────────────────────────────────────────────
def _check_finite(value: float, field_name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{field_name} must be a finite number (no NaN/Inf)")
    return float(value)


def _check_str_no_null_bytes(v: str, field_name: str) -> str:
    if "\x00" in v:
        raise ValueError(f"{field_name} must not contain null bytes")
    return v


def _check_str_no_sqli(v: str, field_name: str) -> str:
    if _SQLI_PATTERNS.search(v):
        raise ValueError(f"{field_name} contains disallowed characters/patterns")
    return v


# ── models ─────────────────────────────────────────────────────────────────
class HistoricalDataPoint(BaseModel):
    date: str
    quantity: float
    available: float = 0.0
    mercury_order_quantity: float = 0.0

    @field_validator("date")
    @classmethod
    def _valid_date(cls, v: str) -> str:
        _check_str_no_null_bytes(v, "date")
        try:
            parsed = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            raise ValueError("date must be a valid ISO 8601 string")
        if parsed.date() < MIN_DATE or parsed.date() > MAX_DATE:
            raise ValueError(f"date {parsed.date()} is outside acceptable range")
        return v

    @field_validator("quantity", mode="before")
    @classmethod
    def _valid_quantity(cls, v):
        v = _check_finite(float(v), "quantity")
        if v < 0:
            raise ValueError("quantity must be >= 0")
        if v > MAX_REALISTIC_QUANTITY:
            raise ValueError(f"quantity exceeds maximum of {MAX_REALISTIC_QUANTITY}")
        return v

    @field_validator("available", "mercury_order_quantity", mode="before")
    @classmethod
    def _valid_nonneg_float(cls, v):
        v = _check_finite(float(v), "field")
        if v < 0:
            raise ValueError("value must be >= 0")
        return v


class ItemInput(BaseModel):
    item_id: str
    current_available: int = 0
    on_order_qty: int = 0
    back_order_qty: int = 0
    order_frequency_days: int = 7
    cost: float
    sale_price: float
    length: float = 1.0
    width: float = 1.0
    height: float = 1.0
    payment_terms_days: int = 30
    sales_terms_days: int = 30
    lead_time_days: float = 14.0
    net_profit_per_unit: float | None = None  # gm after freight, returns, etc.
    historical_lead_times: list[float] = []
    historical_data: list[HistoricalDataPoint] = []

    @field_validator("item_id")
    @classmethod
    def _valid_item_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("item_id must not be empty")
        v = v.strip()
        _check_str_no_null_bytes(v, "item_id")
        _check_str_no_sqli(v, "item_id")
        if len(v) > MAX_ITEM_ID_LENGTH:
            raise ValueError(f"item_id exceeds maximum length of {MAX_ITEM_ID_LENGTH}")
        return v

    @field_validator("current_available", "on_order_qty", "back_order_qty", mode="before")
    @classmethod
    def _valid_int(cls, v):
        if v is None:
            return 0
        if not isinstance(v, int):
            v = int(v)
        if v < 0:
            raise ValueError("integer field must be >= 0")
        if v > MAX_REALISTIC_QUANTITY:
            raise ValueError(f"value exceeds maximum of {MAX_REALISTIC_QUANTITY}")
        return v

    @field_validator("order_frequency_days", "payment_terms_days", "sales_terms_days", mode="before")
    @classmethod
    def _valid_days_int(cls, v):
        if v is None:
            return 0
        if not isinstance(v, int):
            v = int(v)
        if v < 0:
            raise ValueError("days field must be >= 0")
        if v > MAX_REALISTIC_DAYS:
            raise ValueError(f"days field exceeds maximum of {MAX_REALISTIC_DAYS}")
        return v

    @field_validator("cost", mode="before")
    @classmethod
    def _valid_cost(cls, v):
        v = _check_finite(float(v), "cost")
        if v < MIN_REALISTIC_COST:
            raise ValueError("cost must be >= 0")
        if v > MAX_REALISTIC_COST:
            raise ValueError(f"cost exceeds maximum of {MAX_REALISTIC_COST}")
        return v

    @field_validator("sale_price", mode="before")
    @classmethod
    def _valid_sale_price(cls, v):
        v = _check_finite(float(v), "sale_price")
        if v < MIN_REALISTIC_COST:
            raise ValueError("sale_price must be >= 0")
        if v > MAX_REALISTIC_COST:
            raise ValueError(f"sale_price exceeds maximum of {MAX_REALISTIC_COST}")
        return v

    @field_validator("length", "width", "height", mode="before")
    @classmethod
    def _valid_dimension(cls, v):
        v = _check_finite(float(v), "dimension")
        if v < 0:
            raise ValueError("dimension must be >= 0")
        if v > MAX_REALISTIC_DIMENSION:
            raise ValueError(f"dimension exceeds maximum of {MAX_REALISTIC_DIMENSION}")
        return v

    @field_validator("lead_time_days", mode="before")
    @classmethod
    def _valid_lead_time(cls, v):
        v = _check_finite(float(v), "lead_time_days")
        if v < 0:
            raise ValueError("lead_time_days must be >= 0")
        if v > MAX_REALISTIC_DAYS:
            raise ValueError(f"lead_time_days exceeds maximum of {MAX_REALISTIC_DAYS}")
        return v

    @field_validator("historical_lead_times")
    @classmethod
    def _valid_hlt(cls, v: list[float]) -> list[float]:
        return [_check_finite(x, "historical_lead_times") for x in v]

    @field_validator("historical_data")
    @classmethod
    def _valid_hist_data(cls, v: list[HistoricalDataPoint]) -> list[HistoricalDataPoint]:
        if len(v) > MAX_HISTORICAL_DATA_POINTS:
            raise ValueError(
                f"historical_data exceeds maximum of {MAX_HISTORICAL_DATA_POINTS} points"
            )
        return v


class OptimizeRequest(BaseModel):
    items: list[ItemInput]
    cost_of_capital: float = 0.14
    tier: str = "basic"

    @field_validator("cost_of_capital", mode="before")
    @classmethod
    def _valid_coc(cls, v):
        v = _check_finite(float(v), "cost_of_capital")
        if v < MIN_COST_OF_CAPITAL or v > MAX_COST_OF_CAPITAL:
            raise ValueError(
                f"cost_of_capital must be between {MIN_COST_OF_CAPITAL} and {MAX_COST_OF_CAPITAL}"
            )
        return v

    @field_validator("tier")
    @classmethod
    def _valid_tier(cls, v: str) -> str:
        allowed = {"basic", "premium", "elite"}
        if v not in allowed:
            raise ValueError(f"tier must be one of {allowed}, got '{v}'")
        return v

    @model_validator(mode="after")
    def _check_items_limit(self):
        if len(self.items) > MAX_ITEMS_PER_REQUEST:
            raise ValueError(
                f"request contains {len(self.items)} items; maximum is {MAX_ITEMS_PER_REQUEST}"
            )
        return self


class ItemResult(BaseModel):
    item_id: str
    optimal_outp: int
    recommended_order_qty: int
    expected_profit: float
    expected_daily_sales: float
    expected_avg_inventory: float
    cube_usage: float
    profit_per_cube: float
    demand_source: str
    ads: float
    variance: float
    warnings: list[str] = []


class OptimizeResponse(BaseModel):
    items: list[ItemResult]
    compute_time_ms: float


class DemandRequest(BaseModel):
    items: list[ItemInput]

    @model_validator(mode="after")
    def _check_items_limit(self):
        if len(self.items) > MAX_ITEMS_PER_REQUEST:
            raise ValueError(
                f"request contains {len(self.items)} items; maximum is {MAX_ITEMS_PER_REQUEST}"
            )
        return self


class DemandResult(BaseModel):
    item_id: str
    ads: float
    variance: float
    demand_source: str


class DemandResponse(BaseModel):
    items: list[DemandResult]
    compute_time_ms: float


class SimulateRequest(BaseModel):
    """Run simulation directly with provided ads/var (no demand forecasting)."""
    ads: float
    variance: float
    lead_time_days: float = 14.0
    cost: float
    sale_price: float
    length: float = 1.0
    width: float = 1.0
    height: float = 1.0
    payment_terms_days: int = 30
    sales_terms_days: int = 30
    cost_of_capital: float = 0.14
    net_profit_per_unit: float | None = None  # gm after freight, returns, etc.
    current_available: int = 0
    on_order_qty: int = 0
    back_order_qty: int = 0
    order_frequency_days: int = 7
    item_id: str = "direct"

    @field_validator("ads", "variance", mode="before")
    @classmethod
    def _valid_finite_nonneg(cls, v):
        v = _check_finite(float(v), "field")
        if v < 0:
            raise ValueError("value must be >= 0")
        if v > MAX_REALISTIC_QUANTITY:
            raise ValueError(f"value exceeds maximum of {MAX_REALISTIC_QUANTITY}")
        return v

    @field_validator("cost", "sale_price", mode="before")
    @classmethod
    def _valid_price(cls, v):
        v = _check_finite(float(v), "field")
        if v < MIN_REALISTIC_COST:
            raise ValueError("price must be >= 0")
        if v > MAX_REALISTIC_COST:
            raise ValueError(f"price exceeds maximum of {MAX_REALISTIC_COST}")
        return v

    @field_validator("length", "width", "height", mode="before")
    @classmethod
    def _valid_sim_dim(cls, v):
        v = _check_finite(float(v), "dimension")
        if v < 0:
            raise ValueError("dimension must be >= 0")
        if v > MAX_REALISTIC_DIMENSION:
            raise ValueError(f"dimension exceeds maximum of {MAX_REALISTIC_DIMENSION}")
        return v

    @field_validator("lead_time_days", mode="before")
    @classmethod
    def _valid_sim_lt(cls, v):
        v = _check_finite(float(v), "lead_time_days")
        if v < 0:
            raise ValueError("lead_time_days must be >= 0")
        if v > MAX_REALISTIC_DAYS:
            raise ValueError(f"lead_time_days exceeds maximum of {MAX_REALISTIC_DAYS}")
        return v

    @field_validator("cost_of_capital", mode="before")
    @classmethod
    def _valid_sim_coc(cls, v):
        v = _check_finite(float(v), "cost_of_capital")
        if v < MIN_COST_OF_CAPITAL or v > MAX_COST_OF_CAPITAL:
            raise ValueError(f"cost_of_capital must be between {MIN_COST_OF_CAPITAL} and {MAX_COST_OF_CAPITAL}")
        return v

    @field_validator("payment_terms_days", "sales_terms_days", "order_frequency_days", mode="before")
    @classmethod
    def _valid_sim_days(cls, v):
        if v is None:
            return 0
        if not isinstance(v, int):
            v = int(v)
        if v < 0:
            raise ValueError("days field must be >= 0")
        if v > MAX_REALISTIC_DAYS:
            raise ValueError(f"days field exceeds maximum of {MAX_REALISTIC_DAYS}")
        return v

    @field_validator("current_available", "on_order_qty", "back_order_qty", mode="before")
    @classmethod
    def _valid_sim_int(cls, v):
        if v is None:
            return 0
        if not isinstance(v, int):
            v = int(v)
        if v < 0:
            raise ValueError("integer field must be >= 0")
        if v > MAX_REALISTIC_QUANTITY:
            raise ValueError(f"value exceeds maximum of {MAX_REALISTIC_QUANTITY}")
        return v

    @field_validator("item_id")
    @classmethod
    def _valid_sim_item_id(cls, v: str) -> str:
        v = v.strip()
        _check_str_no_null_bytes(v, "item_id")
        _check_str_no_sqli(v, "item_id")
        if len(v) > MAX_ITEM_ID_LENGTH:
            raise ValueError(f"item_id exceeds maximum length of {MAX_ITEM_ID_LENGTH}")
        return v
