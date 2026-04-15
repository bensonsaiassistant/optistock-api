"""Pydantic models for OptiStock API requests and responses."""

from pydantic import BaseModel
from typing import Optional


class HistoricalDataPoint(BaseModel):
    date: str
    quantity: float
    available: float = 0.0
    mercury_order_quantity: float = 0.0


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
    historical_data: list[HistoricalDataPoint] = []


class OptimizeRequest(BaseModel):
    items: list[ItemInput]
    cost_of_capital: float = 0.14
    tier: str = "basic"


class ItemResult(BaseModel):
    item_id: str
    optimal_psl: int
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


class DemandResult(BaseModel):
    item_id: str
    ads: float
    variance: float
    demand_source: str


class DemandResponse(BaseModel):
    items: list[DemandResult]
    compute_time_ms: float
