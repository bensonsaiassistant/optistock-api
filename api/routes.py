"""FastAPI route handlers for OptiStock API.

Security features:
- Global request body size limit (10 MB).
- Global exception handler (never leaks stack traces).
- Input sanitization on all endpoints.
"""

import asyncio
import logging
import time
import uuid

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from .schemas import (
    OptimizeRequest,
    OptimizeResponse,
    ItemResult,
    DemandRequest,
    DemandResponse,
    DemandResult,
    SimulateRequest,
)
from .auth import validate_api_key
from . import storage

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────
MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024  # 10 MB
BATCH_THRESHOLD = 3

router = APIRouter()


# ── Global exception handler ───────────────────────────────────────────────
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler — never leak internal details to the client."""
    if isinstance(exc, HTTPException):
        detail = exc.detail
        if isinstance(detail, dict):
            return JSONResponse(status_code=exc.status_code, content=detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": str(detail), "code": "HTTP_ERROR"},
        )

    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "code": "INTERNAL_ERROR"},
    )


def register_security_handlers(app):
    """Register the global exception handler on the FastAPI app.
    
    Call this in main.py / modal_app.py before including the router.
    """
    app.add_exception_handler(Exception, global_exception_handler)


# ── Request size middleware ────────────────────────────────────────────────
def check_body_size(request: Request) -> None:
    """Raise 413 if the Content-Length exceeds the limit."""
    length = request.headers.get("content-length")
    if length is not None:
        try:
            if int(length) > MAX_REQUEST_BODY_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail={"error": "Request body too large", "code": "PAYLOAD_TOO_LARGE"},
                )
        except ValueError:
            pass  # Malformed header — let Pydantic / body reading fail normally


# ── Demand calculation helpers ─────────────────────────────────────────────
def _historical_to_dict(h) -> dict:
    """Convert HistoricalDataPoint or dict to a standard dict."""
    return {
        "date": h.date if hasattr(h, "date") else h["date"],
        "quantity": h.quantity if hasattr(h, "quantity") else h["quantity"],
        "available": h.available if hasattr(h, "available") else h.get("available", 1.0),
        "mercury_order_quantity": (
            h.mercury_order_quantity
            if hasattr(h, "mercury_order_quantity")
            else h.get("mercury_order_quantity", 0.0)
        ),
    }


def calculate_demand_from_history(
    historical_data: list, item_id: str, tier: str
) -> tuple[float, float, str]:
    """Calculate ads and variance from historical data."""
    if not historical_data or len(historical_data) < 7:
        return 0.0, 0.0, "insufficient_data"

    # Convert to dict list for ML/adjusted_demand consumers
    hist_dict = [_historical_to_dict(h) for h in historical_data]

    if tier in ("premium", "elite") and len(hist_dict) >= 60:
        from demand.ml_predictor import predict_demand_ml
        return predict_demand_ml(hist_dict, item_id)

    from demand.adjusted_demand import calculate_demand_from_history as adj_demand
    ads, variance, source = adj_demand(hist_dict)
    return ads, variance, source


def _calculate_elite_demand_and_lt(item) -> tuple[float, float, float, float, str, str]:
    """Elite tier: predict BOTH demand AND lead time from ML."""
    hist_dict = [
        {
            "date": h.date if hasattr(h, "date") else h["date"],
            "quantity": h.quantity if hasattr(h, "quantity") else h["quantity"],
            "available": h.available if hasattr(h, "available") else h.get("available", 1.0),
            "mercury_order_quantity": (
                h.mercury_order_quantity
                if hasattr(h, "mercury_order_quantity")
                else h.get("mercury_order_quantity", 0.0)
            ),
        }
        for h in item.historical_data
    ]

    historical_lead_times = item.historical_lead_times if item.historical_lead_times else []

    try:
        from demand.ml_predictor import predict_demand_and_lt_ml

        ads, var, lt, lt_var, demand_source, lt_source = predict_demand_and_lt_ml(
            hist_dict, item.item_id, historical_lead_times,
        )
        return ads, var, lt, lt_var, demand_source, lt_source
    except Exception as e:
        logger.warning("Elite ML prediction failed for %s: %s. Falling back.", item.item_id, e)

        if len(hist_dict) >= 60:
            from demand.ml_predictor import predict_demand_ml
            ads, var, demand_source = predict_demand_ml(hist_dict, item.item_id)
        elif len(hist_dict) >= 7:
            ads, var, demand_source = calculate_demand_from_history(hist_dict, item.item_id, "basic")
        else:
            ads, var, demand_source = 0.0, 0.0, "insufficient_data"

        if historical_lead_times:
            from demand.lt_predictor import predict_lead_time_ml
            lt, lt_var, lt_source = predict_lead_time_ml(historical_lead_times, item.item_id)
        else:
            lt = item.lead_time_days
            lt_var = 0.0
            lt_source = "default"

        return ads, var, lt, lt_var, demand_source, lt_source


def run_outp_optimization(
    ads: float, var: float, lt: float, gm: float, cost: float,
    sale_price: float, length: float, width: float, height: float,
    p_terms: int, s_terms: int, cost_of_capital: float,
    lt_variance: float = 0.0,
) -> tuple:
    """Run the OUTP (Order Up To Point) optimization simulation."""
    from simulation.outp_optimizer import calc_opti_outp

    result = calc_opti_outp(
        ads=ads, var=var, lt=lt, gm=gm, cost=cost,
        avg_sale_price=sale_price, length=length, width=width, height=height,
        p_terms=p_terms, s_terms=s_terms, min_of_1=1,
        cost_of_capital=cost_of_capital,
        lt_variance=lt_variance,
    )

    return (
        int(result[0]), float(result[1]), float(result[2]),
        float(result[3]), float(result[4]), float(result[5]),
    )


def run_single_item(item, tier: str, cost_of_capital: float) -> ItemResult:
    """Optimize a single item."""
    warnings: list[str] = []

    if tier == "elite":
        ads, var, lt, lt_var, demand_source, lt_source = _calculate_elite_demand_and_lt(item)
        if ads <= 0:
            warnings.append(f"No demand data available (lead_time_source: {lt_source})")
            return ItemResult(
                item_id=item.item_id, optimal_outp=0, recommended_order_qty=0,
                expected_profit=0, expected_daily_sales=0, expected_avg_inventory=0,
                cube_usage=0, profit_per_cube=0, demand_source=demand_source,
                ads=ads, variance=var, warnings=warnings,
            )
    else:
        ads, var, demand_source = calculate_demand_from_history(
            item.historical_data, item.item_id, tier,
        )
        lt = item.lead_time_days
        lt_var = 0.0

        if ads <= 0:
            warnings.append("No demand data available")
            return ItemResult(
                item_id=item.item_id, optimal_outp=0, recommended_order_qty=0,
                expected_profit=0, expected_daily_sales=0, expected_avg_inventory=0,
                cube_usage=0, profit_per_cube=0, demand_source=demand_source,
                ads=ads, variance=var, warnings=warnings,
            )

    gm = item.sale_price - item.cost
    if gm <= 0:
        warnings.append("Negative or zero gross margin")
        return ItemResult(
            item_id=item.item_id, optimal_outp=0, recommended_order_qty=0,
            expected_profit=0, expected_daily_sales=0, expected_avg_inventory=0,
            cube_usage=0, profit_per_cube=0, demand_source=demand_source,
            ads=ads, variance=var, warnings=warnings,
        )

    optimal_outp, profit, inventory, sales, cube, ppc = run_outp_optimization(
        ads=ads, var=var, lt=lt, gm=gm, cost=item.cost,
        sale_price=item.sale_price, length=item.length, width=item.width,
        height=item.height, p_terms=item.payment_terms_days,
        s_terms=item.sales_terms_days,
        cost_of_capital=cost_of_capital,
        lt_variance=lt_var,
    )

    recommended_qty = max(
        0,
        optimal_outp - item.current_available - item.on_order_qty + item.back_order_qty,
    )

    return ItemResult(
        item_id=item.item_id,
        optimal_outp=optimal_outp,
        recommended_order_qty=recommended_qty,
        expected_profit=profit,
        expected_daily_sales=sales,
        expected_avg_inventory=inventory,
        cube_usage=cube,
        profit_per_cube=ppc,
        demand_source=demand_source,
        ads=ads,
        variance=var,
        warnings=warnings,
    )


def run_batch_outp_optimization(
    items: list, tier: str, cost_of_capital: float,
) -> list[ItemResult]:
    """Optimize multiple items in parallel using get_all_outps (njit parallel)."""
    from simulation.outp_optimizer import get_all_outps

    n = len(items)
    ads_arr = np.zeros(n)
    var_arr = np.zeros(n)
    demand_sources = []
    lt_arr = np.zeros(n)
    lt_var_arr = np.zeros(n)
    gm_arr = np.zeros(n)
    cost_arr = np.zeros(n)
    avg_sale_price_arr = np.zeros(n)
    length_arr = np.zeros(n)
    width_arr = np.zeros(n)
    height_arr = np.zeros(n)
    pterms_arr = np.zeros(n, dtype=np.int64)
    sterms_arr = np.zeros(n, dtype=np.int64)
    min_of_1_arr = np.ones(n, dtype=np.int64)

    for i, item in enumerate(items):
        if tier == "elite":
            ads, var, lt, lt_var, source, lt_source = _calculate_elite_demand_and_lt(item)
            lt_var_arr[i] = lt_var
        else:
            ads, var, source = calculate_demand_from_history(
                item.historical_data, item.item_id, tier,
            )
            lt = item.lead_time_days
            lt_var_arr[i] = 0.0

        ads_arr[i] = ads
        var_arr[i] = var
        demand_sources.append(source)
        lt_arr[i] = lt
        gm_arr[i] = item.sale_price - item.cost
        cost_arr[i] = item.cost
        avg_sale_price_arr[i] = item.sale_price
        length_arr[i] = item.length
        width_arr[i] = item.width
        height_arr[i] = item.height
        pterms_arr[i] = item.payment_terms_days
        sterms_arr[i] = item.sales_terms_days

    results = get_all_outps(
        np.arange(n), ads_arr, var_arr, lt_arr, gm_arr, cost_arr,
        avg_sale_price_arr, length_arr, width_arr, height_arr,
        pterms_arr, sterms_arr, min_of_1_arr, cost_of_capital,
        lt_var_arr=lt_var_arr if tier == "elite" else None,
    )

    item_results = []
    for i, item in enumerate(items):
        optimal_idx = int(results[i, 0])
        optimal_outp = optimal_idx + 1
        profit = float(results[i, 1])
        inventory = float(results[i, 2])
        sales = float(results[i, 3])
        cube = float(results[i, 4])
        ppc = float(results[i, 5])

        recommended_qty = max(
            0,
            optimal_outp - item.current_available - item.on_order_qty + item.back_order_qty,
        )

        item_results.append(ItemResult(
            item_id=item.item_id,
            optimal_outp=optimal_outp,
            recommended_order_qty=recommended_qty,
            expected_profit=profit,
            expected_daily_sales=sales,
            expected_avg_inventory=inventory,
            cube_usage=cube,
            profit_per_cube=ppc,
            demand_source=demand_sources[i],
            ads=ads_arr[i],
            variance=var_arr[i],
            warnings=[],
        ))

    return item_results


# ── Route handlers ─────────────────────────────────────────────────────────
@router.post("/v1/optimize")
async def optimize(
    request: OptimizeRequest,
    api_key: str = Depends(validate_api_key),
):
    check_body_size(request)
    request_id = str(uuid.uuid4())
    start_time = time.time()

    if len(request.items) >= BATCH_THRESHOLD:
        results = run_batch_outp_optimization(
            request.items, request.tier, request.cost_of_capital,
        )
    else:
        results = [
            run_single_item(item, request.tier, request.cost_of_capital)
            for item in request.items
        ]

    compute_time_ms = (time.time() - start_time) * 1000
    response = OptimizeResponse(items=results, compute_time_ms=compute_time_ms)

    storage_task = storage.store_request(
        request_id=request_id,
        api_key=api_key,
        endpoint="/v1/optimize",
        request_data=request.model_dump(),
        response_data=response.model_dump(),
        compute_time_ms=compute_time_ms,
    )
    asyncio.ensure_future(storage_task)

    return response


@router.post("/v1/demand")
async def demand_forecast(
    request: DemandRequest,
    api_key: str = Depends(validate_api_key),
):
    check_body_size(request)
    start_time = time.time()
    results = []

    for item in request.items:
        ads, var, source = calculate_demand_from_history(
            item.historical_data, item.item_id, "basic",
        )
        results.append(DemandResult(
            item_id=item.item_id, ads=ads, variance=var, demand_source=source,
        ))

    compute_time_ms = (time.time() - start_time) * 1000
    response = DemandResponse(items=results, compute_time_ms=compute_time_ms)

    storage_task = storage.store_request(
        request_id=str(uuid.uuid4()),
        api_key=api_key,
        endpoint="/v1/demand",
        request_data=request.model_dump(),
        response_data=response.model_dump(),
        compute_time_ms=compute_time_ms,
    )
    asyncio.ensure_future(storage_task)

    return response


def _validate_admin_key(request: Request) -> str:
    """Check for admin API key (prefix 'admin-'). Raises 403 if not admin."""
    api_key = request.headers.get("X-API-Key", "")
    if not api_key.startswith("admin-"):
        raise HTTPException(
            status_code=403,
            detail={"error": "Admin API key required", "code": "FORBIDDEN"},
        )
    return api_key


@router.get("/v1/requests")
async def list_requests(
    start_date: str,
    end_date: str,
    item_id: str | None = None,
    limit: int = 100,
    _: str = Depends(_validate_admin_key),
):
    """Query stored requests (admin only)."""
    if limit < 1:
        limit = 1
    if limit > 1000:
        limit = 1000

    results = await storage.get_requests(
        start_date=start_date,
        end_date=end_date,
        item_id=item_id,
        limit=limit,
    )
    return {"requests": results, "count": len(results)}


@router.get("/v1/requests/{request_id}")
async def get_single_request(
    request_id: str,
    _: str = Depends(_validate_admin_key),
):
    """Get a single stored request by ID (admin only)."""
    if "\x00" in request_id:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid request ID", "code": "BAD_REQUEST"},
        )
    result = await storage.get_request(request_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "Request not found", "code": "NOT_FOUND"},
        )
    return result


@router.post("/v1/simulate")
async def simulate(
    request: SimulateRequest,
    api_key: str = Depends(validate_api_key),
):
    """Run simulation directly with provided ads/var (no demand forecasting)."""
    check_body_size(request)
    gm = request.sale_price - request.cost
    optimal_outp, profit, inventory, sales, cube, ppc = run_outp_optimization(
        ads=request.ads, var=request.variance, lt=request.lead_time_days, gm=gm,
        cost=request.cost, sale_price=request.sale_price,
        length=request.length, width=request.width, height=request.height,
        p_terms=request.payment_terms_days, s_terms=request.sales_terms_days,
        cost_of_capital=request.cost_of_capital,
    )

    recommended_qty = max(
        0,
        optimal_outp - request.current_available - request.on_order_qty + request.back_order_qty,
    )

    return ItemResult(
        item_id=request.item_id,
        optimal_outp=optimal_outp,
        recommended_order_qty=recommended_qty,
        expected_profit=profit,
        expected_daily_sales=sales,
        expected_avg_inventory=inventory,
        cube_usage=cube,
        profit_per_cube=ppc,
        demand_source="direct_input",
        ads=request.ads,
        variance=request.variance,
        warnings=[],
    )


@router.get("/health")
async def health():
    return {"status": "ok", "service": "optistock-api"}
