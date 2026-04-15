"""FastAPI route handlers for OptiStock API."""

import asyncio
import time
import uuid
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request

from .schemas import (
    OptimizeRequest, OptimizeResponse, ItemResult,
    DemandRequest, DemandResponse, DemandResult,
    SimulateRequest,
)
from .auth import validate_api_key
from . import storage

router = APIRouter()


BATCH_THRESHOLD = 3


def calculate_demand_from_history(
    historical_data: list, item_id: str, tier: str
) -> tuple[float, float, str]:
    """Calculate ads and variance from historical data.

    For basic tier: use calculate_adjusted_demand (outlier-cleaned rolling averages).
    For premium tier with enough data: use ml-regression.
    """
    if not historical_data or len(historical_data) < 7:
        return 0.0, 0.0, "insufficient_data"

    if tier == "premium" and len(historical_data) >= 60:
        from demand.ml_predictor import predict_demand_ml
        return predict_demand_ml(historical_data, item_id)

    # Basic tier or insufficient data
    from demand.adjusted_demand import calculate_demand_from_history as adj_demand

    hist_dict = [
        {
            "date": h.date,
            "quantity": h.quantity,
            "available": h.available,
            "mercury_order_quantity": h.mercury_order_quantity,
        }
        for h in historical_data
    ]
    ads, variance, source = adj_demand(hist_dict)
    return ads, variance, source


def run_psl_optimization(
    ads: float, var: float, lt: float, gm: float, cost: float,
    sale_price: float, length: float, width: float, height: float,
    p_terms: int, s_terms: int, cost_of_capital: float,
) -> tuple:
    """Run the PSL optimization simulation.

    Returns: (optimal_psl, profit, inventory, sales, cube, ppc)
    """
    from simulation.psl_optimizer import calc_opti_psl_3

    result = calc_opti_psl_3(
        ads=ads, var=var, lt=lt, gm=gm, cost=cost,
        avg_sale_price=sale_price, length=length, width=width, height=height,
        p_terms=p_terms, s_terms=s_terms, min_of_1=1,
        cost_of_capital=cost_of_capital,
    )

    return (
        int(result[0]), float(result[1]), float(result[2]),
        float(result[3]), float(result[4]), float(result[5]),
    )


def run_single_item(
    item, tier: str, cost_of_capital: float,
) -> ItemResult:
    """Optimize a single item — demand calc + PSL sweep."""
    warnings: list[str] = []

    ads, var, demand_source = calculate_demand_from_history(
        item.historical_data, item.item_id, tier,
    )

    if ads <= 0:
        warnings.append("No demand data available")
        return ItemResult(
            item_id=item.item_id, optimal_psl=0, recommended_order_qty=0,
            expected_profit=0, expected_daily_sales=0, expected_avg_inventory=0,
            cube_usage=0, profit_per_cube=0, demand_source=demand_source,
            ads=ads, variance=var, warnings=warnings,
        )

    gm = item.sale_price - item.cost
    if gm <= 0:
        warnings.append("Negative or zero gross margin")
        return ItemResult(
            item_id=item.item_id, optimal_psl=0, recommended_order_qty=0,
            expected_profit=0, expected_daily_sales=0, expected_avg_inventory=0,
            cube_usage=0, profit_per_cube=0, demand_source=demand_source,
            ads=ads, variance=var, warnings=warnings,
        )

    optimal_psl, profit, inventory, sales, cube, ppc = run_psl_optimization(
        ads=ads, var=var, lt=item.lead_time_days, gm=gm, cost=item.cost,
        sale_price=item.sale_price, length=item.length, width=item.width,
        height=item.height, p_terms=item.payment_terms_days,
        s_terms=item.sales_terms_days,
        cost_of_capital=cost_of_capital,
    )

    recommended_qty = max(
        0,
        optimal_psl - item.current_available - item.on_order_qty + item.back_order_qty,
    )

    return ItemResult(
        item_id=item.item_id,
        optimal_psl=optimal_psl,
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


def run_batch_psl_optimization(
    items: list, tier: str, cost_of_capital: float,
) -> list[ItemResult]:
    """Optimize multiple items in parallel using get_all_psls (njit parallel)."""
    from simulation.psl_optimizer import get_all_psls

    n = len(items)
    ads_arr = np.zeros(n)
    var_arr = np.zeros(n)
    lt_arr = np.zeros(n)
    gm_arr = np.zeros(n)
    cost_arr = np.zeros(n)
    avg_sale_price_arr = np.zeros(n)
    length_arr = np.zeros(n)
    width_arr = np.zeros(n)
    height_arr = np.zeros(n)
    pterms_arr = np.zeros(n, dtype=np.int64)
    sterms_arr = np.zeros(n, dtype=np.int64)
    min_of_1_arr = np.ones(n, dtype=np.int64)

    # Pre-compute demand for each item
    for i, item in enumerate(items):
        ads, var, _ = calculate_demand_from_history(
            item.historical_data, item.item_id, tier,
        )
        ads_arr[i] = ads
        var_arr[i] = var
        lt_arr[i] = item.lead_time_days
        gm_arr[i] = item.sale_price - item.cost
        cost_arr[i] = item.cost
        avg_sale_price_arr[i] = item.sale_price
        length_arr[i] = item.length
        width_arr[i] = item.width
        height_arr[i] = item.height
        pterms_arr[i] = item.payment_terms_days
        sterms_arr[i] = item.sales_terms_days

    # Run batch optimization (parallel via numba prange)
    results = get_all_psls(
        np.arange(n), ads_arr, var_arr, lt_arr, gm_arr, cost_arr,
        avg_sale_price_arr, length_arr, width_arr, height_arr,
        pterms_arr, sterms_arr, min_of_1_arr, cost_of_capital,
    )

    # Map back to ItemResult objects
    # get_all_psls returns indices in column 0; PSL = index + 1 (min_of_1=1 shift)
    item_results = []
    for i, item in enumerate(items):
        optimal_idx = int(results[i, 0])
        optimal_psl = optimal_idx + 1  # shift because min_of_1=1
        profit = float(results[i, 1])
        inventory = float(results[i, 2])
        sales = float(results[i, 3])
        cube = float(results[i, 4])
        ppc = float(results[i, 5])

        recommended_qty = max(
            0,
            optimal_psl - item.current_available - item.on_order_qty + item.back_order_qty,
        )

        item_results.append(ItemResult(
            item_id=item.item_id,
            optimal_psl=optimal_psl,
            recommended_order_qty=recommended_qty,
            expected_profit=profit,
            expected_daily_sales=sales,
            expected_avg_inventory=inventory,
            cube_usage=cube,
            profit_per_cube=ppc,
            demand_source="adjusted_demand",
            ads=ads_arr[i],
            variance=var_arr[i],
            warnings=[],
        ))

    return item_results


@router.post("/v1/optimize")
async def optimize(
    request: OptimizeRequest,
    api_key: str = Depends(validate_api_key),
):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Route: batch (parallel njit) vs individual
    if len(request.items) >= BATCH_THRESHOLD:
        results = run_batch_psl_optimization(
            request.items, request.tier, request.cost_of_capital,
        )
    else:
        results = [
            run_single_item(item, request.tier, request.cost_of_capital)
            for item in request.items
        ]

    compute_time_ms = (time.time() - start_time) * 1000
    response = OptimizeResponse(items=results, compute_time_ms=compute_time_ms)

    # Store request data (fire-and-forget, never fail the API call)
    storage_task = storage.store_request(
        request_id=request_id,
        api_key=api_key,
        endpoint="/v1/optimize",
        request_data=request.model_dump(),
        response_data=response.model_dump(),
        compute_time_ms=compute_time_ms,
    )
    # Don't await — fire and forget so storage latency doesn't slow the response
    asyncio.ensure_future(storage_task)

    return response


@router.post("/v1/demand")
async def demand_forecast(
    request: DemandRequest,
    api_key: str = Depends(validate_api_key),
):
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

    # Store request data (fire-and-forget)
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
        raise HTTPException(status_code=403, detail="Admin API key required")
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
    result = await storage.get_request(request_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Request not found")
    return result


@router.post("/v1/simulate")
async def simulate(
    request: SimulateRequest,
    api_key: str = Depends(validate_api_key),
):
    """Run simulation directly with provided ads/var (no demand forecasting)."""
    gm = request.sale_price - request.cost
    optimal_psl, profit, inventory, sales, cube, ppc = run_psl_optimization(
        ads=request.ads, var=request.variance, lt=request.lead_time_days, gm=gm,
        cost=request.cost, sale_price=request.sale_price,
        length=request.length, width=request.width, height=request.height,
        p_terms=request.payment_terms_days, s_terms=request.sales_terms_days,
        cost_of_capital=request.cost_of_capital,
    )

    recommended_qty = max(
        0,
        optimal_psl - request.current_available - request.on_order_qty + request.back_order_qty,
    )

    return ItemResult(
        item_id=request.item_id,
        optimal_psl=optimal_psl,
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
