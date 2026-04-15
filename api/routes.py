"""FastAPI route handlers for OptiStock API."""

import time
import numpy as np
from fastapi import APIRouter, Depends, Request

from .schemas import (
    OptimizeRequest, OptimizeResponse, ItemResult,
    DemandRequest, DemandResponse, DemandResult,
)
from .auth import validate_api_key

router = APIRouter()


def calculate_demand_from_history(
    historical_data: list, item_id: str, tier: str
) -> tuple[float, float, str]:
    """Calculate ads and variance from historical data.

    For basic tier: use calculate_adjusted_demand (outlier-cleaned rolling averages).
    For premium tier: would use ml-regression (stubbed for now).
    """
    if not historical_data or len(historical_data) < 7:
        return 0.0, 0.0, "insufficient_data"

    # Use adjusted_demand for proper outlier handling and rolling window selection
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


@router.post("/v1/optimize")
async def optimize(
    request: OptimizeRequest,
    api_key: str = Depends(validate_api_key),
):
    start_time = time.time()
    results = []

    for item in request.items:
        warnings: list[str] = []

        # Calculate demand
        ads, var, demand_source = calculate_demand_from_history(
            item.historical_data, item.item_id, request.tier,
        )

        if ads <= 0:
            warnings.append("No demand data available")
            results.append(ItemResult(
                item_id=item.item_id, optimal_psl=0, recommended_order_qty=0,
                expected_profit=0, expected_daily_sales=0, expected_avg_inventory=0,
                cube_usage=0, profit_per_cube=0, demand_source=demand_source,
                ads=ads, variance=var, warnings=warnings,
            ))
            continue

        # Calculate gross margin
        gm = item.sale_price - item.cost

        if gm <= 0:
            warnings.append("Negative or zero gross margin")
            results.append(ItemResult(
                item_id=item.item_id, optimal_psl=0, recommended_order_qty=0,
                expected_profit=0, expected_daily_sales=0, expected_avg_inventory=0,
                cube_usage=0, profit_per_cube=0, demand_source=demand_source,
                ads=ads, variance=var, warnings=warnings,
            ))
            continue

        # Run PSL optimization
        try:
            optimal_psl, profit, inventory, sales, cube, ppc = run_psl_optimization(
                ads=ads, var=var, lt=item.lead_time_days, gm=gm, cost=item.cost,
                sale_price=item.sale_price, length=item.length, width=item.width,
                height=item.height, p_terms=item.payment_terms_days,
                s_terms=item.sales_terms_days,
                cost_of_capital=request.cost_of_capital,
            )

            # Calculate recommended order quantity
            recommended_qty = max(
                0,
                optimal_psl - item.current_available - item.on_order_qty + item.back_order_qty,
            )

            results.append(ItemResult(
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
            ))
        except Exception as e:
            warnings.append(f"Simulation error: {str(e)}")
            results.append(ItemResult(
                item_id=item.item_id, optimal_psl=0, recommended_order_qty=0,
                expected_profit=0, expected_daily_sales=0, expected_avg_inventory=0,
                cube_usage=0, profit_per_cube=0, demand_source=demand_source,
                ads=ads, variance=var, warnings=warnings,
            ))

    compute_time_ms = (time.time() - start_time) * 1000
    return OptimizeResponse(items=results, compute_time_ms=compute_time_ms)


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
    return DemandResponse(items=results, compute_time_ms=compute_time_ms)


@router.get("/health")
async def health():
    return {"status": "ok", "service": "optistock-api"}
