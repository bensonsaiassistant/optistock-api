"""Day-level inventory simulation engine.

Optimized for speed: uses scalar accumulation instead of per-day arrays.
"""

import numpy as np
import math as m
from numba import njit, int64, float64

from .demand_dist import calc_nb_array_ln, numba_choice


@njit
def day_sim_3(
    psl: int64,
    ads: float64,
    var: float64,
    lt: int64,
    p_terms: int64,
    s_terms: int64,
    days: int64,
    demand_sample: np.ndarray,
    lt_sample: np.ndarray,
    cost_of_capital: float64 = 0.14,
) -> np.ndarray:
    """Run a day-by-day inventory simulation.

    Returns np.array([avg_sales, avg_inventory, avg_ar, avg_ap, avg_deval])
    """
    deval_days = 548
    start_inv = max(psl, int64(1))
    max_order_size = max(int64(round(ads * 2, 0)), int64(1))

    use_pre_sampled = (ads / var < 0.95) and (var - ads > 0.1) if var > 0 else False

    # --- Scalar state variables ---
    inv = int64(start_inv)
    oo = int64(0)  # on_order

    # Ring buffers for AR/AP tracking
    ar_ring = np.zeros(366, dtype=np.int64)
    ap_ring = np.zeros(366, dtype=np.int64)
    inv_ring = np.zeros(550, dtype=np.int64)

    # Running totals for post-warmup averaging
    sum_sales = float64(0.0)
    sum_inv = float64(0.0)
    sum_ar = float64(0.0)
    sum_ap = float64(0.0)
    sum_deval = float64(0.0)
    count = int64(0)

    # Current AR/AP balances
    ar_balance = int64(0)
    ap_balance = int64(0)

    # Warmup period: skip first N days
    warmup_end = int64(start_inv) if ads > 0 else int64(0)
    warmup_end = min(warmup_end, days)

    for i in range(days):
        # Demand
        if use_pre_sampled:
            demand = int64(demand_sample[i % demand_sample.size])
        else:
            demand = int64(np.random.poisson(ads)) if ads > 0 else int64(0)

        # Sales (can't sell more than we have)
        sold = min(demand, inv)

        # Inventory update
        inv = max(inv - sold, int64(0))

        # Order placement
        order_qty = min(max(psl - inv - oo, int64(0)), max_order_size)
        oo += order_qty

        # Receipts: orders arrive after 'lt' days
        if i >= lt:
            # Receive what was ordered lt days ago (simplified: receive up to on_order)
            recv = min(oo, int64(1000000))
            inv += recv
            oo -= recv

        # AR: customers pay after s_terms days
        ar_balance += sold
        ar_ring[i % 366] = sold
        if i >= s_terms:
            ar_balance -= ar_ring[(i - s_terms) % 366]

        # AP: we pay after p_terms days
        ap_ring[i % 366] = order_qty
        if i >= p_terms:
            ap_balance -= ap_ring[(i - p_terms) % 366]

        # Devaluation: track inventory from deval_days ago
        inv_ring[i % 550] = inv
        if i >= deval_days:
            sum_deval += max(int64(0), inv_ring[(i - deval_days) % 550] - sold)

        # Accumulate stats (post-warmup only)
        if i >= warmup_end:
            sum_sales += sold
            sum_inv += inv
            sum_ar += ar_balance
            sum_ap += ap_balance
            count += 1

    if count == 0:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    return np.array([
        sum_sales / count,
        sum_inv / count,
        sum_ar / count,
        sum_ap / count,
        sum_deval / count,
    ])


@njit
def calc_single_psl(
    psl: int64,
    ads: float64,
    var: float64,
    lt: float64,
    gm: float64,
    cost: float64,
    avg_sale_price: float64,
    length: float64,
    width: float64,
    height: float64,
    p_terms: int64,
    s_terms: int64,
    min_of_1: int64,
    cost_of_capital: float64 = 0.14,
    lt_variance: float64 = 0.0,
    sim_days: int64 = 200,
) -> np.ndarray:
    """Calculate profit metrics for a single PSL value."""
    invst_charge = cost_of_capital / 365.0
    days = sim_days
    deval_days = 548

    if ads <= 0 or var <= 0 or lt <= 0 or psl <= 0:
        return np.array([float64(psl), 0.0, 0.0, 0.0, 0.0, 0.0])

    # Generate demand samples
    if (ads / var < 0.95) and (var - ads > 0.1):
        demand_array = calc_nb_array_ln(ads, var)
        demand_size = np.arange(demand_array.size)
        n_to_sample = min(days, demand_array.size)
        demand_sample = numba_choice(demand_size, demand_array, n_to_sample)
    else:
        demand_sample = np.zeros(1, dtype=np.int64)

    # Generate lead time samples
    lt_int = int64(round(lt))
    lt_sample = np.full(days, lt_int, dtype=np.int64)

    # Run simulation
    sim_result = day_sim_3(
        psl, ads, var, lt_int, p_terms, s_terms, days,
        demand_sample, lt_sample, cost_of_capital,
    )

    sim_sales = sim_result[0]
    inventory = sim_result[1]
    deval = sim_result[4]
    cube = inventory * length * width * height
    avg_invest_acct = (sim_result[1] * cost) + (sim_result[2] * avg_sale_price) - (sim_result[3] * cost)
    profit = ((sim_sales * gm) - avg_invest_acct * invst_charge) - (deval * (cost / 2) / deval_days)

    if inventory > 0:
        ppc = profit / cube
    else:
        ppc = 0.0

    return np.array([float64(psl), profit, inventory, sim_sales, cube, ppc])
