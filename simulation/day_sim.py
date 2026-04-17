"""Day-level inventory simulation engine.

Optimized for speed using scalar accumulation (no per-day arrays) and Numba JIT.

Inventory flow per day:
1. **Demand** is sampled (Negative Binomial or Poisson).
2. **Sales** = min(demand, inventory) — can't sell what you don't have.
3. **Order** placed if inventory is below target (OUTP).
4. **Receipts** arrive after lead time days.
5. **Financials** track accounts receivable, payable, and devaluation.
"""
# rebuild-marker-20260417-0230

import numpy as np
import math as m
from numba import njit, int64, float64

from .demand_dist import calc_nb_array_ln, numba_choice

# Days before inventory devalues (≈ 18 months for Amazon FBA long-term storage)
DEVAL_DAYS = 548


@njit
def day_sim(
    outp: int64,
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
    """Run a day-by-day inventory simulation for one OUTP level.

    The simulation models daily demand, sales, ordering, receipts, and
    financial tracking (accounts receivable, accounts payable, devaluation).

    Parameters
    ----------
    outp : Order Up To Point — target inventory level to maintain.
    ads : Average daily sales (mean demand).
    var : Demand variance.
    lt : Supplier lead time in days (integer).
    p_terms : Payment terms — days until you pay your supplier.
    s_terms : Sales terms — days until customers pay you.
    days : Simulation horizon.
    demand_sample : Pre-generated daily demand values (used when NB distribution applies).
    lt_sample : Pre-generated daily lead-time values.
    cost_of_capital : Annual cost-of-capital rate (e.g. 0.14 = 14%).

    Returns
    -------
    np.array([avg_sales, avg_inventory, avg_ar, avg_ap, avg_deval])
        All averages are computed over the post-warmup period.
    """
    start_inv = max(outp, int64(1))
    max_order_size = max(int64(round(ads * 2, 0)), int64(1))

    use_pre_sampled = (ads / var < 0.95) and (var - ads > 0.1) if var > 0 else False

    # Seed Numba's RNG for deterministic results across all OUTP candidates
    np.random.seed(42)

    # --- Scalar state variables ---
    inv = int64(start_inv)       # current on-hand inventory
    oo = int64(0)                # units on order (not yet received)

    # Ring buffers for AR/AP tracking (circular arrays to avoid shifting)
    ar_ring = np.zeros(366, dtype=np.int64)  # accounts receivable ring
    ap_ring = np.zeros(366, dtype=np.int64)  # accounts payable ring
    inv_ring = np.zeros(550, dtype=np.int64)  # inventory snapshot ring

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

    # Warmup period: enough time for several order cycles to stabilize
    # At least 2 lead times + 3 order frequencies, or 30 days minimum
    warmup_days = min(int64(2) * lt + int64(3) * int64(7), int64(100))  # v5-steady-state
    warmup_end = min(warmup_days, days)

    for i in range(days):
        # 1. Demand
        if use_pre_sampled:
            demand = int64(demand_sample[i % demand_sample.size])
        else:
            demand = int64(np.random.poisson(ads)) if ads > 0 else int64(0)

        # 2. Sales — can't sell more than we have
        sold = min(demand, inv)

        # 3. Update inventory
        inv = max(inv - sold, int64(0))

        # 4. Order placement — replenish toward OUTP
        order_qty = min(max(outp - inv - oo, int64(0)), max_order_size)
        oo += order_qty

        # 5. Receipts — orders arrive after 'lt' days
        if i >= lt:
            recv = min(oo, int64(1000000))
            inv += recv
            oo -= recv

        # 6. Accounts Receivable — customers pay after s_terms days
        ar_balance += sold
        ar_ring[i % 366] = sold
        if i >= s_terms:
            ar_balance -= ar_ring[(i - s_terms) % 366]

        # 7. Accounts Payable — we pay supplier after p_terms days
        ap_ring[i % 366] = order_qty
        if i >= p_terms:
            ap_balance -= ap_ring[(i - p_terms) % 366]

        # 8. Devaluation — track inventory from deval_days ago
        inv_ring[i % 550] = inv
        if i >= DEVAL_DAYS:
            sum_deval += max(int64(0), inv_ring[(i - DEVAL_DAYS) % 550] - sold)

        # 9. Accumulate stats (post-warmup only)
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
# rebuild trigger 2026-04-17
# rebuild trigger 2026-04-17-0230
# rebuild-trigger-20260417-v5
