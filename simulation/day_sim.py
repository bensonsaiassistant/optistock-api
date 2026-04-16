"""Day-level inventory simulation engine."""

import numpy as np
import math as m
from numba import njit, prange, int64, float64

from .demand_dist import calc_nb_array_ln, numba_choice


@njit
def day_sim_3(psl, ads, var, lt, p_terms, s_terms, days, demand_sample, lt_sample,
              cost_of_capital=0.14):
    """Run a day-by-day inventory simulation.

    Returns: [avg_sales, avg_inventory, avg_ar, avg_ap, avg_deval]
    """
    deval_days = 548
    start_inv = max(psl, 1)
    inventory = np.zeros(days, dtype=np.int64)
    orders = np.zeros(days, dtype=np.int64)
    receipts = np.zeros(days, dtype=np.int64)
    on_order = np.zeros(days, dtype=np.int64)
    sales = np.zeros(days, dtype=np.int64)
    ar = np.zeros(days, dtype=np.int64)
    ap = np.zeros(days, dtype=np.int64)
    deval = np.zeros(days, dtype=np.int64)
    max_order_size = int64(round(ads * 2, 0))

    # Use pre-sampled demand if variance is high enough; otherwise Poisson
    if ads / var < 0.95 and var - ads > 0.1:
        demand = demand_sample
    else:
        demand = np.zeros(days, dtype=np.int64)

    lead_times = lt_sample
    inventory[0] = start_inv

    for i in range(0, days):
        if not (ads / var < 0.95 and var - ads > 0.1):
            demand[i] = np.int64(np.random.poisson(ads))

        # Order logic
        if i == 0:
            orders[i] = min(max(psl - inventory[i], 0), max(max_order_size, 1))
        else:
            orders[i] = min(max(psl - inventory[i] - on_order[i - 1], 0), max(max_order_size, 1))

        # Receipts
        if i + lead_times[i] < days:
            receipts[i + lead_times[i]] = orders[i] + receipts[i + lead_times[i]]

        # On-order tracking
        if i == 0:
            on_order[i] = orders[i] - receipts[i]
        else:
            on_order[i] = orders[i] + on_order[i - 1] - receipts[i]

        # Sales
        sales[i] = min(demand[i], inventory[i])

        # Inventory update
        if i < days - 1:
            inventory[i + 1] = max(inventory[i] + receipts[i] - sales[i], 0)

        # Accounts receivable / payable
        if i == 0:
            ar[i] = sales[i]
            ap[i] = receipts[i]
        if i < s_terms > 0:
            ar[i] = sales[i] + ar[i - 1]
        if i >= s_terms > 0:
            ar[i] = sales[i] + ar[i - 1] - sales[i - s_terms]
        if i < p_terms > 0:
            ap[i] = receipts[i] + ap[i - 1]
        if i >= p_terms > 0:
            ap[i] = receipts[i] + ap[i - 1] - receipts[i - p_terms]

        # Devaluation tracking
        if i >= deval_days:
            deval[i] = max(inventory[i - deval_days] - sales[i - deval_days:i].sum(), 0)

        i += 1

    # Compute averages (skip warm-up period)
    start_date = int(round(start_inv / ads * 2, 0))
    if start_date >= days:
        start_date = int(start_inv)

    avg_inventory = np.mean(inventory[start_date:days])
    avg_ar = np.mean(ar[start_date:days])
    avg_ap = np.mean(ap[start_date:days])
    avg_sales = np.mean(sales[start_date:days])
    avg_deval = np.mean(deval[start_date:days])

    return np.array([avg_sales, avg_inventory, avg_ar, avg_ap, avg_deval])


@njit
def calc_single_psl(psl, ads, var, lt, gm, cost, avg_sale_price, length, width, height,
                    p_terms, s_terms, min_of_1, cost_of_capital=0.14):
    """Calculate profit metrics for a single PSL value.

    Returns: [psl, profit, inventory, sales, cube, ppc]
    """
    invst_charge = cost_of_capital / 365
    days = 100000
    deval_days = 548
    overhead_pct_of_cogs = 0.047
    cost_per_cubic_ft = 0.005
    lt_var_expon = 1.5

    p_terms = int64(p_terms)
    s_terms = int64(s_terms)

    # Generate lead time samples
    a = (lt ** 2) / (lt ** lt_var_expon)
    b = (lt ** lt_var_expon) / lt
    lt = int64(lt)
    lt_sample = np.zeros(days * 2)
    w = 0
    while np.count_nonzero(lt_sample) <= days:
        lt_for_period = np.random.gamma(a, b)
        lt_period_len = np.random.gamma(a, b)
        lt_for_period = round(lt_for_period, 0)
        lt_period_len = round(lt_period_len, 0)
        lt_for_period = int64(lt_for_period)
        lt_period_len = int64(lt_period_len)
        lt_slice = np.repeat(lt_for_period, lt_period_len)
        lt_sample[w: w + lt_slice.size] = lt_slice
        w += lt_slice.size
    lt_sample = lt_sample[0: days]
    lt_sample = lt_sample.astype(int64)

    # Generate demand samples
    if ads / var < 0.95 and var - ads > 0.1:
        demand_array = calc_nb_array_ln(ads, var)
        demand_size = np.arange(demand_array.size)
        demand_sample = numba_choice(demand_size, demand_array, days)
    else:
        demand_sample = np.zeros(1, dtype=int64)

    # Run simulation
    sim_result = day_sim_3(psl, ads, var, lt, p_terms, s_terms, days, demand_sample, lt_sample,
                           cost_of_capital)
    sales = sim_result[0]
    inventory = sim_result[1]
    deval = sim_result[4]
    cube = inventory * length * width * height
    avg_invest_acct = (sim_result[1] * cost) + (sim_result[2] * avg_sale_price) - (sim_result[3] * cost)
    profit = ((sim_result[0] * gm) - avg_invest_acct * invst_charge) - (deval * (cost / 2))

    if inventory >= 0:
        ppc = profit / cube
    else:
        ppc = 0

    return np.array([float64(psl), profit, inventory, sales, cube, ppc])
