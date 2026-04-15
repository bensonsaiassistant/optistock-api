"""PSL optimizer — finds optimal Profit Sharing Level."""

import numpy as np
import math as m
from numba import njit, prange, int64, float64

from .demand_dist import calc_nb_array_ln, numba_choice
from .day_sim import day_sim_3

# Module-level constants
deval_days = 548
overhead_cost_per_unit = 12.25
cost_per_cubic_ft = 0.005
lt_var_expon = 1.3


@njit
def calc_opti_psl_3(ads, var, lt, gm, cost, avg_sale_price, length, width, height,
                    p_terms, s_terms, min_of_1, cost_of_capital=0.14, oos_penalty=0.0):
    """Find optimal PSL by sweeping levels.

    Args:
        oos_penalty: penalty multiplier per avg OOS day (default 0.0, subtracted from profit)

    Returns: [optimal_psl_index, profit, inventory, sales, cube, ppc]
    """
    invst_charge = cost_of_capital / 365
    days = 100000

    p_terms = int64(p_terms)
    s_terms = int64(s_terms)
    min_of_1 = int64(min_of_1)

    psls_to_calc = int(max(1, ads) * max(1, lt) * 10)

    profit_arr = np.zeros(psls_to_calc, dtype='float64')
    inventory_arr = np.zeros(psls_to_calc, dtype='float64')
    sales_arr = np.zeros(psls_to_calc, dtype='float64')
    cube_arr = np.zeros(psls_to_calc, dtype='float64')
    ppc_arr = np.zeros(psls_to_calc, dtype='float64')
    deval_arr = np.zeros(psls_to_calc, dtype='float64')

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
    lt_sample = lt_sample.astype(np.int64)

    # Generate demand samples
    if ads / var < 0.95 and var - ads > 0.1:
        demand_array = calc_nb_array_ln(ads, var)
        demand_size = np.arange(demand_array.size)
        demand_sample = numba_choice(demand_size, demand_array, days)
        demand_sample = demand_sample.astype(np.int64)
    else:
        demand_sample = np.zeros(1, dtype=np.int64)

    # Sweep PSL levels
    for i in range(1, psls_to_calc):
        sim_result = day_sim_3(i, ads, var, lt, p_terms, s_terms, days, demand_sample,
                               lt_sample, cost_of_capital)

        sales_arr[i] = sim_result[0]
        inventory_arr[i] = sim_result[1]
        deval_arr[i] = sim_result[4]
        avg_oos_days = sim_result[5]
        avg_lost_sales = sim_result[6]

        avg_invest_acct = (sim_result[1] * cost) + (sim_result[2] * avg_sale_price) - (sim_result[3] * cost)
        cube_arr[i] = inventory_arr[i] * length * width * height

        # Profit formula with capital cost and OOS penalty
        oos_penalty_cost = oos_penalty * avg_oos_days
        profit_arr[i] = ((sim_result[0] * gm) - avg_invest_acct * invst_charge) - (deval_arr[i] * ((cost / 2)) / deval_days) - oos_penalty_cost

        # Calculate profit per cube
        if inventory_arr[i] >= 0:
            ppc_arr[i] = profit_arr[i] / cube_arr[i]
        else:
            ppc_arr[i] = 0

        # Early stopping: if profit starts declining over a window
        if i >= 20 and profit_arr[i - 5:i].mean() < profit_arr[i - 20:i - 3].mean():
            sales_arr = sales_arr[:i]
            inventory_arr = inventory_arr[:i]
            profit_arr = profit_arr[:i]
            cube_arr = cube_arr[:i]
            ppc_arr = ppc_arr[:i]
            break

        i += 1

    # Handle min_of_1 flag
    if min_of_1 == 1:
        sales_arr = sales_arr[1:]
        inventory_arr = inventory_arr[1:]
        profit_arr = profit_arr[1:]
        cube_arr = cube_arr[1:]
        ppc_arr = ppc_arr[1:]

    # Find optimal PSL
    result = np.asarray([np.argmax(profit_arr), profit_arr[np.argmax(profit_arr)],
                         inventory_arr[np.argmax(profit_arr)],
                         sales_arr[np.argmax(profit_arr)], cube_arr[np.argmax(profit_arr)],
                         ppc_arr[np.argmax(profit_arr)]])

    if min_of_1 == 1:
        result[0] = result[0] + 1

    full_result = np.stack(
        (np.arange(profit_arr.size).astype(np.int64) - 1, profit_arr, inventory_arr, sales_arr, cube_arr,
         ppc_arr), axis=1)

    return result


@njit(parallel=True)
def get_all_psls(id_arr, ads_arr, var_arr, lt_arr, gm_arr, cost_arr, avg_sale_price_arr, length_arr, width_arr,
                 height_arr, pterms_arr, sterms_arr, min_of_1_arr, cost_of_capital=0.14):
    """Batch version of calc_opti_psl_3 using parallel execution."""
    result_arr = np.zeros((id_arr.size, 6))
    run_on_row = (ads_arr > 0.0) & (lt_arr > 0.0) & (gm_arr > 0.0)

    for i in prange(id_arr.size):
        if run_on_row[i] == True:
            result_line = calc_opti_psl_3(ads_arr[i], var_arr[i], lt_arr[i], gm_arr[i], cost_arr[i],
                                          avg_sale_price_arr[i], length_arr[i], width_arr[i], height_arr[i],
                                          pterms_arr[i], sterms_arr[i], min_of_1_arr[i], cost_of_capital)
        elif min_of_1_arr[i] == 0:
            result_line = np.zeros(6)
        else:
            result_line = np.zeros(6)
            result_line[0] = 1
            result_line[2] = 1
        result_arr[i, :] = result_line

    return result_arr
