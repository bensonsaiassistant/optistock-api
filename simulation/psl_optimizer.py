"""PSL optimizer — finds optimal Profit Sharing Level.

Optimized for speed:
- Pre-generates random samples in pure numpy (outside JIT)
- Uses JIT-compiled inner loop for PSL sweep
- Python wrapper handles sample generation
- Default sim_days=200, max 30 PSL levels — fast but accurate
"""

from typing import Optional

import numpy as np
from numba import njit, int64, float64

from .demand_dist import calc_nb_array_ln, numba_choice
from .day_sim import day_sim_3

# Module-level constants
deval_days = 548


def _generate_lead_time_sample(
    lt: float,
    lt_variance: float,
    days: int,
) -> np.ndarray:
    """Generate lead time samples from a gamma distribution (pure numpy, fast)."""
    lt_var_expon = 1.3
    if lt_variance > 0.0 and lt > 0.0:
        a = (lt * lt) / lt_variance
        b = lt_variance / lt
    else:
        a = (lt ** 2) / (lt ** lt_var_expon)
        b = (lt ** lt_var_expon) / lt

    a = max(a, 0.001)
    b = max(b, 0.001)

    n_periods = days * 2
    lt_values = np.random.gamma(a, b, n_periods)
    lt_lengths = np.random.gamma(a, b, n_periods)

    lt_int_values = np.maximum(np.round(lt_values).astype(np.int64), np.int64(1))
    lt_int_lengths = np.maximum(np.round(lt_lengths).astype(np.int64), np.int64(1))

    flat_samples = np.repeat(lt_int_values, lt_int_lengths)
    n_fill = min(len(flat_samples), days)
    lt_sample = np.full(days, int(round(lt)), dtype=np.int64)
    lt_sample[:n_fill] = flat_samples[:n_fill]
    return lt_sample


@njit
def _calc_opti_psl_3_inner(
    psl_values: np.ndarray,
    ads: float64,
    var: float64,
    lt_int: int64,
    gm: float64,
    cost: float64,
    avg_sale_price: float64,
    length: float64,
    width: float64,
    height: float64,
    p_terms: int64,
    s_terms: int64,
    days: int64,
    demand_sample: np.ndarray,
    lt_sample: np.ndarray,
    cost_of_capital: float64,
) -> np.ndarray:
    """Sweep multiple PSL levels in one JIT call.

    Returns array of shape (n_psls, 6): [psl, profit, inventory, sales, cube, ppc]
    """
    invst_charge = cost_of_capital / 365.0
    n_psls = len(psl_values)
    results = np.zeros((n_psls, 6))

    for idx in range(n_psls):
        psl = int64(psl_values[idx])
        sim_result = day_sim_3(
            psl, ads, var, lt_int, p_terms, s_terms, days,
            demand_sample, lt_sample, cost_of_capital,
        )

        sim_sales = sim_result[0]
        inventory = sim_result[1]
        deval = sim_result[4]

        avg_invest_acct = (
            inventory * cost
            + sim_result[2] * avg_sale_price
            - sim_result[3] * cost
        )
        cube = inventory * length * width * height

        profit = (
            (sim_sales * gm)
            - avg_invest_acct * invst_charge
            - (deval * (cost / 2) / deval_days)
        )

        ppc = profit / cube if cube > 0 else 0.0

        results[idx, 0] = float64(psl)
        results[idx, 1] = profit
        results[idx, 2] = inventory
        results[idx, 3] = sim_sales
        results[idx, 4] = cube
        results[idx, 5] = ppc

    return results


def calc_opti_psl_3(
    ads: float,
    var: float,
    lt: float,
    gm: float,
    cost: float,
    avg_sale_price: float,
    length: float,
    width: float,
    height: float,
    p_terms: int,
    s_terms: int,
    min_of_1: int,
    cost_of_capital: float = 0.14,
    lt_variance: float = 0.0,
    sim_days: int = 200,
) -> np.ndarray:
    """Find optimal PSL by sweeping levels.

    Uses a hybrid approach:
    - Pre-generates random samples in pure numpy (fast)
    - Sweeps PSL levels in a single JIT call (avoids repeated overhead)
    - Returns the best PSL and its metrics

    Returns
    -------
    np.ndarray of length 6: [optimal_psl, profit, inventory, sales, cube, ppc]
    """
    # Guard: require positive demand and lead time
    if ads <= 0 or var <= 0 or lt <= 0:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    days = sim_days
    lt_int = int(round(lt))

    # Generate PSL values to sweep (capped at 30 for speed)
    psls_to_calc = int(max(1, ads) * max(1, lt) * 5)
    psls_to_calc = min(psls_to_calc, 30)
    psl_values = np.arange(1, psls_to_calc + 1, dtype=np.float64)

    # Generate demand samples
    use_pre_sampled = (ads / var < 0.95) and (var - ads > 0.1)
    if use_pre_sampled:
        demand_array = calc_nb_array_ln(ads, var)
        demand_size = np.arange(demand_array.size)
        n_to_sample = min(days, demand_array.size)
        demand_sample = numba_choice(demand_size, demand_array, n_to_sample)
        demand_sample = demand_sample.astype(np.int64)
    else:
        demand_sample = np.random.poisson(ads, days).astype(np.int64)

    # Generate lead time samples (pure numpy, fast)
    lt_sample = _generate_lead_time_sample(lt, lt_variance, days)

    # Run PSL sweep in single JIT call
    results = _calc_opti_psl_3_inner(
        psl_values, ads, var, int64(lt_int), gm, cost,
        avg_sale_price, length, width, height,
        int64(p_terms), int64(s_terms), int64(days),
        demand_sample, lt_sample, cost_of_capital,
    )

    # Handle min_of_1 flag
    if min_of_1 == 1 and results.shape[0] > 1:
        results = results[1:]

    if results.shape[0] == 0:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    best_idx = int(np.argmax(results[:, 1]))
    return results[best_idx]


def get_all_psls(
    id_arr: np.ndarray,
    ads_arr: np.ndarray,
    var_arr: np.ndarray,
    lt_arr: np.ndarray,
    gm_arr: np.ndarray,
    cost_arr: np.ndarray,
    avg_sale_price_arr: np.ndarray,
    length_arr: np.ndarray,
    width_arr: np.ndarray,
    height_arr: np.ndarray,
    pterms_arr: np.ndarray,
    sterms_arr: np.ndarray,
    min_of_1_arr: np.ndarray,
    cost_of_capital: float = 0.14,
    lt_var_arr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Batch version of calc_opti_psl_3. Runs each item sequentially but fast."""
    n = id_arr.size
    result_arr = np.zeros((n, 6))
    run_on_row = (ads_arr > 0.0) & (lt_arr > 0.0) & (gm_arr > 0.0)

    for i in range(n):
        if run_on_row[i]:
            lt_var = 0.0
            if lt_var_arr is not None:
                lt_var = lt_var_arr[i]
            result_line = calc_opti_psl_3(
                ads_arr[i], var_arr[i], lt_arr[i], gm_arr[i], cost_arr[i],
                avg_sale_price_arr[i], length_arr[i], width_arr[i], height_arr[i],
                int(pterms_arr[i]), int(sterms_arr[i]), int(min_of_1_arr[i]),
                cost_of_capital, lt_var,
            )
        elif min_of_1_arr[i] == 0:
            result_line = np.zeros(6)
        else:
            result_line = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        result_arr[i, :] = result_line

    return result_arr
