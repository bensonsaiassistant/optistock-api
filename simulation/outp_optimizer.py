"""OUTP optimizer — finds the Order Up To Point (OUTP) that maximizes profit.

Architecture:
- `_calc_opti_outp_inner` is @njit — sweeps OUTP levels for one item (JIT-compiled)
- `calc_opti_outp` is the Python wrapper — generates random samples, calls the JIT inner function
- `get_all_outps` uses ThreadPoolExecutor for parallel execution across items;
  Numba releases the GIL during JIT computation, so threads run truly in parallel.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from numba import njit, int64, float64

from .demand_dist import calc_nb_array_ln, numba_choice
from .day_sim import day_sim

# Module-level constant: days before inventory devalues
DEVAL_DAYS = 548


def _generate_lead_time_sample(
    lt: float,
    lt_variance: float,
    days: int,
) -> np.ndarray:
    """Generate lead time samples from a Gamma distribution.

    Parameters
    ----------
    lt : float
        Expected lead time in days.
    lt_variance : float
        Variance of lead time.  When zero, a hardcoded exponent (1.3) is used.
    days : int
        Number of days to simulate.

    Returns
    -------
    np.ndarray of shape (days,) with lead-time values per day.
    """
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
def _calc_opti_outp_inner(
    outp_values: np.ndarray,
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
    """Sweep multiple OUTP levels in a single JIT-compiled call.

    For each candidate OUTP value, runs the day-level simulation, computes
    financial metrics (profit, cube usage, profit-per-cube), and returns
    the results so the caller can pick the best.

    Parameters
    ----------
    outp_values : 1-D array of candidate OUTP integers
    ads : Average daily sales
    var : Demand variance
    lt_int : Lead time in days (integer)
    gm : Gross margin per unit
    cost : Unit cost
    avg_sale_price : Average selling price
    length/width/height : Item dimensions (feet)
    p_terms : Payment terms (days until you pay supplier)
    s_terms : Sales terms (days until customer pays you)
    days : Simulation horizon
    demand_sample : Pre-generated demand values
    lt_sample : Pre-generated lead-time values per day
    cost_of_capital : Annual cost-of-capital rate (e.g. 0.14 = 14%)

    Returns
    -------
    np.ndarray of shape (n_outps, 6):
        columns = [outp, profit, inventory, sales, cube, profit_per_cube]
    """
    invst_charge = cost_of_capital / 365.0
    n_outps = len(outp_values)
    results = np.zeros((n_outps, 6))

    for idx in range(n_outps):
        outp = int64(outp_values[idx])
        sim_result = day_sim(
            outp, ads, var, lt_int, p_terms, s_terms, days,
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
            - (deval * (cost / 2) / DEVAL_DAYS)
        )

        ppc = profit / cube if cube > 0 else 0.0

        results[idx, 0] = float64(outp)
        results[idx, 1] = profit
        results[idx, 2] = inventory
        results[idx, 3] = sim_sales
        results[idx, 4] = cube
        results[idx, 5] = ppc

    return results


def calc_opti_outp(
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
    """Find the Order Up To Point (OUTP) that maximizes profit.

    Uses a hybrid approach:
    1. Pre-generates demand and lead-time samples in pure NumPy (fast).
    2. Sweeps all candidate OUTP levels in a single JIT-compiled call
       (avoids repeated Numba overhead).
    3. Returns the OUTP value and its associated financial metrics.

    The sweep is capped at 30 levels for speed.

    Parameters
    ----------
    ads : Average daily sales
    var : Demand variance
    lt : Supplier lead time in days
    gm : Gross margin per unit (sale_price − cost)
    cost : Unit cost
    avg_sale_price : Average selling price
    length/width/height : Item dimensions (feet)
    p_terms : Payment terms in days
    s_terms : Sales terms in days
    min_of_1 : If 1, skip OUTP=0 (force at least 1 unit)
    cost_of_capital : Annual capital cost rate (default 0.14)
    lt_variance : Lead-time variance (0 = use default)
    sim_days : Simulation horizon in days (default 200)

    Returns
    -------
    np.ndarray of length 6: [optimal_outp, profit, inventory, sales, cube, profit_per_cube]
    """
    # Guard: require positive demand and lead time
    if ads <= 0 or var <= 0 or lt <= 0:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    days = sim_days
    lt_int = int(round(lt))

    # Generate OUTP values to sweep (capped at 30 for speed)
    outps_to_calc = int(max(1, ads) * max(1, lt) * 5)
    outps_to_calc = min(outps_to_calc, 30)
    outp_values = np.arange(1, outps_to_calc + 1, dtype=np.float64)

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

    # Run OUTP sweep in single JIT call
    results = _calc_opti_outp_inner(
        outp_values, ads, var, int64(lt_int), gm, cost,
        avg_sale_price, length, width, height,
        int64(p_terms), int64(s_terms), int64(days),
        demand_sample, lt_sample, cost_of_capital,
    )

    # Handle min_of_1 flag: skip the first row (OUTP=0)
    if min_of_1 == 1 and results.shape[0] > 1:
        results = results[1:]

    if results.shape[0] == 0:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    best_idx = int(np.argmax(results[:, 1]))
    return results[best_idx]


def _optimize_single_item(args):
    """Helper for parallel batch optimization — runs one item's OUTP sweep."""
    (i, ads, var, lt, gm, cost, sale_price, length, width, height,
     p_terms, s_terms, min_of_1, coc, lt_var) = args

    if ads <= 0 or lt <= 0 or gm <= 0:
        if min_of_1 == 0:
            return i, np.zeros(6)
        return i, np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    result = calc_opti_outp(
        ads, var, lt, gm, cost, sale_price,
        length, width, height,
        p_terms, s_terms, min_of_1, coc, lt_var,
    )
    return i, result


def get_all_outps(
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
    max_workers: int = 4,
) -> np.ndarray:
    """Batch OUTP optimization with parallel execution across items.

    Uses ThreadPoolExecutor — Numba releases the GIL during JIT computation,
    so threads run truly in parallel on multiple CPU cores.

    Parameters
    ----------
    max_workers : Number of parallel threads (default 4).
                  Numba's internal parallelism is separate from this thread pool.

    Returns
    -------
    np.ndarray of shape (n, 6) with columns
    [outp, profit, inventory, sales, cube, profit_per_cube] per item.
    """
    n = id_arr.size
    result_arr = np.zeros((n, 6))

    # Build argument tuples for each item
    tasks = []
    for i in range(n):
        lt_var = 0.0
        if lt_var_arr is not None:
            lt_var = lt_var_arr[i]
        tasks.append((
            i,
            ads_arr[i], var_arr[i], lt_arr[i], gm_arr[i], cost_arr[i],
            avg_sale_price_arr[i], length_arr[i], width_arr[i], height_arr[i],
            int(pterms_arr[i]), int(sterms_arr[i]), int(min_of_1_arr[i]),
            cost_of_capital, lt_var,
        ))

    # Run in parallel (Numba JIT releases GIL, so threads are truly parallel)
    with ThreadPoolExecutor(max_workers=min(max_workers, n)) as executor:
        for i, result in executor.map(_optimize_single_item, tasks):
            result_arr[i, :] = result

    return result_arr
