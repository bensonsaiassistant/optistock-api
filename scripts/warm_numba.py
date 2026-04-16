"""Warm up Numba JIT-compiled functions to avoid cold-start latency on Modal.

Run this script standalone: `python -m scripts.warm_numba`
Or import `warm_numba_functions()` in the FastAPI lifespan handler.
"""

import sys
import time

import numpy as np


def warm_numba_functions():
    """Call every Numba @njit function once with small dummy data to trigger JIT compilation.

    Returns a list of (function_name, time_ms) tuples.
    """
    timings = []

    # ─── demand_dist ────────────────────────────────────────────────────────
    from simulation.demand_dist import (
        neg_bin_ln,
        calc_nb_array_ln,
        numba_choice,
        find_cdf_indexes,
    )

    # neg_bin_ln(qty, r, p) — simple scalar
    t0 = time.time()
    neg_bin_ln(1, 2.5, 0.6)
    timings.append(("demand_dist.neg_bin_ln", (time.time() - t0) * 1000))

    # calc_nb_array_ln(mean, var)
    #   mean >= var → Poisson path
    #   mean < var  → Negative Binomial path
    t0 = time.time()
    calc_nb_array_ln(5.0, 3.0)  # Poisson
    calc_nb_array_ln(3.0, 5.0)  # NB
    timings.append(("demand_dist.calc_nb_array_ln", (time.time() - t0) * 1000))

    # numba_choice(population, weights, k)
    t0 = time.time()
    pop = np.arange(20, dtype=np.int64)
    weights = np.ones(20, dtype=np.float64) / 20.0
    numba_choice(pop, weights, 5)
    timings.append(("demand_dist.numba_choice", (time.time() - t0) * 1000))

    # find_cdf_indexes(pdf, x)
    t0 = time.time()
    pdf = np.array([0.1, 0.2, 0.3, 0.25, 0.15], dtype=np.float64)
    x = np.array([0.25, 0.5, 0.75], dtype=np.float64)
    find_cdf_indexes(pdf, x)
    timings.append(("demand_dist.find_cdf_indexes", (time.time() - t0) * 1000))

    # ─── day_sim ────────────────────────────────────────────────────────────
    from simulation.day_sim import day_sim

    t0 = time.time()

    # day_sim — keep days small for warmup
    # Condition: ads/var >= 0.95 triggers Poisson path (no demand_sample dependency)
    warmup_days = 100
    dummy_demand = np.zeros(warmup_days, dtype=np.int64)
    dummy_lt = np.full(warmup_days, 3, dtype=np.int64)  # 3-day lead time

    day_sim(
        outp=5,
        ads=2.0,
        var=2.0,
        lt=3,
        p_terms=30,
        s_terms=14,
        days=warmup_days,
        demand_sample=dummy_demand,
        lt_sample=dummy_lt,
        cost_of_capital=0.14,
    )
    timings.append(("day_sim.day_sim", (time.time() - t0) * 1000))

    # ─── outp_optimizer ──────────────────────────────────────────────────────
    from simulation.outp_optimizer import calc_opti_outp, get_all_outps

    t0 = time.time()
    calc_opti_outp(
        ads=2.0,
        var=2.0,
        lt=3.0,
        gm=5.0,
        cost=2.0,
        avg_sale_price=10.0,
        length=1.0,
        width=1.0,
        height=1.0,
        p_terms=30,
        s_terms=14,
        min_of_1=0,
        cost_of_capital=0.14,
    )
    timings.append(("outp_optimizer.calc_opti_outp", (time.time() - t0) * 1000))

    t0 = time.time()
    n_items = 2
    get_all_outps(
        id_arr=np.arange(n_items),
        ads_arr=np.full(n_items, 2.0),
        var_arr=np.full(n_items, 2.0),
        lt_arr=np.full(n_items, 3.0),
        gm_arr=np.full(n_items, 5.0),
        cost_arr=np.full(n_items, 2.0),
        avg_sale_price_arr=np.full(n_items, 10.0),
        length_arr=np.full(n_items, 1.0),
        width_arr=np.full(n_items, 1.0),
        height_arr=np.full(n_items, 1.0),
        pterms_arr=np.full(n_items, 30.0),
        sterms_arr=np.full(n_items, 14.0),
        min_of_1_arr=np.zeros(n_items),
        cost_of_capital=0.14,
    )
    timings.append(("outp_optimizer.get_all_outps", (time.time() - t0) * 1000))

    return timings


if __name__ == "__main__":
    print("Warming up Numba JIT functions…")
    results = warm_numba_functions()
    print(f"\n{'Function':<45} {'Time (ms)':>10}")
    print("-" * 57)
    total_ms = 0.0
    for name, ms in results:
        print(f"  {name:<43} {ms:>8.0f}")
        total_ms += ms
    print("-" * 57)
    print(f"  {'TOTAL':<43} {total_ms:>8.0f}")
    print("\nAll Numba functions JIT-compiled successfully!")
