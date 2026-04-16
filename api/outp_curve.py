"""Extract full OUTP profit curve for visualization."""
import numpy as np
from numba import int64
from simulation.outp_optimizer import _calc_opti_outp_inner
from simulation.demand_dist import calc_nb_array_ln, numba_choice
from simulation.day_sim import day_sim


def get_outp_curve(
    ads: float, var: float, lt: float, gm: float, cost: float,
    sale_price: float, length: float, width: float, height: float,
    p_terms: int, s_terms: int, days: int = 200,
    cost_of_capital: float = 0.14, lt_variance: float = 0.0,
    seed: int = 42
) -> list[dict]:
    """Return profit curve across all OUTP values."""
    import numpy as np
    
    # Generate OUTP values to sweep (same logic as calc_opti_outp)
    lt_int = int(round(lt))
    outps_to_calc = int(max(1, ads) * max(1, lt) * 3)
    outps_to_calc = max(outps_to_calc, 1)
    outp_values = np.arange(1, outps_to_calc + 1, dtype=np.float64)
    
    # Generate demand samples (deterministic)
    rng = np.random.RandomState(seed)
    use_pre_sampled = (ads / var < 0.95) and (var - ads > 0.1)
    if use_pre_sampled:
        demand_array = calc_nb_array_ln(ads, var)
        demand_size = np.arange(demand_array.size)
        n_to_sample = min(days, demand_array.size)
        demand_sample = numba_choice(demand_size, demand_array, n_to_sample)
        demand_sample = demand_sample.astype(np.int64)
    else:
        demand_sample = rng.poisson(ads, days).astype(np.int64)
    
    # Generate lead time samples
    np.random.seed(99)
    from simulation.outp_optimizer import _generate_lead_time_sample
    lt_sample = _generate_lead_time_sample(lt, lt_variance, days)
    
    # Run sweep
    results = _calc_opti_outp_inner(
        outp_values, ads, var, int64(lt_int), gm, cost,
        sale_price, length, width, height,
        int64(p_terms), int64(s_terms), int64(days),
        demand_sample, lt_sample, cost_of_capital,
    )
    
    # Skip first row if it's OUTP=0
    if results.shape[0] > 1:
        results = results[1:]
    
    return [{"outp": int(r[0]), "profit": float(r[1])} for r in results]
