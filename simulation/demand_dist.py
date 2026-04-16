"""Demand distribution utilities — log-space Negative Binomial & Poisson PMFs."""

from typing import Tuple

import numpy as np
import math as m
from numba import njit, prange, int64, float64


@njit
def neg_bin_ln(qty: float64, r: float64, p: float64) -> float64:
    """Log-space Negative Binomial PMF (from sim2.py)."""
    log_result = (m.lgamma(qty + r) - m.lgamma(qty + 1) - m.lgamma(r)
                  + (r * m.log(p)) + (qty * m.log(1 - p)))
    return log_result


@njit
def poisson_pmf(k: float64, lambd: float64) -> float64:
    """Log-space Poisson PMF (from sim2.py)."""
    log_result = k * m.log(lambd) - lambd - m.lgamma(k + 1)
    return log_result


@njit
def calc_nb_array_ln(mean: float64, var: float64) -> np.ndarray:
    """Build a Negative Binomial (or Poisson fallback) probability array in log-space.

    Returns a 1-D array of probabilities that sums to 1.0.
    """
    nbarray = np.zeros(2000, dtype=np.float64)
    if mean == 0:
        return nbarray

    if mean >= var:
        # Poisson fallback
        for i in range(nbarray.size):
            nbarray[i] = poisson_pmf(i, mean)
            if np.exp(nbarray[i]) < 1e-50 and i > mean:
                nbarray = nbarray[:i]
                break
    else:
        # Negative Binomial
        r = -(mean * var) / (mean - var)
        p = r / (r + mean)
        for i in range(0, nbarray.size):
            nbarray[i] = neg_bin_ln(i, r, p)
            if np.exp(nbarray[i]) < 1e-20 and i > mean:
                nbarray = nbarray[:i]
                break

    # Normalize
    if mean > 50:
        max_val = nbarray.max()
        nbarray = np.exp(nbarray - max_val)
        nbarray = nbarray / nbarray.sum()
    else:
        nbarray = np.exp(nbarray)
        nbarray /= np.sum(nbarray)

    return nbarray


@njit
def numba_choice(population: np.ndarray, weights: np.ndarray, k: int64) -> np.ndarray:
    """Weighted random sampling WITH replacement.

    Parameters
    ----------
    population : array of values to sample from
    weights : probability weights (must sum > 0)
    k : number of samples

    Returns
    -------
    Array of length k with sampled values.
    """
    wc = np.cumsum(weights)
    m_ = wc[-1]
    n_pop = population.shape[0]
    sample = np.empty(k, population.dtype)
    for i in range(k):
        r = m_ * np.random.rand()
        idx = np.searchsorted(wc, r, side='right')
        idx = min(idx, n_pop - 1)  # clamp to valid range
        sample[i] = population[idx]
    return sample


@njit
def find_cdf_indexes(pdf: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return interpolated CDF index values for each quantile target in x."""
    cdf = np.cumsum(pdf)
    total = cdf[-1]
    # Guard against zero-sum PDF
    if total <= 0.0:
        return np.zeros_like(x, dtype=np.float64)
    cdf = cdf / total
    result = np.empty_like(x, dtype=np.float64)
    for i in range(x.shape[0]):
        index = np.searchsorted(cdf, x[i])
        if index == 0:
            result[i] = 0.0
        elif index >= len(pdf):
            result[i] = float(index - 1)
        else:
            left_cdf = cdf[index - 1]
            right_cdf = cdf[index]
            denom = right_cdf - left_cdf
            if denom == 0.0:
                result[i] = float(index - 1)
            else:
                result[i] = (index - 1) + (x[i] - left_cdf) / denom
    return result
