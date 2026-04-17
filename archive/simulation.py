import numpy as np
import math as m
from numba import njit, prange, int64, vectorize, float64
import pandas as pd


@vectorize
def normal_pdf(x, mean, sd):
    var = float(sd) ** 2
    denom = (2 * m.pi * var) ** .5
    num = m.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return num / denom


@njit
def apply_norm(x, norm_arr, profit_arr):
    if profit_arr.size <= norm_arr.size:
        diff = norm_arr.size - profit_arr.size
        profit_arr = np.concatenate((np.full(diff + 1, profit_arr[profit_arr.size - 1]), profit_arr))
    if norm_arr.size % 2 == 0:
        norm_arr_even = norm_arr
    else:
        norm_arr_even = np.concatenate((np.arange(1), norm_arr))
    if profit_arr.size % 2 == 0:
        profit_arr_even = profit_arr
    else:
        profit_arr_even = np.concatenate((profit_arr, np.arange(1)))
    half_norm = int64(norm_arr_even.size / 2)
    if x < half_norm:
        # trim left side of normal
        part_profit_arr = profit_arr_even[0: x + half_norm]
        part_norm_arr = norm_arr_even[half_norm - x:]
        result = float64((part_norm_arr * part_profit_arr).sum())
    elif half_norm <= x and x + half_norm < profit_arr_even.size:
        part_profit_arr = profit_arr_even[x - half_norm: x + half_norm]
        result = float64((norm_arr_even * part_profit_arr).sum())
    else:
        # trim right side of normal
        part_profit_arr = profit_arr_even[x - half_norm:]
        part_norm_arr = norm_arr_even[0: half_norm + profit_arr_even.size - x]
        result = float64((part_norm_arr * part_profit_arr).sum())
    return result


@vectorize
def neg_bin(qty, r, p):
    result = (m.gamma(qty + r) / (m.gamma(qty + 1) * m.gamma(r))) * (p ** r) * ((1 - p) ** qty)
    if np.isnan(result) is True:
        result = 0.0
    elif np.isinf(result) is True:
        result = 0.0
    return result


@njit
def calc_nb_array(ads, var):
    max_size = np.arange(500)
    if ads / var < 0.95 and var - ads > 0.1:
        r = -(ads * var) / (ads - var)
        p = r / (r + ads)
        atrydex = neg_bin(max_size, r, p)
        dexlen = int(np.argmax(atrydex == 0.0) + 1)
        dex = np.arange(dexlen - 1)
        nbarray = neg_bin(dex, r, p)
        nbarray /= nbarray.sum()
    else:
        nbarray = np.zeros(1)
    return nbarray


@njit
def neg_bin_ln(qty, r, p):
    # result = m.exp(m.lgamma(qty + r) - m.lgamma(qty + 1) - m.lgamma(r)) * (p ** r) * ((1 - p) ** qty)
    # the above works when vectorized if needed
    # should be able to turn remaining terms to logs like the below, but does not work currently when vectorized
    # wasn't working as a vectorized function, so changed to work as loop, now functions as loop
    result = m.exp(m.lgamma(qty + r) - m.lgamma(qty + 1) - m.lgamma(r) + (r * m.log(p)) + (qty * m.log(1 - p)))
    if np.isnan(result) is True:
        result = 0.0
    elif np.isinf(result) is True:
        result = 0.0
    result = round(result, 10)
    return result


@njit
def calc_nb_array_ln(ads, var):
    nbarray = np.zeros(1000, dtype=float64)
    if ads / var < 0.95 and var - ads > 0.1:
        r = -(ads * var) / (ads - var)
        p = r / (r + ads)
        for i in range(0, nbarray.size):
            nbarray[i] = neg_bin_ln(i, r, p)
            if (nbarray[i] == 0.0) and (i > 0):
                nbarray = nbarray[:i + 1]
                break
    else:
        nbarray = np.zeros(1, dtype=float64)
    return nbarray


# for the output of the model, if the 98% quantile value is really low (less than 1/90?), you don't need a variance estimation, you just poisson the ads to get your full probability array
@njit
def calc_quantiles(pmf, quantiles):
    cdf = np.cumsum(pmf)
    quantile_values = np.zeros(len(quantiles))

    for i, q in enumerate(quantiles):
        idx = np.searchsorted(cdf, q)
        quantile_values[i] = idx

    return quantile_values


@njit
def numba_choice(population, weights, k):
    # Get cumulative weights
    wc = np.cumsum(weights)
    # Total of weights
    m_ = wc[-1]
    # Arrays of sample and sampled indices
    sample = np.empty(k, population.dtype)
    sample_idx = np.full(k, -1, np.int32)
    # Sampling loop
    i = 0
    while i < k:
        # Pick random weight value
        r = m_ * np.random.rand()
        # Get corresponding index
        idx = np.searchsorted(wc, r, side='right')
        # Check index was not selected before
        # If not using Numba you can just do `np.isin(idx, sample_idx)`
        for j in range(i):
            if sample_idx[j] == idx:
                continue
        # Save sampled value and index
        sample[i] = population[idx]
        sample_idx[i] = population[idx]
        i += 1
    return sample


@njit
def day_sim(psl, ads, var, lt, p_terms, s_terms, days, demand_sample):
    start_inv = max(psl, 1)
    inventory = np.zeros(days, dtype=int64)
    lead_times = np.zeros(days, dtype=int64)
    orders = np.zeros(days, dtype=int64)
    receipts = np.zeros(days, dtype=int64)
    on_order = np.zeros(days, dtype=int64)
    sales = np.zeros(days, dtype=int64)
    ar = np.zeros(days, dtype=int64)
    ap = np.zeros(days, dtype=int64)
    if ads < var - 0.1:
        demand = demand_sample
    else:
        demand = np.zeros(days, dtype=int64)
    inventory[0] = start_inv
    for i in range(0, days):
        if ads >= var - 0.1:
            demand[i] = np.random.poisson(ads)
        lead_times[i] = np.int_(round(np.random.chisquare(lt), 0))
        sales[i] = max(inventory[i], demand[i])
        if i == 0:
            orders[i] = max(psl - inventory[i], 0)
        else:
            orders[i] = max(psl - inventory[i] - on_order[i - 1], 0)
        if i + lead_times[i] <= days:
            receipts[i + lead_times[i]] = orders[i] + receipts[i + lead_times[i]]
        if i == 0:
            on_order[i] = orders[i] - receipts[i]
        else:
            on_order[i] = orders[i] + on_order[i - 1] - receipts[i]
        sales[i] = min(demand[i], inventory[i])
        if i < days:
            inventory[i + 1] = max(inventory[i] + receipts[i] - sales[i], 0)
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
        i += 1
    avg_inventory = np.mean(inventory[round(start_inv / ads * 2, 0):days])
    avg_ar = np.mean(ar[round(start_inv / ads * 2, 0):days])
    avg_ap = np.mean(ap[round(start_inv / ads * 2, 0):days])
    avg_sales = np.mean(sales[round(start_inv / ads * 2, 0):days])
    values = np.array([avg_sales, avg_inventory, avg_ar, avg_ap])
    return values


@njit
def day_sim_surge(psl, ads, var, lt, p_terms, s_terms, days, demand_sample, surge_demand_sample, surge_len,
                  surge_chance, surge_demand_mult, surge_lt_mult):
    start_inv = max(psl, 1)
    inventory = np.zeros(days, dtype=int64)
    lead_times = np.zeros(days, dtype=int64)
    orders = np.zeros(days, dtype=int64)
    receipts = np.zeros(days, dtype=int64)
    on_order = np.zeros(days, dtype=int64)
    sales = np.zeros(days, dtype=int64)
    surge = np.zeros(days, dtype=int64)
    ar = np.zeros(days, dtype=int64)
    ap = np.zeros(days, dtype=int64)
    if ads < var - 0.1:
        demand = demand_sample
        demand_surge = surge_demand_sample
    else:
        demand = np.zeros(days, dtype=int64)
        demand_surge = np.zeros(days, dtype=int64)
    inventory[0] = start_inv
    for i in range(0, days):
        if (i % surge_len == 0 and i >= surge_len * 2 and np.random.uniform(0.0, 1.0) <= surge_chance) \
                or (i % surge_len != 0 and surge[i - 1] == 1 and i >= surge_len * 2):
            surge[i] = 1
            if ads >= var - 0.1:
                demand_surge[i] = np.random.poisson(ads * surge_demand_mult)
            lead_times[i] = np.int_(round(np.random.chisquare(lt * surge_lt_mult), 0))
            sales[i] = max(inventory[i], demand_surge[i])
            if i == 0:
                orders[i] = max(psl - inventory[i], 0)
            else:
                orders[i] = max(psl - inventory[i] - on_order[i - 1], 0)
            if i + lead_times[i] <= days:
                receipts[i + lead_times[i]] = orders[i] + receipts[i + lead_times[i]]
            if i == 0:
                on_order[i] = orders[i] - receipts[i]
            else:
                on_order[i] = orders[i] + on_order[i - 1] - receipts[i]
            sales[i] = min(demand_surge[i], inventory[i])
            if i < days:
                inventory[i + 1] = max(inventory[i] + receipts[i] - sales[i], 0)
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
        else:
            surge[i] = 0
            if ads >= var - 0.1:
                demand[i] = np.random.poisson(ads)
            lead_times[i] = np.int_(round(np.random.chisquare(lt), 0))
            sales[i] = max(inventory[i], demand[i])
            if i == 0:
                orders[i] = max(psl - inventory[i], 0)
            else:
                orders[i] = max(psl - inventory[i] - on_order[i - 1], 0)
            if i + lead_times[i] <= days:
                receipts[i + lead_times[i]] = orders[i] + receipts[i + lead_times[i]]
            if i == 0:
                on_order[i] = orders[i] - receipts[i]
            else:
                on_order[i] = orders[i] + on_order[i - 1] - receipts[i]
            sales[i] = min(demand[i], inventory[i])
            if i < days:
                inventory[i + 1] = max(inventory[i] + receipts[i] - sales[i], 0)
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
        i += 1
    avg_inventory = np.mean(inventory[round(start_inv / ads * 2, 0):days])
    avg_ar = np.mean(ar[round(start_inv / ads * 2, 0):days])
    avg_ap = np.mean(ap[round(start_inv / ads * 2, 0):days])
    avg_sales = np.mean(sales[round(start_inv / ads * 2, 0):days])
    values = np.array([avg_sales, avg_inventory, avg_ar, avg_ap])
    return values


@njit
def calc_opti_psl(ads, var, lt, gm, cost, avg_sale_price, length, width, height, p_terms, s_terms):
    invst_charge = .14 / 365
    days = 100000
    psls_to_calc = int(max(1, ads) * max(1, lt) * 10)
    profit_arr = np.zeros(psls_to_calc, dtype='float64')
    inventory_arr = np.zeros(psls_to_calc, dtype='float64')
    sales_arr = np.zeros(psls_to_calc, dtype='float64')
    cube_arr = np.zeros(psls_to_calc, dtype='float64')
    ppc_arr = np.zeros(psls_to_calc, dtype='float64')
    if ads < var - 0.1:
        demand_array = calc_nb_array_ln(ads, var)
        demand_size = np.arange(demand_array.size)
        demand_sample = numba_choice(demand_size, demand_array, days)
    else:
        demand_sample = np.zeros(1, dtype=int64)
    for i in range(1, psls_to_calc):
        sim_result = day_sim(i, ads, var, lt, p_terms, s_terms, days, demand_sample)
        sales_arr[i] = sim_result[0]
        inventory_arr[i] = sim_result[1]
        avg_invest_acct = (sim_result[1] * cost) + (sim_result[2] * avg_sale_price) - (sim_result[3] * cost)
        profit_arr[i] = (sim_result[0] * gm) - avg_invest_acct * invst_charge
        cube_arr[i] = inventory_arr[i] * length * width * height
        ppc_arr[i] = profit_arr[i] / cube_arr[i]
        if i >= 20 and profit_arr[i - 5:i].mean() < profit_arr[i - 20:i - 3].mean():
            sales_arr = sales_arr[:i]
            inventory_arr = inventory_arr[:i]
            profit_arr = profit_arr[:i]
            cube_arr = cube_arr[:i]
            ppc_arr = ppc_arr[:i]
            break
        i += 1
    result = [np.argmax(profit_arr), profit_arr[np.argmax(profit_arr)], inventory_arr[np.argmax(profit_arr)],
              sales_arr[np.argmax(profit_arr)], cube_arr[np.argmax(profit_arr)], ppc_arr[np.argmax(profit_arr)]]
    return result


@njit
def calc_opti_psl_surge(ads, var, lt, gm, cost, avg_sale_price, length, width, height, p_terms, s_terms, surge_len,
                        surge_chance, surge_demand_mult, surge_lt_mult):
    invst_charge = .14 / 365
    days = 100000
    psls_to_calc = int(max(1, ads) * max(1, lt) * 10)
    profit_arr = np.zeros(psls_to_calc, dtype='float64')
    inventory_arr = np.zeros(psls_to_calc, dtype='float64')
    sales_arr = np.zeros(psls_to_calc, dtype='float64')
    cube_arr = np.zeros(psls_to_calc, dtype='float64')
    ppc_arr = np.zeros(psls_to_calc, dtype='float64')
    if ads < var - 0.1:
        demand_array = calc_nb_array_ln(ads, var)
        demand_size = np.arange(demand_array.size)
        demand_sample = numba_choice(demand_size, demand_array, days)
        demand_array_surge = calc_nb_array_ln(ads * surge_demand_mult, var ** surge_demand_mult)
        demand_size_surge = np.arange(demand_array_surge.size)
        demand_sample_surge = numba_choice(demand_size_surge, demand_array_surge, days)
    else:
        demand_sample = np.zeros(1, dtype=int64)
        demand_sample_surge = np.zeros(1, dtype=int64)
    for i in range(1, psls_to_calc):
        sim_result = day_sim_surge(i, ads, var, lt, p_terms, s_terms, days, demand_sample, demand_sample_surge,
                                   surge_len, surge_chance, surge_demand_mult, surge_lt_mult)
        sales_arr[i] = sim_result[0]
        inventory_arr[i] = sim_result[1]
        avg_invest_acct = (sim_result[1] * cost) + (sim_result[2] * avg_sale_price) - (sim_result[3] * cost)
        profit_arr[i] = (sim_result[0] * gm) - avg_invest_acct * invst_charge
        cube_arr[i] = inventory_arr[i] * length * width * height
        ppc_arr[i] = profit_arr[i] / cube_arr[i]
        if i >= 10 and profit_arr[i - 3:i].mean() < profit_arr[i - 10:i - 3].mean():
            sales_arr = sales_arr[:i]
            inventory_arr = inventory_arr[:i]
            profit_arr = profit_arr[:i]
            cube_arr = cube_arr[:i]
            ppc_arr = ppc_arr[:i]
            break
        i += 1
    result = [np.argmax(profit_arr), profit_arr[np.argmax(profit_arr)], inventory_arr[np.argmax(profit_arr)],
              sales_arr[np.argmax(profit_arr)], cube_arr[np.argmax(profit_arr)], ppc_arr[np.argmax(profit_arr)]]
    return result


@njit
def day_sim_2(psl, ads, var, lt, p_terms, s_terms, days, demand_sample, lt_sample):
    start_inv = max(psl, 1)
    inventory = np.zeros(days, dtype=int64)
    orders = np.zeros(days, dtype=int64)
    receipts = np.zeros(days, dtype=int64)
    on_order = np.zeros(days, dtype=int64)
    sales = np.zeros(days, dtype=int64)
    ar = np.zeros(days, dtype=int64)
    ap = np.zeros(days, dtype=int64)
    max_order_size = int64(round(ads * 2, 0))
    if ads < var - 0.1:
        demand = demand_sample
    else:
        demand = np.zeros(days, dtype=int64)
    lead_times = lt_sample
    inventory[0] = start_inv
    for i in range(0, days):
        if ads >= var - 0.1:
            demand[i] = np.random.poisson(ads)
        sales[i] = max(inventory[i], demand[i])
        if i == 0:
            orders[i] = min(max(psl - inventory[i], 0), max(max_order_size,1))
        else:
            orders[i] = min(max(psl - inventory[i] - on_order[i - 1], 0), max(max_order_size,1))
        if i + lead_times[i] <= days:
            receipts[i + lead_times[i]] = orders[i] + receipts[i + lead_times[i]]
        if i == 0:
            on_order[i] = orders[i] - receipts[i]
        else:
            on_order[i] = orders[i] + on_order[i - 1] - receipts[i]
        sales[i] = min(demand[i], inventory[i])
        if i < days:
            inventory[i + 1] = max(inventory[i] + receipts[i] - sales[i], 0)
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
        i += 1
    avg_inventory = np.mean(inventory[round(start_inv / ads * 2, 0):days])
    avg_ar = np.mean(ar[round(start_inv / ads * 2, 0):days])
    avg_ap = np.mean(ap[round(start_inv / ads * 2, 0):days])
    avg_sales = np.mean(sales[round(start_inv / ads * 2, 0):days])
    values = np.array([avg_sales, avg_inventory, avg_ar, avg_ap])
    return values


@njit
def calc_opti_psl_2(ads, var, lt, gm, cost, avg_sale_price, length, width, height, p_terms, s_terms, min_of_1):
    invst_charge = .14 / 365
    days = 100000
    lt_var_expon = 1.5
    p_terms = int64(p_terms)
    s_terms = int64(s_terms)
    min_of_1 = int64(min_of_1)
    psls_to_calc = int(max(1, ads) * max(1, lt) * 10)
    profit_arr = np.zeros(psls_to_calc, dtype='float64')
    inventory_arr = np.zeros(psls_to_calc, dtype='float64')
    sales_arr = np.zeros(psls_to_calc, dtype='float64')
    cube_arr = np.zeros(psls_to_calc, dtype='float64')
    ppc_arr = np.zeros(psls_to_calc, dtype='float64')
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
    if ads < var - 0.1:
        demand_array = calc_nb_array_ln(ads, var)
        demand_size = np.arange(demand_array.size)
        demand_sample = numba_choice(demand_size, demand_array, days)
    else:
        demand_sample = np.zeros(1, dtype=int64)
    for i in range(1, psls_to_calc):
        sim_result = day_sim_2(i, ads, var, lt, p_terms, s_terms, days, demand_sample, lt_sample)
        sales_arr[i] = sim_result[0]
        inventory_arr[i] = sim_result[1]
        avg_invest_acct = (sim_result[1] * cost) + (sim_result[2] * avg_sale_price) - (sim_result[3] * cost)
        profit_arr[i] = (sim_result[0] * gm) - avg_invest_acct * invst_charge
        cube_arr[i] = inventory_arr[i] * length * width * height
        if inventory_arr[i] >= 0:
            ppc_arr[i] = profit_arr[i] / cube_arr[i]
        else:
            ppc_arr[i] = 0
        #if i >= 20 and profit_arr[i] <= 0:
        if i >= 20 and profit_arr[i - 5:i].mean() < profit_arr[i - 20:i - 3].mean():
            sales_arr = sales_arr[:i]
            inventory_arr = inventory_arr[:i]
            profit_arr = profit_arr[:i]
            cube_arr = cube_arr[:i]
            ppc_arr = ppc_arr[:i]
            break
        i += 1
    if min_of_1 == 1:
        sales_arr = sales_arr[1:]
        inventory_arr = inventory_arr[1:]
        profit_arr = profit_arr[1:]
        cube_arr = cube_arr[1:]
        ppc_arr = ppc_arr[1:]
    # plan is to multiply profit arr by nearby values modified by normal distribution, then select that as highest profit PSL.
    #norm_pdf = normal_pdf(np.arange(-profit_arr.size, profit_arr.size), 0, m.sqrt((lt ** lt_var_expon) * var))
    #norm_pdf = np.round_(norm_pdf, 4, np.empty_like(norm_pdf))
    #norm_pdf = norm_pdf.ravel()[np.flatnonzero(norm_pdf)]
    #norm_pdf /= norm_pdf.sum()
    #normalized_profit = np.arange(profit_arr.size).astype(float64)
    #for n in range(0, profit_arr.size):
    #    normalized_profit[n] = apply_norm(normalized_profit[n], norm_pdf, profit_arr)
    #norm_result = np.asarray([np.argmax(normalized_profit), profit_arr[np.argmax(normalized_profit)],
    #                     inventory_arr[np.argmax(normalized_profit)],
    #                     sales_arr[np.argmax(normalized_profit)], cube_arr[np.argmax(normalized_profit)],
    #                     ppc_arr[np.argmax(normalized_profit)]])
    result = np.asarray([np.argmax(profit_arr), profit_arr[np.argmax(profit_arr)],
                         inventory_arr[np.argmax(profit_arr)],
                         sales_arr[np.argmax(profit_arr)], cube_arr[np.argmax(profit_arr)],
                         ppc_arr[np.argmax(profit_arr)]])
    if min_of_1 == 1:
        result[0] = result[0] + 1
    full_result = np.stack(
        (np.arange(profit_arr.size).astype(float64) - 1, profit_arr, inventory_arr, sales_arr, cube_arr,
         ppc_arr), axis=1)

    return result

#live
@njit
def calc_opti_psl_3(ads, var, lt, gm, cost, avg_sale_price, length, width, height, p_terms, s_terms, min_of_1):
    deval_days = 548
    invst_charge = .14 / 365
    overhead_pct_of_cogs = 0.047
    cost_per_cubic_ft = 0.005
    days = 100000
    lt_var_expon = 1.3
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
    if ads / var < 0.95 and var - ads > 0.1:
        demand_array = calc_nb_array_ln(ads, var)
        demand_size = np.arange(demand_array.size)
        demand_sample = numba_choice(demand_size, demand_array, days)
        demand_sample = demand_sample.astype(np.int64)
    else:
        demand_sample = np.zeros(1, dtype=np.int64)
    for i in range(1, psls_to_calc):
        sim_result = day_sim_3(i, ads, var, lt, p_terms, s_terms, days, demand_sample, lt_sample)
        sales_arr[i] = sim_result[0]
        inventory_arr[i] = sim_result[1]
        deval_arr[i] = sim_result[4]
        avg_invest_acct = (sim_result[1] * cost) + (sim_result[2] * avg_sale_price) - (sim_result[3] * cost)
        cube_arr[i] = inventory_arr[i] * length * width * height
        # previous formula
        profit_arr[i] = ((sim_result[0] * gm) - avg_invest_acct * invst_charge) - (deval_arr[i] * ((cost / 2)) / deval_days)
        # profit_arr[i] = ((sim_result[0] * gm) - avg_invest_acct * invst_charge) - (
        #             deval_arr[i] * ((cost / 2)) / deval_days) - (sim_result[0] * cost * overhead_pct_of_cogs) - (
        #                             cube_arr[i] * cost_per_cubic_ft)
        # the above adds in overhead as a % of COGS to reduce profitability (might want to reduce overhead_pct_of_cogs as PSL increases somehow)
        cube_arr[i] = inventory_arr[i] * length * width * height
        # if cube_arr[i] >= 0:
        if inventory_arr[i] >= 0:
            ppc_arr[i] = profit_arr[i] / cube_arr[i]
        else:
            ppc_arr[i] = 0
        if i >= 20 and profit_arr[i - 5:i].mean() < profit_arr[i - 20:i - 3].mean():
            sales_arr = sales_arr[:i]
            inventory_arr = inventory_arr[:i]
            profit_arr = profit_arr[:i]
            cube_arr = cube_arr[:i]
            ppc_arr = ppc_arr[:i]
            break
        i += 1
    if min_of_1 == 1:
        sales_arr = sales_arr[1:]
        inventory_arr = inventory_arr[1:]
        profit_arr = profit_arr[1:]
        cube_arr = cube_arr[1:]
        ppc_arr = ppc_arr[1:]
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

#testing
@njit
def calc_opti_psl_4(ads, var, lt, gm, cost, avg_sale_price, length, width, height, p_terms, s_terms, min_of_1):
    deval_days = 548
    invst_charge = .14 / 365
    overhead_cost_per_unit = 12.25
    cost_per_cubic_ft = 0.005
    days = 100000
    lt_var_expon = 1.3
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
    if ads / var < 0.95 and var - ads > 0.1:
        demand_array = calc_nb_array_ln(ads, var)
        demand_size = np.arange(demand_array.size)
        demand_sample = numba_choice(demand_size, demand_array, days)
    else:
        demand_sample = np.zeros(1, dtype=int64)
    for i in range(1, psls_to_calc):
        sim_result = day_sim_3(i, ads, var, lt, p_terms, s_terms, days, demand_sample, lt_sample)
        sales_arr[i] = sim_result[0]
        inventory_arr[i] = sim_result[1]
        deval_arr[i] = sim_result[4]
        avg_invest_acct = (sim_result[1] * cost) + (sim_result[2] * avg_sale_price) - (sim_result[3] * cost)
        cube_arr[i] = inventory_arr[i] * length * width * height
        # previous formula
        # profit_arr[i] = ((sim_result[0] * gm) - avg_invest_acct * invst_charge) - (deval_arr[i] * ((cost / 2)) / deval_days)
        # the below changed to add cost per unit as the overhead instead of as COGS.
        profit_arr[i] = (sim_result[0] * gm) - (avg_invest_acct * invst_charge) - (deval_arr[i] * ((cost / 2)) / deval_days) - (sim_result[0] * overhead_cost_per_unit) - (cube_arr[i] * cost_per_cubic_ft)
        # the below is only reducing the cubic impact
        # profit_arr[i] = ((sim_result[0] * gm) - avg_invest_acct * invst_charge) - (
        #             deval_arr[i] * ((cost / 2)) / deval_days) - (cube_arr[i] * cost_per_cubic_ft)
        if inventory_arr[i] >= 0:
            ppc_arr[i] = profit_arr[i] / cube_arr[i]
        else:
            ppc_arr[i] = 0
        if i >= 20 and profit_arr[i - 5:i].mean() < profit_arr[i - 20:i - 3].mean():
            sales_arr = sales_arr[:i]
            inventory_arr = inventory_arr[:i]
            profit_arr = profit_arr[:i]
            cube_arr = cube_arr[:i]
            ppc_arr = ppc_arr[:i]
            break
        i += 1
    if min_of_1 == 1:
        sales_arr = sales_arr[1:]
        inventory_arr = inventory_arr[1:]
        profit_arr = profit_arr[1:]
        cube_arr = cube_arr[1:]
        ppc_arr = ppc_arr[1:]
    result = np.asarray([np.argmax(profit_arr), profit_arr[np.argmax(profit_arr)],
                         inventory_arr[np.argmax(profit_arr)],
                         sales_arr[np.argmax(profit_arr)], cube_arr[np.argmax(profit_arr)],
                         ppc_arr[np.argmax(profit_arr)]])
    if min_of_1 == 1:
        result[0] = result[0] + 1
    full_result = np.stack(
        (np.arange(profit_arr.size).astype(float64) - 1, profit_arr, inventory_arr, sales_arr, cube_arr,
         ppc_arr), axis=1)

    return result


@njit
def calc_single_psl(psl, ads, var, lt, gm, cost, avg_sale_price, length, width, height, p_terms, s_terms, min_of_1):
    invst_charge = .14 / 365
    days = 100000
    deval_days = 548
    overhead_pct_of_cogs = 0.047
    cost_per_cubic_ft = 0.005
    lt_var_expon = 1.5
    p_terms = int64(p_terms)
    s_terms = int64(s_terms)
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
    if ads / var < 0.95 and var - ads > 0.1:
        demand_array = calc_nb_array_ln(ads, var)
        demand_size = np.arange(demand_array.size)
        demand_sample = numba_choice(demand_size, demand_array, days)
    else:
        demand_sample = np.zeros(1, dtype=int64)
    sim_result = day_sim_3(psl, ads, var, lt, p_terms, s_terms, days, demand_sample, lt_sample)
    sales = sim_result[0]
    inventory = sim_result[1]
    deval = sim_result[4]
    cube = inventory * length * width * height
    avg_invest_acct = (sim_result[1] * cost) + (sim_result[2] * avg_sale_price) - (sim_result[3] * cost)
    profit = ((sim_result[0] * gm) - avg_invest_acct * invst_charge) - (deval * (cost / 2))
    # profit = ((sim_result[0] * gm) - avg_invest_acct * invst_charge) - (
    #         deval * ((cost / 2)) / deval_days) - (sim_result[0] * cost * overhead_pct_of_cogs) - (
    #                         cube * cost_per_cubic_ft)
    if inventory >= 0:
        ppc = profit / cube
    else:
        ppc = 0
    result = np.array([float64(psl), profit, inventory, sales, cube, ppc])
    return result

#live
@njit
def day_sim_3(psl, ads, var, lt, p_terms, s_terms, days, demand_sample, lt_sample):
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
    if ads / var < 0.95 and var - ads > 0.1:
        demand = demand_sample
    else:
        demand = np.zeros(days, dtype=np.int64)
    lead_times = lt_sample
    inventory[0] = start_inv
    for i in range(0, days):
        if not (ads / var < 0.95 and var - ads > 0.1):
            demand[i] = np.int64(np.random.poisson(ads))
        if i == 0:
            orders[i] = min(max(psl - inventory[i], 0), max(max_order_size,1))
        else:
            orders[i] = min(max(psl - inventory[i] - on_order[i - 1], 0), max(max_order_size,1))
        if i + lead_times[i] < days:
            receipts[i + lead_times[i]] = orders[i] + receipts[i + lead_times[i]]
        if i == 0:
            on_order[i] = orders[i] - receipts[i]
        else:
            on_order[i] = orders[i] + on_order[i - 1] - receipts[i]
        sales[i] = min(demand[i], inventory[i])
        if i < days-1:
            inventory[i + 1] = max(inventory[i] + receipts[i] - sales[i], 0)
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
        if i >= deval_days:
            deval[i] = max(inventory[i - deval_days] - sales[i - deval_days:i].sum(),0)
        i += 1
    start_date = int(round(start_inv / ads * 2, 0))
    if start_date >= days:
        start_date = int(start_inv)
        # print(start_inv)
    avg_inventory = np.mean(inventory[start_date:days])
    avg_ar = np.mean(ar[start_date:days])
    avg_ap = np.mean(ap[start_date:days])
    avg_sales = np.mean(sales[start_date:days])
    avg_deval = np.mean(deval[start_date:days])
    values = np.array([avg_sales, avg_inventory, avg_ar, avg_ap, avg_deval])
    return values


@njit(parallel=True)
def get_all_psls(id_arr, ads_arr, var_arr, lt_arr, gm_arr, cost_arr, avg_sale_price_arr, length_arr, width_arr,
                 height_arr, pterms_arr, sterms_arr, min_of_1_arr):
    result_arr = np.zeros((id_arr.size, 6))
    run_on_row = (ads_arr > 0.0) & (lt_arr > 0.0) & (gm_arr > 0.0)
    for i in prange(id_arr.size):
        # print(id_arr[i])
        if run_on_row[i] == True:
            result_line = calc_opti_psl_3(ads_arr[i], var_arr[i], lt_arr[i], gm_arr[i], cost_arr[i],
                                          avg_sale_price_arr[i], length_arr[i], width_arr[i], height_arr[i],
                                          pterms_arr[i], sterms_arr[i], min_of_1_arr[i])
        elif min_of_1_arr[i] == 0:
            result_line = np.zeros(6)
        else:
            result_line = np.zeros(6)
            result_line[0] = 1
            result_line[2] = 1
        result_arr[i,:] = result_line
    return result_arr


@njit(parallel=True)
def get_all_psls_testing(id_arr, ads_arr, var_arr, lt_arr, gm_arr, cost_arr, avg_sale_price_arr, length_arr, width_arr,
                 height_arr, pterms_arr, sterms_arr, min_of_1_arr):
    result_arr = np.zeros((id_arr.size, 6))
    run_on_row = (ads_arr > 0.0) & (lt_arr > 0.0) & (gm_arr > 0.0)
    for i in prange(id_arr.size):
        if run_on_row[i] == True:
            result_line = calc_opti_psl_4(ads_arr[i], var_arr[i], lt_arr[i], gm_arr[i], cost_arr[i],
                                          avg_sale_price_arr[i], length_arr[i], width_arr[i], height_arr[i],
                                          pterms_arr[i], sterms_arr[i], min_of_1_arr[i])
        elif min_of_1_arr[i] == 0:
            result_line = np.zeros(6)
        else:
            result_line = np.zeros(6)
            result_line[0] = 1
            result_line[2] = 1
        result_arr[i,:] = result_line
    return result_arr


@njit(parallel=True)
def get_current_estimates(id_arr, ads_arr, var_arr, lt_arr, gm_arr, cost_arr, avg_sale_price_arr, length_arr, width_arr,
                 height_arr, pterms_arr, sterms_arr, min_of_1_arr, current_psl_arr):
    result_arr = np.zeros((id_arr.size, 6))
    run_on_row = (ads_arr > 0.0) & (lt_arr > 0.0) & (gm_arr > 0.0)
    for i in prange(id_arr.size):
        if run_on_row[i] == True:
            result_line = calc_single_psl(current_psl_arr[i], ads_arr[i], var_arr[i], lt_arr[i], gm_arr[i], cost_arr[i],
                                          avg_sale_price_arr[i], length_arr[i], width_arr[i], height_arr[i],
                                          pterms_arr[i], sterms_arr[i], min_of_1_arr[i])
        elif min_of_1_arr[i] == 0:
            result_line = np.zeros(6)
        else:
            result_line = np.zeros(6)
            result_line[0] = 1
            result_line[2] = 1
        result_arr[i,:] = result_line
    return result_arr


@njit
def historical_day_sim(days, demand_sample, lt_sample, start_inv):
    inventory = np.zeros(days, dtype=int64)
    orders = np.zeros(days, dtype=int64)
    receipts = np.zeros(days, dtype=int64)
    on_order = np.zeros(days, dtype=int64)
    sales = np.zeros(days, dtype=int64)
    psl = np.zeros(days, dtype=int64)
    demand = demand_sample
    lead_times = lt_sample.astype(int64)
    inventory[0] = start_inv
    for i in range(0, days):
        if i > 61:
            psl[i] = np.sum(sales[i-61:i-1])
        else:
            psl[i] = start_inv
        sales[i] = max(inventory[i], demand[i])
        if i == 0:
            orders[i] = max(psl[i] - inventory[i], 0)
        else:
            orders[i] = max(psl[i] - inventory[i] - on_order[i - 1], 0)
        if i + lead_times[i] <= days:
            receipts[i + lead_times[i]] = orders[i] + receipts[i + lead_times[i]]
        if i == 0:
            on_order[i] = orders[i] - receipts[i]
        else:
            on_order[i] = orders[i] + on_order[i - 1] - receipts[i]
        sales[i] = min(demand[i], inventory[i])
        if i < days:
            inventory[i + 1] = max(inventory[i] + receipts[i] - sales[i], 0)
        i += 1
    avg_inventory = np.mean(inventory)
    avg_sales = np.mean(sales)
    values = np.vstack((sales, inventory))
    return values


# plan is to calculate what Profit Sharing for the company would be had we stuck to 8wk Moving Average for PSLs
#just saving this code in case need to run again
def run_hist_sim():
    hist_demand_lts = pd.read_parquet(r"C:\Users\benso\PycharmProjects\PSL_Calculation_v4\hist_demand_lts.pq")
    hist_demand_lts = hist_demand_lts.reset_index()
    hist_demand_lts = hist_demand_lts[hist_demand_lts['Date'] >= '2020-05-01']
    hist_demand_lts = hist_demand_lts.set_index(['Internal ID', 'Date'])
    starting_inventory = pd.read_csv(
        r"C:\Users\benso\PycharmProjects\PSL_Calculation_v4\Hist Analysis Starting Inventory.csv")
    starting_inventory = starting_inventory.set_index('Internal ID')
    hist_demand_lts = hist_demand_lts.join(starting_inventory)
    hist_demand_lts['Starting Inventory'] = hist_demand_lts['Starting Inventory'].fillna(0)
    hist_demand_lts['Lead Time'] = hist_demand_lts['Lead Time'].bfill()
    # hist_demand_lts_filt = hist_demand_lts.loc[list_filter]
    # demand_in = np.asarray(hist_demand_lts_filt['Quantity On Date'])
    # lts_in = np.asarray(hist_demand_lts_filt['Lead Time'])
    # start_inv = np.asarray(hist_demand_lts_filt['Starting Inventory'])
    # result = sim.historical_day_sim(demand_in.size, demand_in, lts_in, start_inv[0])
    # hist_demand_lts_filt['Inventory'] = result[1,:]
    # hist_demand_lts_filt['Sales'] = result[2, :]
    # print(hist_demand_lts_filt)
    pd.options.mode.chained_assignment = None
    hist_demand_lts = hist_demand_lts.reset_index()
    item_list = np.unique(hist_demand_lts['Internal ID'])
    print(item_list.size)
    hist_demand_lts = hist_demand_lts.set_index(['Internal ID', 'Date'])
    hist_demand_lts_filt = pd.DataFrame()
    result_sum = 0
    for i in range(0, item_list.size):
        hist_demand_lts_filt = hist_demand_lts.loc[item_list[i]]
        demand_in = np.asarray(hist_demand_lts_filt['Quantity On Date'])
        lts_in = np.asarray(hist_demand_lts_filt['Lead Time'])
        start_inv = np.asarray(hist_demand_lts_filt['Starting Inventory'])
        result = historical_day_sim(demand_in.size, demand_in, lts_in, start_inv[0])
        hist_demand_lts_filt['Inventory'] = result[1, :]
        hist_demand_lts_filt['Sales'] = result[0, :]
        hist_demand_lts_filt['Internal ID'] = item_list[i]
        hist_demand_lts_filt = hist_demand_lts_filt.reset_index()
        # resultDF = pd.concat((resultDF, hist_demand_lts_filt), axis=0)
        hist_demand_lts_filt['Sales $'] = hist_demand_lts_filt['Rate'] * hist_demand_lts_filt['Sales']
        hist_demand_lts_filt['Inventory $'] = hist_demand_lts_filt['Rate'] * 0.8 * hist_demand_lts_filt['Inventory']
        hist_demand_lts_filt_2021 = hist_demand_lts_filt[
            (hist_demand_lts_filt['Date'] <= '2021-12-31') & (hist_demand_lts_filt['Date'] >= '2021-01-01')]
        result_sum += hist_demand_lts_filt_2021['Sales $'].sum()
        print(round(i / item_list.size * 100, 1))
    print(result_sum)
    # hist_demand_lts.to_parquet(r"C:\Users\benso\PycharmProjects\PSL_Calculation_v4\hist_demand_lts.pq")


@njit
def day_sim_example(psl, ads, var, lt, p_terms, s_terms, days, demand_sample, lt_sample):
    deval_days = 548
    start_inv = max(psl, 1)
    inventory = np.zeros(days, dtype=int64)
    orders = np.zeros(days, dtype=int64)
    receipts = np.zeros(days, dtype=int64)
    on_order = np.zeros(days, dtype=int64)
    sales = np.zeros(days, dtype=int64)
    ar = np.zeros(days, dtype=int64)
    ap = np.zeros(days, dtype=int64)
    deval = np.zeros(days, dtype=int64)
    max_order_size = int64(round(ads * 2, 0))
    if ads < var - 0.1:
        demand = demand_sample
    else:
        demand = np.zeros(days, dtype=int64)
    lead_times = lt_sample
    inventory[0] = start_inv
    for i in range(0, days):
        if ads >= var - 0.1:
            demand[i] = np.random.poisson(ads)
        if i == 0:
            orders[i] = min(max(psl - inventory[i], 0), max(max_order_size,1))
        else:
            orders[i] = min(max(psl - inventory[i] - on_order[i - 1], 0), max(max_order_size,1))
        if i + lead_times[i] <= days:
            receipts[i + lead_times[i]] = orders[i] + receipts[i + lead_times[i]]
        if i == 0:
            on_order[i] = orders[i] - receipts[i]
        else:
            on_order[i] = orders[i] + on_order[i - 1] - receipts[i]
        sales[i] = min(demand[i], inventory[i])
        if i < days:
            inventory[i + 1] = max(inventory[i] + receipts[i] - sales[i], 0)
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
        if i >= deval_days:
            deval[i] = max(inventory[i - deval_days] - sales[i - deval_days:i].sum(),0)
        i += 1
    avg_inventory = np.mean(inventory[round(start_inv / ads * 2, 0):days])
    avg_ar = np.mean(ar[round(start_inv / ads * 2, 0):days])
    avg_ap = np.mean(ap[round(start_inv / ads * 2, 0):days])
    avg_sales = np.mean(sales[round(start_inv / ads * 2, 0):days])
    avg_deval = np.mean(deval[round(start_inv / ads * 2, 0):days])
    values = np.array([avg_sales, avg_inventory, avg_ar, avg_ap, avg_deval])
    full_sim = np.stack((sales, inventory, lead_times, demand, orders, receipts, on_order, ar, ap, deval), axis=1)
    return full_sim


@njit
def sim_export(psl, ads, var, lt, gm, cost, avg_sale_price, p_terms, s_terms):
    deval_days = 548
    invst_charge = .14 / 365
    days = 100000
    lt_var_expon = 1.5
    p_terms = int64(p_terms)
    s_terms = int64(s_terms)
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
    if ads < var - 0.1:
        demand_array = calc_nb_array_ln(ads, var)
        demand_size = np.arange(demand_array.size)
        demand_sample = numba_choice(demand_size, demand_array, days)
    else:
        demand_sample = np.zeros(1, dtype=int64)
    sim_result = day_sim_example(psl, ads, var, lt, p_terms, s_terms, days, demand_sample, lt_sample)
    deval = sim_result[4,:]
    invest_acct = (sim_result[1,:] * cost) + (sim_result[2,:] * avg_sale_price) - (sim_result[3,:] * cost)
    profit = ((sim_result[0,:] * gm) - invest_acct * invst_charge) - (deval * ((cost / 2)) / deval_days)
    #result = np.concatenate((sim_result.astype(float64), profit, invest_acct, deval.astype(float64)))
    return sim_result


@njit
def big_buy_day_sim(ads, var, p_terms, s_terms, days, demand_sample, start_inv, cost, gm, price, additional_buy, additional_buy_offset, rebate, days_to_monopoly, monopoly_ads_mult, current_inventory):
    deval_days = 548
    invst_charge = .14 / 365
    inventory = np.zeros(days, dtype=int64)
    receipts = np.zeros(days, dtype=int64)
    sales = np.zeros(days, dtype=int64)
    ar = np.zeros(days, dtype=int64)
    ap = np.zeros(days, dtype=int64)
    deval = np.zeros(days, dtype=int64)
    if ads / var < 0.95 and var - ads > 0.1:
        demand = demand_sample
    else:
        demand = np.zeros(days, dtype=int64)
    inventory[0] = current_inventory
    receipts[0] = start_inv
    receipts[additional_buy_offset] = additional_buy + receipts[additional_buy_offset]
    for i in range(0, days):
        if not (ads / var < 0.95 and var - ads > 0.1):
            demand[i] = np.random.poisson(ads)
            if i >= days_to_monopoly:
                demand[i] = np.random.poisson(ads * monopoly_ads_mult)
        sales[i] = min(demand[i], inventory[i])
        if i < days:
            inventory[i + 1] = max(inventory[i] + receipts[i] - sales[i], 0)
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
        if i >= deval_days:
            deval[i] = max(inventory[i - deval_days] - sales[i - deval_days:i].sum(),0)
        i += 1
    invest_acct = (inventory * cost) + (ar * price) - (ap * cost)
    prev_day_deval = deval[1:days]
    prev_day_deval = np.hstack((np.asarray([0]), prev_day_deval))
    profit = (sales * (gm + rebate)) - (invest_acct * invst_charge) - ((deval - prev_day_deval) * ((cost / 2)))
    result = np.sum(profit)
    days_to_zero_inventory = np.where(inventory[1:] == 0)[0][0]
    return result, days_to_zero_inventory


@njit
def calc_big_buy(ads_range, var_range, price_range, cost_range, p_terms_range, s_terms_range, start_inv_range,
                 additional_buy_range, additional_buy_offset_range, rebate_range, days_to_monopoly_range,
                 monopoly_ads_mult_range, item_number_range, current_inventory_range):
    num_sims = 100
    total_size = int64((ads_range.size * var_range.size * price_range.size * cost_range.size
                  * p_terms_range.size * s_terms_range.size * start_inv_range.size
                  * additional_buy_range.size * additional_buy_offset_range.size * rebate_range.size
                  * days_to_monopoly_range.size * monopoly_ads_mult_range.size)) + 1
    print(total_size)
    result = np.zeros((total_size, 16), dtype=float64)  # create an empty array with 16 columns for the output values
    row = 0  # initialize the row counter
    for i in range(0, start_inv_range.size):
        start_inv = start_inv_range[i]
        for b in range(0, additional_buy_range.size):
            additional_buy = additional_buy_range[b]
            for o in range(0, additional_buy_offset_range.size):
                additional_buy_offset = additional_buy_offset_range[o]
                for a in range(0, ads_range.size):
                    ads = ads_range[a]
                    var = var_range[a]
                    days = max((((start_inv + additional_buy) / ads) * 2) + p_terms_range.max() + 180, p_terms_range.max() + 180)
                    days = int64(days)
                    for m in range(0, days_to_monopoly_range.size):
                        days_to_monopoly = days_to_monopoly_range[m]
                        for mm in range(0, monopoly_ads_mult_range.size):
                            monopoly_ads_mult = monopoly_ads_mult_range[mm]
                            demand_sample_array = np.zeros((num_sims, days), dtype=int64)
                            for d in range(0, num_sims):
                                if ads / var < 0.95 and var - ads > 0.1:
                                    demand_array = calc_nb_array_ln(ads, var)
                                    ads_monopoly = ads * monopoly_ads_mult
                                    var_monoploy = var*monopoly_ads_mult*4
                                    mono_demand_array = calc_nb_array_ln(ads_monopoly, var_monoploy)
                                    demand_size = np.arange(demand_array.size)
                                    mono_demand_size = np.arange(mono_demand_array.size)
                                    demand_sample_pre = numba_choice(demand_size, demand_array, days_to_monopoly)
                                    demand_sample_post = numba_choice(mono_demand_size, mono_demand_array, days - days_to_monopoly)
                                    # print(demand_sample_pre)
                                    # print(demand_sample_post)
                                    # demand_sample_pre = demand_sample[:days_to_monopoly]
                                    # demand_sample_post = demand_sample[days_to_monopoly:]
                                    # demand_sample_post = demand_sample_post.astype(float64) * monopoly_ads_mult
                                    # demand_sample_post = demand_sample_post.astype(int64)
                                    demand_sample = np.hstack((demand_sample_pre, demand_sample_post))

                                else:
                                    demand_sample = np.zeros(days, dtype=int64)
                                demand_sample_array[d] = demand_sample
                            for p in range(0, price_range.size):
                                price = price_range[p]
                                for c in range(0, cost_range.size):
                                    cost = cost_range[c]
                                    for r in range(0, rebate_range.size):
                                        rebate = rebate_range[r]
                                        gm = price - cost
                                        for pt in range(0, p_terms_range.size):
                                            p_terms = p_terms_range[pt]
                                            for s in range(0, s_terms_range.size):
                                                s_terms = s_terms_range[s]
                                                # call the big_buy_day_sim function and append the output values to the result array
                                                current_inventory = current_inventory_range[0]
                                                profit = np.zeros(num_sims, dtype=float64)
                                                days_to_0_inventory = np.zeros(num_sims, dtype=int64)
                                                for j in range(0, num_sims):
                                                    demand_sample_in = demand_sample_array[j]
                                                    profit[j], days_to_0_inventory[j] = big_buy_day_sim(ads, var, p_terms, s_terms, days, demand_sample_in, start_inv, cost, gm, price, additional_buy, additional_buy_offset, rebate, days_to_monopoly, monopoly_ads_mult, current_inventory)
                                                # jank way to get tigris current inventory value out of profit
                                                profit_avg = np.mean(profit) - (current_inventory * (gm - rebate) * (cost / (cost_range.max() / 0.95)))
                                                days_to_0_inventory_avg = np.maximum(np.mean(days_to_0_inventory), 1)
                                                result[row] = np.array([item_number_range[0], start_inv, additional_buy, additional_buy_offset, ads, var, price, cost, gm, p_terms, s_terms, rebate, days_to_monopoly, monopoly_ads_mult, profit_avg, days_to_0_inventory_avg])
                                                row += 1  # increment the row counter
    return result
