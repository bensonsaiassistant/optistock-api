import numpy as np
import pandas as pd
from numba import njit, prange
import math as m
import matplotlib.pyplot as plt
import time
import random
from numpy import int64, float64
import sys

np.set_printoptions(suppress=True)


@njit
def neg_bin_ln(qty, r, p):
    log_result = m.lgamma(qty + r) - m.lgamma(qty + 1) - m.lgamma(r) + (r * m.log(p)) + (qty * m.log(1 - p))
    return log_result


@njit
def poisson_pmf(k, lambd):
    log_result = k * m.log(lambd) - lambd - m.lgamma(k + 1)
    return log_result


@njit(fastmath=True, parallel=False)
def logsum_exp_reduceat(arr, indices):
    res = np.empty(indices.shape[0], dtype=arr.dtype)
    for i in prange(indices.shape[0] - 1):
        r = 0.
        for j in range(indices[i], indices[i + 1]):
            r += np.exp(arr[j])
        res[i] = np.log(r)
    r = 0.
    for j in range(indices[-1], arr.shape[0]):
        r += np.exp(arr[j])
    res[-1] = np.log(r)
    return res


@njit
def calc_nb_array_ln(mean, var):
    nbarray = np.zeros(2000, dtype=np.float64)
    if mean == 0:
        return nbarray
    if mean >= var:  # Compute Poisson PMF if mean gte var
        for i in range(nbarray.size):
            nbarray[i] = poisson_pmf(i, mean)
            if np.exp(nbarray[i]) < 1e-50 and i > mean:
                nbarray = nbarray[:i]
                break
    else:  # Compute Negative Binomial distribution
        r = -(mean * var) / (mean - var)
        p = r / (r + mean)
        for i in range(0, nbarray.size):
            nbarray[i] = neg_bin_ln(i, r, p)
            if np.exp(nbarray[i]) < 1e-20 and i > mean:
                nbarray = nbarray[:i]
                break

    if mean > 50:
        # Convert log-probabilities back to probabilities using logsum_exp_reduceat
        nbarray = np.exp(nbarray - logsum_exp_reduceat(nbarray, np.array([0])))
    else:
        # Convert log-probabilities back to probabilities using direct exp
        nbarray = np.exp(nbarray)
        nbarray /= np.sum(nbarray)
    return nbarray


# TODO: Need to scale with size what part of the diff array is most important depending on ads.
#  When ADS is low, upper quantiles are most important
#  Current implementation struggles with ADS around 0.01 to 0.1 or so
#  When ADS is high, mid quantiles are the most important
@njit
def calc_loss(mean, var, neural_quantiles, quantile_targets):
    if mean > var:
        var = mean
    nb_arr = calc_nb_array_ln(mean, var)
    nb_quantiles = find_cdf_indexes(nb_arr, quantile_targets)

    diff = (neural_quantiles - nb_quantiles)
    loss = (diff ** 2).sum()

    return loss


@njit
def loss_func(x, obs_quantiles, quantile_targets):
    return calc_loss(x[0], x[1], obs_quantiles, quantile_targets)


@njit
def gradient_descent(loss_func, x_init, obs_quantiles, quantile_targets, lr=0.00001, num_iter=10000, perturbation = 0.001):
    x = x_init.copy()


    for _ in range(num_iter):
        perturbed_x = x + np.eye(len(x)) * perturbation  # Create a matrix of perturbed x values
        losses_perturbed = np.array([loss_func(x_val, obs_quantiles, quantile_targets) for x_val in perturbed_x])
        loss_original = loss_func(x, obs_quantiles, quantile_targets)

        grad = (losses_perturbed - loss_original) / perturbation

        x_temp = x - lr * grad

        if x_temp[0] >= 0:
            x[0] = x_temp[0]

        if x_temp[1] >= x[0]:
            x[1] = x_temp[1]

    if x[1] < x[0]:
        x[1] = x[0]

    return x


@njit
def nelder_mead(loss_func, x_init, obs_quantiles, quantile_targets, alpha=1, gamma=2, rho=0.5, sigma=0.5, maxiter=10000,
                tol=1e-100):
    n = len(x_init)
    simplex = np.zeros((n + 1, n))

    step_size = 0.000001  # You can adjust this value as needed to make the simplex tighter or looser

    for i in range(n + 1):
        simplex[i] = x_init
        if i < n:
            simplex[i, i] += step_size

    func_vals = np.zeros(n + 1)
    for i in range(n + 1):
        func_vals[i] = loss_func(simplex[i], obs_quantiles, quantile_targets)

    for _ in range(maxiter):
        order = func_vals.argsort()
        simplex = simplex[order]
        func_vals = func_vals[order]

        centroid = np.sum(simplex[:-1], axis=0) / n
        reflected = centroid + alpha * (centroid - simplex[-1])

        # Apply constraints
        reflected[0] = max(reflected[0], 0)  # Ensure x[0] is always not negative
        if reflected[1] < reflected[0]:
            reflected[1] = reflected[0]

        f_reflected = loss_func(reflected, obs_quantiles, quantile_targets)

        if f_reflected < func_vals[0]:
            expanded = centroid + gamma * (reflected - centroid)

            # Apply constraints
            expanded[0] = max(expanded[0], 0)  # Ensure x[0] is always not negative
            if expanded[1] < expanded[0]:
                expanded[1] = expanded[0]

            f_expanded = loss_func(expanded, obs_quantiles, quantile_targets)

            if f_expanded < func_vals[0]:
                simplex[-1] = expanded
                func_vals[-1] = f_expanded
            else:
                simplex[-1] = reflected
                func_vals[-1] = f_reflected

        elif f_reflected < func_vals[-2]:
            simplex[-1] = reflected
            func_vals[-1] = f_reflected

        else:
            contracted = centroid + rho * (simplex[-1] - centroid)

            # Apply constraints
            contracted[0] = max(contracted[0], 0)  # Ensure x[0] is always not negative
            if contracted[1] < contracted[0]:
                contracted[1] = contracted[0]

            f_contracted = loss_func(contracted, obs_quantiles, quantile_targets)

            if f_contracted < func_vals[-1]:
                simplex[-1] = contracted
                func_vals[-1] = f_contracted

            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    # Apply constraints during shrink
                    simplex[i, 0] = max(simplex[i, 0], 0)  # Ensure x[0] is always not negative
                    if simplex[i, 1] < simplex[i, 0]:
                        simplex[i, 1] = simplex[i, 0]
                    func_vals[i] = loss_func(simplex[i], obs_quantiles, quantile_targets)

        if np.abs(func_vals[0] - func_vals[-1]) < tol:
            break

    return simplex[0]


@njit()
def pso(loss_func, x_init, obs_quantiles, quantile_targets, n_particles=50, maxiter=500, w=1, c1=1.5, c2=1.5, tol=1e-15):
    n = len(x_init)

    # Initialize particle positions and velocities
    particles = np.random.uniform(low=0, high=2, size=(n_particles, n)) + x_init
    velocities = np.random.uniform(low=-0.1, high=0.1, size=(n_particles, n))

    personal_best_positions = np.copy(particles)
    personal_best_scores = np.array([loss_func(pos, obs_quantiles, quantile_targets) for pos in particles])

    # Find global best position
    global_best_position = particles[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)

    for _ in range(maxiter):
        for i in range(n_particles):
            # Update velocities
            inertia = w * np.random.uniform(low=0.5, high=1.0) * velocities[i]
            cognitive = c1 * np.random.random() * (personal_best_positions[i] - particles[i])
            social = c2 * np.random.random() * (global_best_position - particles[i])
            velocities[i] = inertia + cognitive + social

            # Update particle positions
            particles[i] += velocities[i]

            # Apply constraints
            particles[i, 0] = max(particles[i, 0], 0)
            if particles[i, 1] < particles[i, 0]:
                particles[i, 1] = particles[i, 0]

            # Update personal best
            current_score = loss_func(particles[i], obs_quantiles, quantile_targets)
            if current_score < personal_best_scores[i]:
                personal_best_scores[i] = current_score
                personal_best_positions[i] = particles[i]

                # Update global best
                if current_score < global_best_score:
                    global_best_score = current_score
                    global_best_position = particles[i]
        if global_best_score < tol:
            break
    return global_best_position


@njit
def min_loss(obs_quantiles, quantile_targets):
    x0 = np.asarray([max(0.25, obs_quantiles[3]), max(obs_quantiles[3]* 2, 0.5, max(0.25, obs_quantiles[3]))])
    res = pso(loss_func, x0, obs_quantiles, quantile_targets)
    return res


@njit
def find_cdf_indexes(pdf, x):
    # Calculate the cumulative sum of the pdf
    cdf = np.cumsum(pdf)

    # Normalize to 1
    cdf /= cdf[-1]

    # Pre-allocate an output array of the same size as x
    result = np.empty_like(x, dtype=np.float64)

    # Loop through each element of x
    for i in range(x.shape[0]):
        # Find the index where the cdf is just greater than x[i]
        index = np.searchsorted(cdf, x[i])
        if index == 0:
            result[i] = 0.0
        elif index == len(pdf):
            result[i] = float(index-1)
        else:
            # Linear interpolation between the two closest indices
            left_cdf = cdf[index-1]
            right_cdf = cdf[index]
            interp_index = index - 1 + (x[i] - left_cdf) / (right_cdf - left_cdf)
            result[i] = interp_index

    return result


@njit
def calc_mean_var_estimates(mean, var):

    # output quantiles from neural network
    # nn_output = np.array([0.0, 3, 3, 5, 9, 15, 45]) # for 1/730 example output
    # nn_output = np.array([0.1, 1.2, 4.5, 5.2, 5.5, 6.0, 7.2, 10.1, 12.3])
    nn_output = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 2, 6, 10])

    # quantile probabilities
    quantile_targets = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 0.99999])
    # quantile_targets = np.array([0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])

    nn_output = calc_nb_array_ln(mean, var)

    quantiles = find_cdf_indexes(nn_output, quantile_targets)
    # print(quantiles)

    # calculate outputs
    result_min_loss = min_loss(quantiles, quantile_targets)

    # print(result_min_loss[0], result_min_loss[1])
    #
    # # generate negative binomial array based on results
    # min_loss_nb_array = calc_nb_array_ln(result_min_loss[0], result_min_loss[1])
    #
    # # convert to cumulative distributions
    # min_loss_nb_array_cumulative = np.cumsum(min_loss_nb_array)
    #
    # # plot original neural_quantiles as points
    # plt.scatter(quantiles, quantile_targets, label='Original')
    # #
    # # # plot min_loss output
    # plt.plot(min_loss_nb_array_cumulative, label='Min Loss')
    # #
    # # # add a legend
    # plt.legend()
    # #
    # # # show the plot
    # plt.show()
    return result_min_loss[0]

quantile_targets = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 0.99999])
quantile_targets_lt = np.array([0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])

@njit(parallel=True)
def parallel_run():
    runs = 10000
    res = np.zeros((runs, 3), dtype=float)
    for i in prange(runs):
        ads = np.random.uniform(0, 100)
        var = np.random.uniform(1, 10) * ads
        res[i, 0] = calc_mean_var_estimates(ads, var)
        res[i, 1] = ads
        res[i, 2] = var
    return res


@njit
def day_sim(psl, ads, var, lt, p_terms, s_terms, days, demand_sample, lt_sample):
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
    if ads / var < 0.95 and var - ads > 0.1:
        demand = demand_sample
    else:
        demand = np.zeros(days, dtype=int64)
    lead_times = lt_sample
    inventory[0] = start_inv
    for i in range(0, days):
        if not (ads / var < 0.95 and var - ads > 0.1):
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
    return values


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
def calc_opti_psl(ads, var, lt, gm, cost, avg_sale_price, length, width, height, p_terms, s_terms, min_of_1):
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
        sim_result = day_sim(i, ads, var, lt, p_terms, s_terms, days, demand_sample, lt_sample)
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


# old
@njit(parallel=True)
def get_all_purch_qtys(id_arr, ads_arr, var_arr, lt_arr, lt_var_arr, gm_arr, cost_arr, sale_price_arr, length_arr, width_arr,
                 height_arr, pterms_arr, sterms_arr, next_purch_days_away_arr, start_inv_arr):
    result_arr = np.zeros((id_arr.size, 6))
    run_on_row = (ads_arr > 0.0) & (lt_arr > 0.0) & (gm_arr > 0.0)
    for i in prange(id_arr.size):
        if run_on_row[i] == True:
            result_line = calc_opti_psl(ads_arr[i], var_arr[i], lt_arr[i], lt_var_arr[i], gm_arr[i], cost_arr[i],
                                          sale_price_arr[i], length_arr[i], width_arr[i], height_arr[i],
                                          pterms_arr[i], sterms_arr[i], next_purch_days_away_arr[i], start_inv_arr[i])
        else:
            result_line = np.zeros(6)
            result_line[0] = 1
            result_line[2] = 1
        result_arr[i,:] = result_line
    return result_arr


# @njit
# def purch_sim_orig(purch_today, purch_next_op, next_purch_days_away, p_terms, s_terms, demand_sample, lt_today, lt_next_op, start_inv, on_order, cost, gm, sale_price, length, width, height, days, cancel_qty):
#     # TODO:
#     #   mercury will need:
#     #   Merc quantity available and prices
#     #   If merc price is lower than tigris, use merc inventory first
#     #   When we use merc inventory, gm becomes gm - (merc_price - tigris_cost)
#
#     # TODO:
#     #   Rebate + payment terms valuation when purchasing an item
#     #   Purchasing terms value is realized p_terms days away from the purchase
#     #   Rebate is realized when the item is received? (could also just tack it on at the end) only tigris purchases count
#     #   Dont need to track the above with arrays, can probably just tack on
#
#     overhead_cost_per_unit = 12.25
#     cost_per_cubic_ft = 0.005
#     deval_days = days
#     inventory = np.zeros(deval_days, dtype=int64)
#     sales = np.zeros(deval_days, dtype=int64)
#     purch_inventory_sales = np.zeros(deval_days, dtype=int64)
#     cum_sales = 0
#     receipts = np.zeros(deval_days, dtype=int64)
#     ar = np.zeros(deval_days, dtype=int64)
#     ap = np.zeros(deval_days, dtype=int64)
#     mercury_sales = np.zeros(deval_days, dtype=int64)
#     demand = demand_sample
#     inventory[0] = start_inv
#     left_on_order = on_order - cancel_qty
#     already_on_order_daily_receipts = int64(max(left_on_order // lt_today, left_on_order))
#     j = 1
#     while left_on_order > 0:
#         receipts[j] = min(already_on_order_daily_receipts, left_on_order)
#         left_on_order = left_on_order - receipts[j]
#         j += 1
#     receipts[lt_today] += purch_today
#     receipts[lt_next_op + next_purch_days_away] += purch_next_op
#     for i in range(0, deval_days):
#         # TODO: can have sales if we're out of stock if mercury is in stock!
#         sales[i] = min(demand[i], inventory[i])
#         cum_sales += sales[i]
#         gap = cum_sales - (start_inv + left_on_order)
#         if cum_sales > (start_inv + left_on_order):
#             purch_inventory_sales[i] = min(sales[i], gap)
#         if i < deval_days:
#             # TODO: If we sell merc inventory instead, then inventory does not go down by sales[i]
#             inventory[i + 1] = max(inventory[i] + receipts[i] - sales[i], 0)
#         if i == 0:
#             ar[i] = purch_inventory_sales[i]
#             ap[i] = receipts[i]
#         if i < s_terms and s_terms > 0:
#             ar[i] = purch_inventory_sales[i] + ar[i - 1]
#         if i >= s_terms and s_terms > 0:
#             ar[i] = purch_inventory_sales[i] + ar[i - 1] - purch_inventory_sales[i - s_terms]
#         if i < p_terms and p_terms > 0:
#             ap[i] = receipts[i] + ap[i - 1]
#         if i >= p_terms and p_terms > 0:
#             ap[i] = receipts[i] + ap[i - 1] - receipts[i - p_terms]
#         if inventory[i] + ar[i] + ap[i] + np.sum(receipts[i:]) == 0:
#             break
#         i += 1
#     deval = inventory[-1] * cost
#     # TODO: This needs to change with mercury because GM is not always the same
#     revenue = sales.sum() * gm
#     investment_charge_account = (inventory * cost) + (ar * sale_price) + (ap * cost)
#     investment_charge = (investment_charge_account * (0.14/365)).sum()
#     profit = revenue - investment_charge - deval
#
#     # adding warehouse cost and overhead:
#     # warehouse_cost = (inventory * length * width * height * cost_per_cubic_ft).sum()
#     # overhead_cost = sales.sum() * overhead_cost_per_unit
#     # profit = revenue - investment_charge - deval - warehouse_cost - overhead_cost
#
#     exp_inventory = inventory[lt_today + 1]
#
#     return profit, exp_inventory
#
#
# @njit
# def purch_sim_merc(purch_today, purch_next_op, next_purch_days_away, p_terms, s_terms, demand_sample, lt_today, lt_next_op, start_inv, on_order, cost, gm, sale_price, length, width, height, days, cancel_qty, merc_prices, value_on_purchase, cop):
#     # TODO:
#     #   mercury will need:
#     #   Merc quantity available and prices
#     #   If merc price is lower than tigris, use merc inventory first
#     #   When we use merc inventory, gm becomes gm - (merc_price - tigris_cost)
#
#     # TODO:
#     #   Rebate + payment terms valuation when purchasing an item
#     #   Purchasing terms value is realized p_terms days away from the purchase
#     #   Rebate is realized when the item is received? (could also just tack it on at the end) only tigris purchases count
#     #   Dont need to track the above with arrays, can probably just tack on
#
#     overhead_cost_per_unit = 12.25
#     cost_per_cubic_ft = 0.005
#     deval_days = days
#     inventory = np.zeros(deval_days, dtype=int64)
#     sales = np.zeros(deval_days, dtype=int64)
#     purch_inventory_sales = np.zeros(deval_days, dtype=int64)
#     cum_sales = 0
#     receipts = np.zeros(deval_days, dtype=int64)
#     ar = np.zeros(deval_days, dtype=int64)
#     ap = np.zeros(deval_days, dtype=int64)
#     demand = demand_sample
#     inventory[0] = start_inv
#     left_on_order = on_order - cancel_qty
#     already_on_order_daily_receipts = int64(max(left_on_order // lt_today, left_on_order))
#     merc_cap = demand.sum() * .75
#
#     merc_inv = np.zeros(deval_days, dtype=int64)
#     tigris_gm = (gm * cost)
#     merc_gm = cost - merc_prices + tigris_gm + cop
#     merc_prices = np.column_stack((merc_prices, merc_gm))
#     # ONLY BUY MERCURY IF GM > 0 (WE WILL MAKE MONEY)
#     merc_prices = merc_prices[merc_prices[:, 1] > 0]
#     merc_inv[0] = len(merc_prices)
#     merc_sales = np.zeros(deval_days, dtype=int64)
#     merc_gm = np.zeros(deval_days, dtype=float64)
#
#     j = 1
#     while left_on_order > 0:
#         receipts[j] = min(already_on_order_daily_receipts, left_on_order)
#         left_on_order = left_on_order - receipts[j]
#         j += 1
#     receipts[min(lt_today, deval_days)] += purch_today
#     receipts[min(lt_next_op + next_purch_days_away, deval_days)] += purch_next_op
#     for i in range(0, deval_days):
#         sales[i] = min(demand[i], inventory[i] + merc_inv[i])
#         # print(demand[i], inventory[i], merc_inv[i])
#         # print(f'demand: {demand[i]}, inv: {inventory[i]}, merc_inv: {merc_inv[i]}')
#
#         # ALLOCATING SALES WHERE MERC IS CHEAPER FIRST
#         daily_gm = np.zeros(sales[i], dtype=float64)  # Calculating Mercury GM for each Day
#         for unit in range(0, max(min(sales[i], merc_inv[i]),0)):  # For each unit we will sell on a given day decide whether it should be mercury or regular
#             price, gm_val = merc_prices[unit]
#             if price < (cost - value_on_purchase + cop) and price > 0 and len(daily_gm) < merc_cap:
#                 daily_gm[unit] = gm_val
#
#         daily_gm = daily_gm[daily_gm != 0]
#         merc_prices = merc_prices[len(daily_gm):] # Remove the unit from merc_prices once it's been used
#
#         # This will allocate the rest of mercury inventory that is needed to meet demand/ sales
#         if len(daily_gm) + inventory[i] < sales[i]:
#             default_merc = sales[i] - (len(daily_gm) + inventory[i])
#             daily_gm = np.concatenate((daily_gm, merc_prices[:default_merc, 1]))
#             merc_prices = merc_prices[default_merc:]
#
#         merc_sales[i] = len(daily_gm)
#         merc_gm[i] = np.mean(daily_gm) if np.any(daily_gm != 0) else 0
#
#         cum_sales += sales[i]
#         gap = cum_sales - (start_inv + left_on_order)
#         if cum_sales > (start_inv + left_on_order):
#             purch_inventory_sales[i] = min(sales[i], gap)
#         if i + 1 < deval_days:
#             # TODO: If we sell merc inventory instead, then inventory does not go down by sales[i]
#             inventory[i + 1] = max(inventory [i] + receipts[i] - sales[i] + merc_sales[i], 0)
#             merc_inv[i + 1] = max(merc_inv[i] - merc_sales[i], 0)  # Calculate how much inventory there is for tomorrow
#         if i == 0:
#             ar[i] = purch_inventory_sales[i]
#             ap[i] = receipts[i]
#         if i < s_terms and s_terms > 0:
#             ar[i] = purch_inventory_sales[i] + ar[i - 1]
#         if i >= s_terms and s_terms > 0:
#             ar[i] = purch_inventory_sales[i] + ar[i - 1] - purch_inventory_sales[i - s_terms]
#         if i < p_terms and p_terms > 0:
#             ap[i] = receipts[i] + ap[i - 1]
#         if i >= p_terms and p_terms > 0:
#             ap[i] = receipts[i] + ap[i - 1] - receipts[i - p_terms]
#         if inventory[i] + ar[i] + ap[i] + np.sum(receipts[i:]) == 0:
#             break
#         i += 1
#     deval = inventory[-1] * cost
#     # TODO: This needs to change with mercury because GM is not always the same
#     # sales includes mercury so taking that quantity out then getting tigris gm. Merc GM is already calculated so just need to adjust for quantity
#     # np.set_printoptions(threshold=sys.maxsize)
#     revenue = (sales.sum() - merc_sales.sum()) * tigris_gm + (merc_sales * merc_gm * 0.5).sum() # Mercury Profits set to half for now
#     # print(f'sales: {sales.sum()}, merc_sales: {merc_sales.sum()}, tigris_gm: {tigris_gm}, merc_sales: {merc_sales}, merc_gm: {merc_gm}')
#     investment_charge_account = (inventory * cost) + (ar * sale_price) + (ap * cost)
#     investment_charge = (investment_charge_account * (0.14/365)).sum()
#     # print(f'revenue {revenue}, investment_charge {investment_charge}, deval {deval}, value_on_purchase {value_on_purchase}, purch_today {purch_today}, purch_next_op {purch_next_op}')
#     profit = revenue - investment_charge - deval + (value_on_purchase * purch_today) + (value_on_purchase * purch_next_op)
#     # print(revenue, investment_charge, deval, (value_on_purchase * purch_today), (value_on_purchase * purch_next_op))
#     # print(f'Rev: {revenue}, Invest: {investment_charge}, Deval: {deval}')
#
#     # adding warehouse cost and overhead:
#     # warehouse_cost = (inventory * length * width * height * cost_per_cubic_ft).sum()
#     # overhead_cost = sales.sum() * overhead_cost_per_unit
#     # profit = revenue - investment_charge - deval - warehouse_cost - overhead_cost
#
#     # output = np.column_stack((sales, inventory, merc_inv, merc_gm, merc_sales))
#     # output_df = pd.DataFrame(output)
#     # output_df.to_csv('output.csv')
#     # print(f'Inventory: {inventory},LT: {lt_today},NEXT LT:  {min(next_purch_days_away + lt_next_op + 1, deval_days)}')
#     exp_inventory = np.mean(inventory[lt_today + 1 : min(next_purch_days_away + lt_next_op + 1, deval_days)])
#     return profit, exp_inventory


@njit
def purch_sim(purch_today, purch_next_op, next_purch_days_away, p_terms, s_terms, demand_sample, lt_today, lt_next_op,
              start_inv, on_order, cost, gm, sale_price, length, width, height, days, cancel_qty, value_on_purchase, cop, brand):

    overhead_cost_per_unit = 12.25
    cost_per_cubic_ft = 0.005
    deval_days = days
    inventory = np.zeros(deval_days, dtype=int64)
    sales = np.zeros(deval_days, dtype=int64)
    purch_inventory_sales = np.zeros(deval_days, dtype=int64)
    cum_sales = 0
    receipts = np.zeros(deval_days, dtype=int64)
    ar = np.zeros(deval_days, dtype=int64)
    ap = np.zeros(deval_days, dtype=int64)
    demand = demand_sample
    inventory[0] = start_inv
    left_on_order = on_order - cancel_qty
    already_on_order_daily_receipts = int64(max(left_on_order // lt_today, left_on_order))

    profit_per_unit = (gm * cost) - cop
    # print(gm, cost, tigris_gm)

    j = 1
    while left_on_order > 0:
        receipts[j] = min(already_on_order_daily_receipts, left_on_order)
        left_on_order = left_on_order - receipts[j]
        j += 1
    receipts[min(lt_today, deval_days)] += purch_today
    receipts[min(lt_next_op + next_purch_days_away, deval_days)] += purch_next_op
    for i in range(0, deval_days):
        sales[i] = min(demand[i], inventory[i])

        cum_sales += sales[i]
        gap = cum_sales - (start_inv + left_on_order)
        if cum_sales > (start_inv + left_on_order):
            purch_inventory_sales[i] = min(sales[i], gap)
        if i + 1 < deval_days:
            inventory[i + 1] = max(inventory [i] + receipts[i] - sales[i], 0)
        if i == 0:
            ar[i] = purch_inventory_sales[i]
            ap[i] = receipts[i]
        if i < s_terms and s_terms > 0:
            ar[i] = purch_inventory_sales[i] + ar[i - 1]
        if i >= s_terms and s_terms > 0:
            ar[i] = purch_inventory_sales[i] + ar[i - 1] - purch_inventory_sales[i - s_terms]
        if i < p_terms and p_terms > 0:
            ap[i] = receipts[i] + ap[i - 1]
        if i >= p_terms and p_terms > 0:
            ap[i] = receipts[i] + ap[i - 1] - receipts[i - p_terms]
        if inventory[i] + ar[i] + ap[i] + np.sum(receipts[i:]) == 0:
            break
        i += 1
    deval = inventory[-1] * cost

    revenue = (sales.sum() * profit_per_unit)
    investment_charge_account = (inventory * cost) + (ar * sale_price) + (ap * cost)
    investment_charge = (investment_charge_account * (0.14/365)).sum()
    profit = revenue - investment_charge - deval + (value_on_purchase * purch_today) + (value_on_purchase * purch_next_op)
    exp_inventory = np.mean(inventory[lt_today + 1 : min(next_purch_days_away + lt_next_op + 1, deval_days)])

    # RINNAI OOS PENALTY
    mean_demand = np.mean(demand)
    if brand == 'Rinnai' and mean_demand > .5: #probably edit this to be by ipg
        first_non_zero_index = np.argmax(inventory != 0) #filter for first day with inventory - don't want to penalize for current OOS, only future
        inventory = inventory[first_non_zero_index:lt_today + lt_next_op]
        oos_inventory = inventory[inventory == 0]
        oos_days = len(oos_inventory)
        oos_penalty = (mean_demand * oos_days) * (profit_per_unit + cost + cop)
        profit = revenue - investment_charge - deval + (value_on_purchase * purch_today) + (value_on_purchase * purch_next_op) - oos_penalty

    return profit, exp_inventory



@njit(parallel=True)
def calc_purch_qty(ads_arr, var_arr, lt_today_est, lt_today_var, lt_next_op_est, lt_next_op_var, gm, cost,
                      sale_price,
                      length, width, height, p_terms, s_terms, next_purch_days_away, start_inv, on_order_qty,
                      value_on_purchase, cop, brand, order_multiple=1):
    p_terms = int64(p_terms)
    s_terms = int64(s_terms)
    runs = 500
    # runs = 1
    deval_days = 1096
    max_ads = np.max(ads_arr)
    max_var = np.max(var_arr)
    max_purch_qty = int64(np.maximum(np.minimum(np.round((max_ads) * (lt_today_est), 0), 100), 10))
    # max_purch_qty=10
    # print(f'Max Purchase Quantity: {max_purch_qty}')
    cancel_qty = 0

    # 20% bump if Rinnai - Amelia 3/23/2025
    # if brand == 'Rinnai':
    #     lt_today_est = lt_today_est * 1.2
    #     lt_today_var = lt_today_var * 1.2
    #     lt_next_op_est = lt_next_op_est * 1.2
    #     lt_next_op_var = lt_next_op_var * 1.2
    # # MIN LEAD TIMES TO 14 DAYS - AMELIA 10/15/2024
    # lt_today_est = max(lt_today_est, 14)
    # lt_today_var = max(lt_today_var, 14)
    # lt_next_op_est = max(lt_next_op_est, 14)
    # lt_next_op_var = max(lt_next_op_var, 14)

    buy_profit = np.zeros((runs, max_purch_qty, max_purch_qty))
    exp_inv = np.zeros((runs, max_purch_qty, max_purch_qty))
    cancel_profit = np.zeros((on_order_qty + 1, runs))
    def generate_shared_variables():
        # calc lead time today and next op for run
        lt_today_arr = calc_nb_array_ln(lt_today_est, lt_today_var)
        lt_today_arr_range = np.arange(lt_today_arr.size)
        lt_today = numba_choice(lt_today_arr_range, lt_today_arr, 1)[0]
        # lt must be at least 1 day
        lt_today = max(lt_today, 1)
        lt_next_op_arr = calc_nb_array_ln(lt_next_op_est, lt_next_op_var)
        lt_next_op_arr_range = np.arange(lt_next_op_arr.size)
        lt_next_op = numba_choice(lt_next_op_arr_range, lt_next_op_arr, 1)[0]
        # if lt_next_op is lower, then there will never be a buy. experiment later with removing this constraint
        lt_next_op = max(lt_next_op, lt_today - next_purch_days_away + 1)
        lt_next_op = min(lt_next_op, 730)
        days = deval_days + lt_next_op + next_purch_days_away
        # calculate demand sample
        long_ads = np.mean(ads_arr)
        long_var = np.mean(var_arr)
        demand_sample = np.zeros(ads_arr.size + days)
        for j in range(ads_arr.size):
            demand_array = calc_nb_array_ln(ads_arr[j], var_arr[j])
            demand_range = np.arange(demand_array.size)
            demand_sample[j] = numba_choice(demand_range, demand_array, 1)[0]
        long_demand_array = calc_nb_array_ln(long_ads, long_var)
        long_demand_range = np.arange(long_demand_array.size)
        demand_sample[ads_arr.size:] = numba_choice(long_demand_range, long_demand_array,
                                                    demand_sample.size - ads_arr.size)
        return demand_sample, lt_today, lt_next_op, days

    for run in prange(0, runs):
        demand_sample, lt_today, lt_next_op, days = generate_shared_variables()
        for buy_today in range(0, max_purch_qty):
            for move in range(0, buy_today + 1):
                sim_res_final, exp_inventory_final = purch_sim((buy_today - move) * order_multiple, move * order_multiple,
                                                               next_purch_days_away,p_terms,s_terms, demand_sample,
                                                               lt_today, lt_next_op, start_inv,on_order_qty, cost, gm,
                                                               sale_price, length, width, height, days, cancel_qty,
                                                            value_on_purchase, cop, brand)
                buy_profit[run, move, buy_today] = sim_res_final
                exp_inv[run, move, buy_today] = exp_inventory_final
                # print(f'run {run}, move {move}, buy_today {buy_today}')

        for cancel in range(1, on_order_qty + 1): #just run this from zero to replace the profit calc below??
            sim_res_cancel, exp_inventory_cancel = purch_sim(0, 0, next_purch_days_away, p_terms, s_terms,
                                                             demand_sample, lt_today,
                                                             lt_next_op, start_inv, on_order_qty, cost, gm, sale_price,
                                                             length, width, height,
                                                             days, cancel, value_on_purchase, cop, brand)
            cancel_profit[cancel, run] = sim_res_cancel
    # print(f'cancel profit: {cancel_profit}')

    # Average buy today and move vals across runs
    buy_averages = np.zeros((max_purch_qty, max_purch_qty))
    for r in range(runs):
        buy_averages += buy_profit[r, :, :]
    buy_averages = buy_averages/runs

    buy_averages = np.maximum(buy_averages, 0)
    # print(buy_averages)
    move, buy_today = np.argwhere(buy_averages == buy_averages.max())[0]
    number_to_buy_today = int64(buy_today - move)
    # print(move, buy_today, number_to_buy_today)

    # exp_inv_arr = exp_inv[(exp_inv[:, 1] == move) & (exp_inv[:, 2] == buy_today)]
    # expected_inventory = np.mean(exp_inv_arr)

    cancel_averages = np.zeros(on_order_qty + 1)
    for i, sub_array in enumerate(cancel_profit):
        cancel_averages[i] = np.maximum(np.mean(sub_array), 0)
    optimal_cancel = np.argmax(cancel_averages)
    number_to_cancel = int64(optimal_cancel)

    # determine expected inventory levels:
    exp_inv_arr = np.zeros(runs, dtype=float64)
    for inv_run in prange(0, runs):
        demand_sample, lt_today, lt_next_op, days = generate_shared_variables()
        days = int(round(deval_days + lt_next_op + next_purch_days_away, 0))
        sim_res, exp_inventory = purch_sim(buy_today, move, next_purch_days_away, p_terms, s_terms,
                                           demand_sample, lt_today, lt_next_op,
                                           start_inv, on_order_qty, cost, gm, sale_price, length, width, height, days,
                                           number_to_cancel, value_on_purchase, cop, brand)
        # print(f'run: {inv_run}, demand: {demand_sample[0]}, lt1: {lt_today}, lt2: {lt_next_op}, profit: {float(sim_res)}, inv: {float(exp_inventory)}')
        # print(demand_sample)
        # print(exp_inventory)

        exp_inv_arr[inv_run] = exp_inventory

    expected_inventory = np.mean(exp_inv_arr)
    # print(exp_inv_arr)
    # print(expected_inventory)

    # determine savings due to cancelling:
    cancel_savings_arr = np.zeros(runs, dtype=float64)
    if number_to_cancel > 0:
        for cancel_run in prange(0, runs):
            demand_sample, lt_today, lt_next_op, days = generate_shared_variables()
            days = int(round(deval_days + lt_next_op + next_purch_days_away, 0))
            sim_res, exp_inventory = purch_sim(number_to_buy_today, 0, next_purch_days_away, p_terms, s_terms,
                                               demand_sample, lt_today, lt_next_op,
                                               start_inv, on_order_qty, cost, gm, sale_price, length, width, height,
                                               days, 0, value_on_purchase, cop, brand)
            cancel_res, exp_inventory = purch_sim(number_to_buy_today, 0, next_purch_days_away, p_terms, s_terms,
                                                  demand_sample, lt_today, lt_next_op,
                                                  start_inv, on_order_qty, cost, gm, sale_price, length, width, height,
                                                  days, number_to_cancel, value_on_purchase, cop, brand)
            cancel_savings_arr[cancel_run] = cancel_res - sim_res

    cancel_savings = np.round(np.mean(cancel_savings_arr), 2)
    if number_to_cancel > 0 and cancel_savings < 0:
        number_to_cancel = 0
        cancel_savings = 0
    if number_to_buy_today > 0:
        cancel_savings = 0
        number_to_cancel = 0

    return number_to_buy_today, number_to_cancel, expected_inventory, cancel_savings


# @njit(parallel=True)
# def calc_purch_qty_merc(ads_arr, var_arr, lt_today_est, lt_today_var, lt_next_op_est, lt_next_op_var, gm, cost,
#                       sale_price,
#                       length, width, height, p_terms, s_terms, next_purch_days_away, start_inv, on_order_qty,
#                       merc_prices,
#                       value_on_purchase, cop, order_multiple=1):
#     p_terms = int64(p_terms)
#     s_terms = int64(s_terms)
#     runs = 200
#     deval_days = 1096
#     max_purch_qty = 1000
#     max_ads = np.max(ads_arr)
#     # print(max_ads)
#     max_var = np.max(var_arr)
#     # print(max_var)
#     # print(lt_today_est)
#     # POTENTIALLY INCREASE MAX NUMBER LATER TO ACCOUNT FOR GROWTH
#     max_purch_qty = int64(np.maximum(np.minimum(np.round((max_ads) * (lt_today_est), 0), 100), 10))
#     # max_purch_qty = int64(np.maximum(np.minimum(np.round((max_ads + max_var), 0), 100), 10))
#     # max_purch_qty=5
#     # runs = 10
#     print(max_purch_qty)
#
#     total_to_buy = 0
#     max_no_push = 0
#     cancel_qty = 0
#
#     # MIN LEAD TIMES TO 14 DAYS - AMELIA 10/15/2024
#     lt_today_est = max(lt_today_est, 14)
#     lt_next_op_est = max(lt_next_op_est, 14)
#
#     buy_profit = np.zeros((runs, max_purch_qty, max_purch_qty))
#     cancel_profit = np.zeros((on_order_qty + 1, runs))
#
#     def generate_shared_variables():
#         # calc lead time today and next op for run
#         lt_today_arr = calc_nb_array_ln(lt_today_est, lt_today_var)
#         lt_today_arr_range = np.arange(lt_today_arr.size)
#         lt_today = numba_choice(lt_today_arr_range, lt_today_arr, 1)[0]
#         # lt must be at least 1 day
#         lt_today = max(lt_today, 1)
#         lt_next_op_arr = calc_nb_array_ln(lt_next_op_est, lt_next_op_var)
#         lt_next_op_arr_range = np.arange(lt_next_op_arr.size)
#         lt_next_op = numba_choice(lt_next_op_arr_range, lt_next_op_arr, 1)[0]
#         # if lt_next_op is lower, then there will never be a buy. experiment later with removing this constraint
#         lt_next_op = max(lt_next_op, lt_today - next_purch_days_away + 1)
#         days = deval_days + lt_next_op + next_purch_days_away
#         # calculate demand sample
#         long_ads = np.mean(ads_arr)
#         long_var = np.mean(var_arr)
#         demand_sample = np.zeros(ads_arr.size + days)
#         for j in range(ads_arr.size):
#             demand_array = calc_nb_array_ln(ads_arr[j], var_arr[j])
#             demand_range = np.arange(demand_array.size)
#             demand_sample[j] = numba_choice(demand_range, demand_array, 1)[0]
#         long_demand_array = calc_nb_array_ln(long_ads, long_var)
#         long_demand_range = np.arange(long_demand_array.size)
#         demand_sample[ads_arr.size:] = numba_choice(long_demand_range, long_demand_array,
#                                                     demand_sample.size - ads_arr.size)
#
#         return demand_sample, lt_today, lt_next_op, days
#
#     for run in prange(0, runs):
#         demand_sample, lt_today, lt_next_op, days = generate_shared_variables()
#         for buy_today in range(0, max_purch_qty):
#             for move in range(0, buy_today + 1):
#                 sim_res_final, exp_inventory_final = purch_sim((buy_today - move) * order_multiple, move * order_multiple,
#                                                                next_purch_days_away,p_terms,s_terms, demand_sample,
#                                                                lt_today, lt_next_op, start_inv,on_order_qty, cost, gm,
#                                                                sale_price, length, width, height,days, cancel_qty,
#                                                                merc_prices, value_on_purchase, cop)
#                 buy_profit[run, move, buy_today] = sim_res_final
#                 # print(f'run {run}, move {move}, buy_today {buy_today}')
#
#         for cancel in range(1, on_order_qty + 1):
#             sim_res_cancel, exp_inventory_cancel = purch_sim(0, 0, next_purch_days_away, p_terms, s_terms,
#                                                              demand_sample, lt_today,
#                                                              lt_next_op, start_inv, on_order_qty, cost, gm, sale_price,
#                                                              length, width, height,
#                                                              days, cancel, merc_prices, value_on_purchase, cop)
#             cancel_profit[cancel, run] = sim_res_cancel
#
#     # Average buy today and move vals across runs
#     # print(buy_profit)
#     # print(cancel_profit)
#     # np.savetxt('buy_profit_ru199ip.csv', buy_profit, delimiter=',')
#     # np.savetxt('cancel_profit_ru199ip.csv', cancel_profit, delimiter=',')
#     buy_averages = np.zeros((max_purch_qty, max_purch_qty))
#     for r in range(runs):
#         buy_averages += buy_profit[r, :, :]
#     buy_averages = buy_averages/runs
#
#     buy_averages = np.maximum(buy_averages, 0)
#     move, buy_today = np.argwhere(buy_averages == buy_averages.max())[0]
#     number_to_buy_today = int64(buy_today - move)
#
#     cancel_averages = np.zeros(on_order_qty + 1)
#     for i, sub_array in enumerate(cancel_profit):
#         cancel_averages[i] = np.maximum(np.mean(sub_array), 0)
#     optimal_cancel = np.argmax(cancel_averages)
#     number_to_cancel = int64(optimal_cancel)
#
#     # determine expected inventory levels:
#     exp_inv_arr = np.zeros(runs, dtype=float64)
#     for inv_run in prange(0, runs):
#         demand_sample, lt_today, lt_next_op, days = generate_shared_variables()
#         days = int(round(deval_days + lt_next_op + next_purch_days_away, 0))
#         sim_res, exp_inventory = purch_sim(number_to_buy_today, 0, next_purch_days_away, p_terms, s_terms,
#                                            demand_sample, lt_today, lt_next_op,
#                                            start_inv, on_order_qty, cost, gm, sale_price, length, width, height, days,
#                                            number_to_cancel, merc_prices, value_on_purchase, cop)
#         exp_inv_arr[inv_run] = exp_inventory
#
#     expected_inventory = np.mean(exp_inv_arr)
#
#     # determine savings due to cancelling:
#     cancel_savings_arr = np.zeros(runs, dtype=float64)
#     if number_to_cancel > 0:
#         for cancel_run in prange(0, runs):
#             demand_sample, lt_today, lt_next_op, days = generate_shared_variables()
#             days = int(round(deval_days + lt_next_op + next_purch_days_away, 0))
#             sim_res, exp_inventory = purch_sim(number_to_buy_today, 0, next_purch_days_away, p_terms, s_terms,
#                                                demand_sample, lt_today, lt_next_op,
#                                                start_inv, on_order_qty, cost, gm, sale_price, length, width, height,
#                                                days, 0, merc_prices, value_on_purchase, cop)
#             cancel_res, exp_inventory = purch_sim(number_to_buy_today, 0, next_purch_days_away, p_terms, s_terms,
#                                                   demand_sample, lt_today, lt_next_op,
#                                                   start_inv, on_order_qty, cost, gm, sale_price, length, width, height,
#                                                   days, number_to_cancel, merc_prices, value_on_purchase, cop)
#             cancel_savings_arr[cancel_run] = cancel_res - sim_res
#
#     cancel_savings = np.round(np.mean(cancel_savings_arr), 2)
#
#     if number_to_cancel > 0 and cancel_savings < 0:
#         number_to_cancel = 0
#         cancel_savings = 0
#     if number_to_buy_today > 0:
#         cancel_savings = 0
#         number_to_cancel = 0
#
#     return number_to_buy_today, number_to_cancel, expected_inventory, cancel_savings
#
#
# @njit(parallel=True)
# def calc_purch_qty_v1(ads_arr, var_arr, lt_today_est, lt_today_var, lt_next_op_est, lt_next_op_var, gm, cost, sale_price,
#                    length, width, height, p_terms, s_terms, next_purch_days_away, start_inv, on_order_qty, merc_prices,
#                    value_on_purchase, cop, order_multiple=1):
#     p_terms = int64(p_terms)
#     s_terms = int64(s_terms)
#     runs = 200
#     deval_days = 1096
#     max_purch_qty = 1000
#     buy_results = np.zeros(runs, dtype=int64)
#     cancel_results = np.zeros(runs, dtype=int64)
#     total_to_buy = 0
#     max_no_push = 0
#     cancel_qty = 0
#     def generate_shared_variables():
#         # calc lead time today and next op for run
#         lt_today_arr = calc_nb_array_ln(lt_today_est, lt_today_var)
#         lt_today_arr_range = np.arange(lt_today_arr.size)
#         lt_today = numba_choice(lt_today_arr_range, lt_today_arr, 1)[0]
#         # lt must be at least 1 day
#         lt_today = max(lt_today, 1)
#         lt_next_op_arr = calc_nb_array_ln(lt_next_op_est, lt_next_op_var)
#         lt_next_op_arr_range = np.arange(lt_next_op_arr.size)
#         lt_next_op = numba_choice(lt_next_op_arr_range, lt_next_op_arr, 1)[0]
#         # if lt_next_op is lower, then there will never be a buy. experiment later with removing this constraint
#         lt_next_op = max(lt_next_op, lt_today - next_purch_days_away + 1)
#         days = deval_days + lt_next_op + next_purch_days_away
#         # calculate demand sample
#         long_ads = np.mean(ads_arr)
#         long_var = np.mean(var_arr)
#         demand_sample = np.zeros(ads_arr.size + days)
#         for j in range(ads_arr.size):
#             demand_array = calc_nb_array_ln(ads_arr[j], var_arr[j])
#             demand_range = np.arange(demand_array.size)
#             demand_sample[j] = numba_choice(demand_range, demand_array, 1)[0]
#         long_demand_array = calc_nb_array_ln(long_ads, long_var)
#         long_demand_range = np.arange(long_demand_array.size)
#         demand_sample[ads_arr.size:] = numba_choice(long_demand_range, long_demand_array, demand_sample.size - ads_arr.size)
#
#         return demand_sample, lt_today, lt_next_op, days
#
#     for run in prange(0, runs):
#         demand_sample, lt_today, lt_next_op, days = generate_shared_variables()
#
#         profit_a = np.zeros(max_purch_qty, dtype=float64)
#         for i in range(max_purch_qty):
#             days = int(round(deval_days + lt_next_op + next_purch_days_away,0))
#             sim_res, exp_inventory = purch_sim(i * order_multiple, 0, next_purch_days_away, p_terms, s_terms, demand_sample, lt_today, lt_next_op,
#                                                start_inv, on_order_qty, cost, gm, sale_price, length, width, height, days, cancel_qty, merc_prices, value_on_purchase, cop)
#             profit_a[i] = sim_res
#             if i > 0 and profit_a[i] < profit_a[i - 1]: #IF THIS RUN IS LESS PROFITABLE THAN THE LAST BUY THE LAST NUM AND STOP
#                 total_to_buy = i - 1
#                 max_no_push = profit_a[i - 1]
#                 # IDEA: INSTEAD OF BREAKING HERE IT RUNS FOR X RUNS EVERYTIME (MAYBE X IS A FUNCTION OF ADS TO AVOID BEING TOO COMPUTATIONALLY EXPENSIVE UNECESSARILY)
#                 # AND THEN TAKE THE MAX PROFIT FROM THE Is AND THEN AVERAGE ACROSS ALL RUNS TO GET E[PROFIT] THEN THAT TELLS YOU HOW MANY TO ORDER FOR "to_buy"
#                 break
#         moved_to_next_op = 0
#         last_res = max_no_push
#         # THIS PART TESTS IF WE MOVE A UNIT TO THE NEXT PURCH OPPORTUNITY IF IT IS MORE PROFITABLE
#         if total_to_buy > 0:
#             for move in range(1, total_to_buy + 1):
#                 sim_res_final, exp_inventory_final = purch_sim((total_to_buy - move) * order_multiple, move * order_multiple, next_purch_days_away, p_terms, s_terms, demand_sample,
#                                                                lt_today, lt_next_op, start_inv, on_order_qty, cost, gm, sale_price, length, width, height,
#                                                                days, cancel_qty, merc_prices, value_on_purchase, cop)
#                 if move == total_to_buy:
#                     moved_to_next_op = total_to_buy
#                     break
#                 if sim_res_final < last_res:
#                     moved_to_next_op = move - 1
#                     break
#                 else:
#                     last_res = sim_res_final
#             buy_results[run] = total_to_buy - moved_to_next_op
#         else:
#             qty_to_cancel = 0
#             for cancel in range(1, on_order_qty + 1):
#                 sim_res_cancel, exp_inventory_cancel = purch_sim(0, 0, next_purch_days_away, p_terms, s_terms, demand_sample, lt_today,
#                                                                  lt_next_op, start_inv, on_order_qty, cost, gm, sale_price, length, width, height,
#                                                                  days, cancel, merc_prices, value_on_purchase, cop)
#                 if cancel == on_order_qty:
#                     qty_to_cancel = on_order_qty
#                     break
#                 if sim_res_cancel < last_res:
#                     qty_to_cancel = cancel - 1
#                     break
#                 else:
#                     last_res = sim_res_cancel
#             cancel_results[run] = qty_to_cancel
#
#     number_to_buy_today = int64(round(np.mean(buy_results), 0))
#     weights = np.where(buy_results == 0, 0.5, 1)
#     number_to_buy_today = int64(round(np.average(buy_results, weights=weights),0))
#     number_to_cancel = int64(round(np.mean(cancel_results), 0))
#
#     # determine expected inventory levels:
#     exp_inv_arr = np.zeros(runs, dtype=float64)
#     for inv_run in prange(0, runs):
#         demand_sample, lt_today, lt_next_op, days = generate_shared_variables()
#         days = int(round(deval_days + lt_next_op + next_purch_days_away,0))
#         sim_res, exp_inventory = purch_sim(number_to_buy_today, 0, next_purch_days_away, p_terms, s_terms, demand_sample, lt_today, lt_next_op,
#                                            start_inv, on_order_qty, cost, gm, sale_price, length, width, height, days, number_to_cancel, merc_prices, value_on_purchase, cop)
#         exp_inv_arr[inv_run] = exp_inventory
#
#     expected_inventory = np.mean(exp_inv_arr)
#
#     # determine savings due to cancelling:
#     cancel_savings_arr = np.zeros(runs, dtype=float64)
#     if number_to_cancel > 0:
#         for cancel_run in prange(0, runs):
#             demand_sample, lt_today, lt_next_op, days = generate_shared_variables()
#             days = int(round(deval_days + lt_next_op + next_purch_days_away,0))
#             sim_res, exp_inventory = purch_sim(number_to_buy_today, 0, next_purch_days_away, p_terms, s_terms, demand_sample, lt_today, lt_next_op,
#                                                start_inv, on_order_qty, cost, gm, sale_price, length, width, height, days, 0, merc_prices, value_on_purchase, cop)
#             cancel_res, exp_inventory = purch_sim(number_to_buy_today, 0, next_purch_days_away, p_terms, s_terms, demand_sample, lt_today, lt_next_op,
#                                                   start_inv, on_order_qty, cost, gm, sale_price, length, width, height, days, number_to_cancel, merc_prices, value_on_purchase, cop)
#             cancel_savings_arr[cancel_run] = cancel_res - sim_res
#
#     cancel_savings = np.round(np.mean(cancel_savings_arr), 2)
#
#     if number_to_cancel > 0 and cancel_savings < 0:
#         number_to_cancel = 0
#         cancel_savings = 0
#     if number_to_buy_today > 0:
#         cancel_savings = 0
#         number_to_cancel = 0
#
#     return number_to_buy_today, number_to_cancel, expected_inventory, cancel_savings
#
#
# @njit(parallel=True)
# def calc_purch_qty_v2(ads_arr, var_arr, lt_today_est, lt_today_var, lt_next_op_est, lt_next_op_var, gm, cost, sale_price,
#                    length, width, height, p_terms, s_terms, next_purch_days_away, start_inv, on_order_qty, merc_prices,
#                    value_on_purchase, cop, order_multiple=1):
#     p_terms = int64(p_terms)
#     s_terms = int64(s_terms)
#     runs = 10
#     deval_days = 1096
#     max_purch_qty = 1000
#     max_ads = np.max(ads_arr)
#     max_var = np.max(var_arr)
#     max_purch_qty = int64(np.maximum(np.minimum(np.round((max_ads + max_var) * (lt_today_est), 0), 1000), 100))
#
#     total_to_buy = 0
#     max_no_push = 0
#     cancel_qty = 0
#
#     buy_profit = np.zeros((max_purch_qty, runs))
#     cancel_profit = np.zeros((on_order_qty+1, runs))
#
#     def generate_shared_variables():
#         # calc lead time today and next op for run
#         lt_today_arr = calc_nb_array_ln(lt_today_est, lt_today_var)
#         lt_today_arr_range = np.arange(lt_today_arr.size)
#         lt_today = numba_choice(lt_today_arr_range, lt_today_arr, 1)[0]
#         # lt must be at least 1 day
#         lt_today = max(lt_today, 1)
#         lt_next_op_arr = calc_nb_array_ln(lt_next_op_est, lt_next_op_var)
#         lt_next_op_arr_range = np.arange(lt_next_op_arr.size)
#         lt_next_op = numba_choice(lt_next_op_arr_range, lt_next_op_arr, 1)[0]
#         # if lt_next_op is lower, then there will never be a buy. experiment later with removing this constraint
#         lt_next_op = max(lt_next_op, lt_today - next_purch_days_away + 1)
#         days = deval_days + lt_next_op + next_purch_days_away
#         # calculate demand sample
#         long_ads = np.mean(ads_arr)
#         long_var = np.mean(var_arr)
#         demand_sample = np.zeros(ads_arr.size + days)
#         for j in range(ads_arr.size):
#             demand_array = calc_nb_array_ln(ads_arr[j], var_arr[j])
#             demand_range = np.arange(demand_array.size)
#             demand_sample[j] = numba_choice(demand_range, demand_array, 1)[0]
#         long_demand_array = calc_nb_array_ln(long_ads, long_var)
#         long_demand_range = np.arange(long_demand_array.size)
#         demand_sample[ads_arr.size:] = numba_choice(long_demand_range, long_demand_array, demand_sample.size - ads_arr.size)
#
#         return demand_sample, lt_today, lt_next_op, days
#
#     for run in prange(0, runs):
#         demand_sample, lt_today, lt_next_op, days = generate_shared_variables()
#         # print(np.mean(demand_sample))
#
#         profit_a = np.zeros(max_purch_qty, dtype=float64)
#         for i in range(max_purch_qty):
#             days = int(round(deval_days + lt_next_op + next_purch_days_away,0))
#             sim_res, exp_inventory = purch_sim(i * order_multiple, 0, next_purch_days_away, p_terms, s_terms, demand_sample, lt_today, lt_next_op,
#                                                start_inv, on_order_qty, cost, gm, sale_price, length, width, height, days, cancel_qty, merc_prices, value_on_purchase, cop)
#             # print(f'sim profit: {sim_res},sim inv:  {exp_inventory}')
#             profit_a[i] = sim_res
#             if i > 0 and profit_a[i] < profit_a[i - 1]:
#                 total_to_buy = i - 1
#                 max_no_push = profit_a[i - 1]
#                 break
#         last_res = max_no_push
#         # print(last_res, total_to_buy)
#         buy_profit[total_to_buy, run] = last_res
#         cancel_profit[0, run] = last_res
#         # THIS PART TESTS IF WE MOVE A UNIT TO THE NEXT PURCH OPPORTUNITY IF IT IS MORE PROFITABLE
#         if total_to_buy > 0:
#             for move in range(1, max_purch_qty):
#                 sim_res_final, exp_inventory_final = purch_sim(np.maximum((total_to_buy - move) * order_multiple, 0), move * order_multiple, next_purch_days_away, p_terms, s_terms, demand_sample,
#                                                                    lt_today, lt_next_op, start_inv, on_order_qty, cost, gm, sale_price, length, width, height,
#                                                                    days, cancel_qty, merc_prices, value_on_purchase, cop)
#                 buy_profit[total_to_buy - move, run] = sim_res_final
#
#                 # print(total_to_buy - move, sim_res_final)
#
#                     # if total_to_buy == move:
#                     #     break
#         else:
#             for cancel in range(1, on_order_qty + 1):
#                 sim_res_cancel, exp_inventory_cancel = purch_sim(0, 0, next_purch_days_away, p_terms, s_terms, demand_sample, lt_today,
#                                                                  lt_next_op, start_inv, on_order_qty, cost, gm, sale_price, length, width, height,
#                                                                  days, cancel, merc_prices, value_on_purchase, cop)
#                 cancel_profit[cancel, run] = sim_res_cancel
#     # print(f'buy profit: {buy_profit}')
#     # AVERAGING PROFIT ACROSS RUNS THEN PICKING QUANTITY W/ MAX PROFIT
#     buy_averages = np.zeros(max_purch_qty)
#     for i, sub_array in enumerate(buy_profit):
#         buy_averages[i] = np.maximum(np.mean(sub_array), 0)
#     number_to_buy_today = int64(np.argmax(buy_averages))
#     np.set_printoptions(threshold=sys.maxsize)
#     # print(buy_profit, buy_averages, number_to_buy_today)
#     # print(f'avgs: {buy_averages},buy today: {number_to_buy_today}')
#
#     # print(cancel_profit)
#     cancel_averages = np.zeros(on_order_qty+1)
#     for i, sub_array in enumerate(cancel_profit):
#         cancel_averages[i] = np.maximum(np.mean(sub_array),0)
#     optimal_cancel = np.argmax(cancel_averages)
#     number_to_cancel = int64(optimal_cancel)
#     # print(cancel_averages, optimal_cancel, number_to_cancel)
#
#
#     # determine expected inventory levels:
#     exp_inv_arr = np.zeros(runs, dtype=float64)
#     for inv_run in prange(0, runs):
#         demand_sample, lt_today, lt_next_op, days = generate_shared_variables()
#         days = int(round(deval_days + lt_next_op + next_purch_days_away,0))
#         sim_res, exp_inventory = purch_sim(number_to_buy_today, 0, next_purch_days_away, p_terms, s_terms, demand_sample, lt_today, lt_next_op,
#                                            start_inv, on_order_qty, cost, gm, sale_price, length, width, height, days, number_to_cancel, merc_prices, value_on_purchase, cop)
#         exp_inv_arr[inv_run] = exp_inventory
#
#     expected_inventory = np.mean(exp_inv_arr)
#
#     # determine savings due to cancelling:
#     cancel_savings_arr = np.zeros(runs, dtype=float64)
#     if number_to_cancel > 0:
#         for cancel_run in prange(0, runs):
#             demand_sample, lt_today, lt_next_op, days = generate_shared_variables()
#             days = int(round(deval_days + lt_next_op + next_purch_days_away,0))
#             sim_res, exp_inventory = purch_sim(number_to_buy_today, 0, next_purch_days_away, p_terms, s_terms, demand_sample, lt_today, lt_next_op,
#                                                start_inv, on_order_qty, cost, gm, sale_price, length, width, height, days, 0, merc_prices, value_on_purchase, cop)
#             cancel_res, exp_inventory = purch_sim(number_to_buy_today, 0, next_purch_days_away, p_terms, s_terms, demand_sample, lt_today, lt_next_op,
#                                                   start_inv, on_order_qty, cost, gm, sale_price, length, width, height, days, number_to_cancel, merc_prices, value_on_purchase, cop)
#             cancel_savings_arr[cancel_run] = cancel_res - sim_res
#
#     cancel_savings = np.round(np.mean(cancel_savings_arr), 2)
#
#     if number_to_cancel > 0 and cancel_savings < 0:
#         number_to_cancel = 0
#         cancel_savings = 0
#     if number_to_buy_today > 0:
#         cancel_savings = 0
#         number_to_cancel = 0
#
#     return number_to_buy_today, number_to_cancel, expected_inventory, cancel_savings



def main():
    start_time = time.time()

    quantile_nn_output = pd.read_csv('fake_quantile_outputs_demand.csv')
    output_np_array = quantile_nn_output.drop(columns='Item').values
    ads_var_est = get_ads_var_estimates(output_np_array, quantile_targets)
    lt_nn_output = pd.read_csv('fake_quantile_outputs_lead_time.csv')
    output_lt_np_array = lt_nn_output.drop(columns='Item').values
    lt_est_today = get_ads_var_estimates(np.array(output_lt_np_array[0][np.newaxis, :]), quantile_targets_lt)
    next_op_days = 7
    lt_est_next_op = get_ads_var_estimates(np.array(output_lt_np_array[next_op_days][np.newaxis, :]), quantile_targets_lt)

    purch_qty = calc_purch_qty(ads_var_est[:,0], ads_var_est[:,1], lt_est_today[0,0], lt_est_today[0,1], lt_est_next_op[0,0], lt_est_next_op[0,1], 20, 80, 100, 1, 1, 1, 30, 30, next_op_days, 32)
    print(purch_qty)

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"elapsed time: {elapsed_time} seconds")


@njit(parallel=True)
def get_ads_var_estimates(nn_arr, quantile_targets):
    out_arr = np.zeros((nn_arr.shape[0], 2))
    for i in prange(0, nn_arr.shape[0]):
        res = min_loss(nn_arr[i,:], quantile_targets)
        round_res = np.zeros_like(res)
        round_res = np.round_(res, decimals=5, out=round_res)
        out_arr[i,:] = round_res
    return out_arr


if __name__ == '__main__':
    main()
