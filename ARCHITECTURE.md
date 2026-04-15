# OptiStock API — Architecture

## Overview

OptiStock is a paid API hosted on [Modal](https://modal.com) that calculates profit-maximizing purchase quantities for inventory items. It combines **demand forecasting** (moving averages + ML regression) with **Monte Carlo simulation** optimization to recommend how many units of each item to order right now.

The core insight: the simulation computes an optimal *order-up-to* stock level (PSL), and the API translates that into an actionable *order quantity* by factoring in current inventory positions.

---

## Key Design Decisions

### 1. PSL = Order Up To Point

The simulation engine outputs an **optimal PSL** (Periodic Stock Level) — the total inventory ceiling to target. The API then derives the actionable order quantity:

```
recommended_order_qty = max(0, optimal_psl - current_available - on_order_qty + back_order_qty)
```

This means the customer doesn't need to understand PSL; they get a "how many to buy now" number directly.

### 2. Order Frequency

`order_frequency_days` determines when the next purchase decision will occur. This is critical for simulation accuracy — it sets the review horizon. Must be provided per item or inferred from historical patterns.

### 3. Tiered Demand Prediction

| Tier | Engine | Output | When to Use |
|------|--------|--------|-------------|
| **Basic** | `calculate_adjusted_demand` | `final_avg`, `final_var` | Insufficient data for ML; well-tuned moving average |
| **Premium** | `ml-regression` (XGBoost / BayesianRidge) | `ads`, `variance` | Sufficient historical data exists |

Both tiers produce average daily sales (ADS) and variance, which feed directly into the simulation.

### 4. Cost of Capital

Previously hardcoded at 14% annual (`invst_charge = .14 / 365`). Now a **configurable API input** (`cost_of_capital_annual`), converted to a daily rate internally. This directly affects the holding cost calculation in the simulation.

### 5. Simulation Engine

Cleaned version of the original `simulation.py`, retaining only active functions. Uses the improved `neg_bin_ln` from `sim2.py` — **log-space computation** for numerical stability with extreme parameter values. All core simulation functions are JIT-compiled with Numba `@njit`.

### 6. Hosting on Modal

The entire API runs on Modal:
- **FastAPI** for HTTP handling
- **Numba JIT** for simulation performance
- Cold-start JIT warmup is the main latency concern (~2–5s per function on first call)

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/optimize` | Full pipeline: demand forecast → simulate → order recommendation |
| `POST` | `/v1/demand` | Demand prediction only (returns ADS/variance) |
| `POST` | `/v1/simulate` | Simulation only (provide ADS/variance directly) |
| `GET` | `/health` | Liveness check |

### Authentication

All endpoints require a valid API key passed via the `X-API-Key` header. Validated by `api/auth.py`.

---

## Request & Response Schemas

### Optimize Request (`POST /v1/optimize`)

```json
{
  "items": [
    {
      "item_id": "string",
      "current_available": 0,
      "on_order_qty": 0,
      "back_order_qty": 0,
      "order_frequency_days": 7,
      "cost": 50.0,
      "sale_price": 100.0,
      "length": 1.0,
      "width": 1.0,
      "height": 1.0,
      "payment_terms_days": 30,
      "sales_terms_days": 30,
      "lead_time_days": 14,
      "historical_data": {
        "dates": ["2025-01-01", "..."],
        "quantities": [5, "..."],
        "available_flags": [1, "..."]
      }
    }
  ],
  "cost_of_capital_annual": 0.14,
  "tier": "basic | premium"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `item_id` | string | Unique item identifier |
| `current_available` | int | Units currently in stock |
| `on_order_qty` | int | Units already on order |
| `back_order_qty` | int | Units on back order (added to available position) |
| `order_frequency_days` | int | Days between purchase decisions |
| `cost` | float | Unit purchase cost |
| `sale_price` | float | Unit selling price |
| `length/width/height` | float | Physical dimensions (for cube calculation) |
| `payment_terms_days` | int | Days to pay supplier |
| `sales_terms_days` | int | Days customer has to pay |
| `lead_time_days` | int | Supplier lead time |
| `historical_data` | object | Time series of demand with availability flags |
| `cost_of_capital_annual` | float | Annual cost of capital rate (e.g. 0.14) |
| `tier` | string | `"basic"` or `"premium"` |

### Optimize Response

```json
{
  "items": [
    {
      "item_id": "string",
      "optimal_psl": 150,
      "recommended_order_qty": 50,
      "expected_profit": 1234.56,
      "expected_daily_sales": 5.2,
      "expected_avg_inventory": 75.3,
      "cube_usage": 75.3,
      "profit_per_cube": 16.4,
      "demand_source": "adjusted_demand | ml_regression",
      "ads": 5.2,
      "variance": 8.1,
      "warnings": []
    }
  ],
  "compute_time_ms": 3200
}
```

| Field | Type | Description |
|-------|------|-------------|
| `optimal_psl` | int | Simulated optimal order-up-to level |
| `recommended_order_qty` | int | `max(0, optimal_psl - available - on_order + back_order)` |
| `expected_profit` | float | Expected profit at optimal PSL |
| `expected_daily_sales` | float | Predicted average daily sales |
| `expected_avg_inventory` | float | Expected average inventory level |
| `cube_usage` | float | Expected average cubic volume occupied |
| `profit_per_cube` | float | `expected_profit / cube_usage` |
| `demand_source` | string | `"adjusted_demand"` or `"ml_regression"` |
| `ads` | float | Average daily sales used for simulation |
| `variance` | float | Variance used for simulation |
| `warnings` | string[] | Any issues encountered (e.g. insufficient data) |
| `compute_time_ms` | int | Total pipeline compute time |

---

## Pipeline Logic

```
┌─────────────────────────────────────────────────────────────────────┐
│                        POST /v1/optimize                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐     ┌───────────────┐     ┌────────────────────┐  │
│  │  1. DEMAND    │────▶│  2. SIMULATE   │────▶│  3. ORDER QTY     │  │
│  │  FORECAST     │     │  (Monte Carlo)│     │  CALCULATION      │  │
│  └──────────────┘     └───────────────┘     └────────────────────┘  │
│         │                     │                      │               │
│    ┌────┴────┐          calc_opti_psl_3     max(0, psl - avail     │
│    │         │                │              - on_order + back)     │
│  basic    premium        optimal_psl                                   │
│    │         │                │                      │               │
│    ▼         ▼                ▼                      ▼               │
│  final_avg  ads          expected_profit      recommended_order_qty  │
│  final_var  variance     avg_inventory                              │
│                         cube_usage                                  │
│                         profit_per_cube                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Step 1: Demand Forecast

For each item:
- **Basic tier:** Run `calculate_adjusted_demand()` → produces `final_avg` (ADS) and `final_var` (variance)
- **Premium tier:** Run `ml_predictor.predict()` → uses XGBoost/BayesianRidge to predict ADS and variance from historical features

Both paths output `(ads, variance, demand_source)`.

### Step 2: Monte Carlo Simulation

Run `calc_opti_psl_3(ads, variance, cost, sale_price, ...)` which:
1. Fits a negative binomial distribution to `(ads, variance)` using log-space `neg_bin_ln`
2. Simulates daily demand over the order cycle via `day_sim_3`
3. Sweeps candidate PSL values to find the one maximizing expected profit
4. Returns optimal PSL, expected profit, average inventory, and other metrics

### Step 3: Order Quantity Calculation

Translate the abstract PSL into a concrete order recommendation:

```
recommended_order_qty = max(0, optimal_psl - current_available - on_order_qty + back_order_qty)
```

If the current position already exceeds the optimal PSL, the recommendation is zero (don't order).

---

## Data Flow

```
Client Request
      │
      ▼
┌─────────────┐    API Key     ┌─────────────┐
│  FastAPI     │──Validation──▶│  api/auth.py │
│  (modal_app) │               └─────────────┘
│              │
│              │    Per Item:
│              │    ┌──────────────────────────────────┐
│              │    │ tier == "basic"?                  │
│              │    │   YES → demand/adjusted_demand.py │
│              │    │   NO  → demand/ml_predictor.py     │
│              │    └──────────┬───────────────────────┘
│              │               │ (ads, variance)
│              │               ▼
│              │    ┌──────────────────────────────────┐
│              │    │ simulation/psl_optimizer.py        │
│              │    │   calc_opti_psl_3()               │
│              │    │     ├── demand_dist.py (neg_bin_ln)│
│              │    │     └── day_sim.py (day_sim_3)     │
│              │    └──────────┬───────────────────────┘
│              │               │ (optimal_psl, metrics)
│              │               ▼
│              │    recommended_order_qty calculation
│              │               │
│              │               ▼
│              │    Response JSON
└─────────────┘
```

---

## File Structure

```
optistock-api/
├── ARCHITECTURE.md          ← This file
├── modal_app.py             # FastAPI app deployed on Modal
├── simulation/
│   ├── __init__.py
│   ├── demand_dist.py       # neg_bin_ln, calc_nb_array_ln, numba_choice
│   ├── day_sim.py           # day_sim_3
│   └── psl_optimizer.py     # calc_opti_psl_3, get_all_psls, calc_single_psl
├── demand/
│   ├── adjusted_demand.py   # calculate_adjusted_demand (cleaned up)
│   └── ml_predictor.py      # Wrapper for ml-regression predictions
├── api/
│   ├── schemas.py           # Pydantic models
│   ├── auth.py              # API key validation
│   └── routes.py            # FastAPI route handlers
├── tests/
│   └── test_simulation.py
├── pyproject.toml
└── README.md
```

### Module Responsibilities

| Module | Key Functions | Purpose |
|--------|--------------|---------|
| `modal_app.py` | FastAPI app, Modal decorators | Entry point; deploys to Modal |
| `simulation/demand_dist.py` | `neg_bin_ln`, `calc_nb_array_ln`, `numba_choice` | Negative binomial distribution in log-space; Numba-accelerated sampling |
| `simulation/day_sim.py` | `day_sim_3` | Simulates daily demand over an order cycle |
| `simulation/psl_optimizer.py` | `calc_opti_psl_3`, `get_all_psls`, `calc_single_psl` | Finds optimal PSL by sweeping candidate levels; `get_all_psls` parallelizes across items |
| `demand/adjusted_demand.py` | `calculate_adjusted_demand` | Moving-average demand forecaster → `final_avg`, `final_var` |
| `demand/ml_predictor.py` | `predict` | Wrapper calling ml-regression (XGBoost/BayesianRidge) for premium tier |
| `api/schemas.py` | Pydantic models | Request/response validation |
| `api/auth.py` | API key validation | Ensures valid `X-API-Key` header |
| `api/routes.py` | Route handlers | Maps endpoints to pipeline logic |

---

## Numba + Modal Considerations

- **JIT warmup:** All `@njit` functions are slow on first call (~2–5s each). Cold starts on Modal will incur this penalty.
- **Batch efficiency:** Use `get_all_psls()` to parallelize across items in a single request, amortizing warmup cost.
- **Modal image:** Python 3.11 + numpy + numba + pandas + xgboost + fastapi.
- **Warming strategy:** Consider a `/warmup` endpoint or Modal lifecycle hook to trigger JIT compilation before production traffic arrives.

---

## Cost Model

The simulation optimizes **expected profit**, which accounts for:

- **Revenue:** `sale_price × expected_units_sold`
- **Purchase cost:** `cost × expected_units_purchased`
- **Holding cost:** `cost_of_capital_annual / 365 × cost × avg_inventory × days`
- **Payment float:** The gap between `payment_terms_days` and `sales_terms_days` affects cash flow timing

The `cost_of_capital_annual` input (default 0.14) is converted to a daily rate internally:

```python
invst_charge = cost_of_capital_annual / 365
```
