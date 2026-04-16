# OptiStock API

**Profit-maximizing inventory purchase optimization.** Tell it what you sell, how it sells, and how much it costs — get back exactly how many units to order.

Deployed on [Modal](https://modal.com). Python + FastAPI + Numba + XGBoost.

---

## Quick Start

### 1. Local Development (5 minutes)

```bash
cd optistock-api
pip install -e .
OPTISTOCK_API_KEYS="" uvicorn modal_app:fastapi_app --reload --port 8000
```

### 2. Your First Request

```bash
curl -X POST http://localhost:8000/v1/optimize \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key" \
  -d '{
    "items": [{
      "item_id": "SKU-001",
      "cost": 3.50,
      "sale_price": 8.99,
      "current_available": 45,
      "on_order_qty": 20,
      "back_order_qty": 5,
      "lead_time_days": 14,
      "payment_terms_days": 30,
      "sales_terms_days": 30,
      "historical_data": [
        {"date": "2024-01-01", "quantity": 5, "available": 100},
        {"date": "2024-01-02", "quantity": 7, "available": 95},
        {"date": "2024-01-03", "quantity": 3, "available": 88}
      ]
    }],
    "tier": "basic",
    "cost_of_capital": 0.14
  }'
```

### 3. Response

```json
{
  "items": [{
    "item_id": "SKU-001",
    "optimal_outp": 112,
    "recommended_order_qty": 42,
    "expected_profit": 487.32,
    "expected_daily_sales": 5.2,
    "expected_avg_inventory": 67.5,
    "cube_usage": 67.5,
    "profit_per_cube": 7.22,
    "demand_source": "adjusted_demand",
    "ads": 5.2,
    "variance": 8.3,
    "warnings": []
  }],
  "compute_time_ms": 1350
}
```

---

## Three Tiers

| Tier | Demand Method | Lead Time | Min Data | Latency |
|------|--------------|-----------|----------|---------|
| **basic** | Rolling averages with outlier cleaning | Fixed (provided) | 7+ days | ~50ms |
| **premium** | ML regression (XGBoost) with deviation guardrails | Fixed (provided) | 60+ days | 1–3s |
| **elite** | ML regression for demand + ML regression for lead time | ML-predicted with variance | 60+ demand, 30+ lead time | 2–5s |

### How to Choose

- **Basic** → You're testing, or you have sparse data. Fast, reliable, always works.
- **Premium** → You have 2+ months of daily sales history. ML catches seasonal patterns, trends, and day-of-week effects that rolling averages miss.
- **Elite** → You also track historical lead times (actual delivery dates). The ML predicts both what you'll sell AND when it'll arrive, with uncertainty baked into the simulation.

---

## API Reference

### `POST /v1/optimize` — Full Optimization Pipeline

Demand forecast → Monte Carlo simulation → order recommendation.

**Request body:**
```json
{
  "items": [...],
  "tier": "basic | premium | elite",
  "cost_of_capital": 0.14
}
```

**Item fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `item_id` | string | ✅ | — | Unique identifier (max 128 chars, no SQL patterns) |
| `cost` | float | ✅ | — | Unit cost (≥ 0, ≤ $1M) |
| `sale_price` | float | ✅ | — | Unit sale price (≥ 0, ≤ $1M) |
| `current_available` | int | | 0 | Units currently in stock |
| `on_order_qty` | int | | 0 | Units already ordered, not yet received |
| `back_order_qty` | int | | 0 | Units customers are waiting for |
| `lead_time_days` | float | | 14.0 | Supplier lead time in days (elite tier can override with ML) |
| `order_frequency_days` | int | | 7 | How often you place orders |
| `payment_terms_days` | int | | 30 | Days until you pay your supplier |
| `sales_terms_days` | int | | 30 | Days until customers pay you |
| `length` | float | | 1.0 | Item length (feet) |
| `width` | float | | 1.0 | Item width (feet) |
| `height` | float | | 1.0 | Item height (feet) |
| `historical_data` | array | | [] | Daily sales history (see format below) |
| `historical_lead_times` | array | | [] | Past lead time observations in days (elite tier) |

**Historical data point format:**
```json
{"date": "2024-01-15", "quantity": 5.0, "available": 100.0, "mercury_order_quantity": 0.0}
```
- `date`: ISO 8601 string (2020–2035 range)
- `quantity`: Units sold that day
- `available`: Units in stock that day (0 = out of stock, used for demand adjustment)
- `mercury_order_quantity`: Reserved field (set to 0 if unused)

**Max limits:** 50 items per request, 10,000 historical data points per item.

### `POST /v1/demand` — Demand Forecast Only

Returns predicted average daily sales (ADS) and variance without running the full simulation. Useful for planning.

**Request body:** Same `items` array as `/v1/optimize`. Always uses basic tier (rolling averages).

### `POST /v1/simulate` — Direct Simulation

Skip demand forecasting entirely. Provide your own ADS and variance.

```json
{
  "ads": 5.2,
  "variance": 8.3,
  "lead_time_days": 14.0,
  "cost": 3.50,
  "sale_price": 8.99,
  "item_id": "SKU-001",
  "current_available": 45,
  "on_order_qty": 20,
  "back_order_qty": 5,
  "payment_terms_days": 30,
  "sales_terms_days": 30,
  "cost_of_capital": 0.14,
  "length": 1.0,
  "width": 1.0,
  "height": 1.0,
  "order_frequency_days": 7
}
```

### `GET /health` — Liveness Check

```json
{"status": "ok", "service": "optistock-api"}
```

### `GET /v1/requests` — Query Stored Requests (Admin)

```
GET /v1/requests?start_date=2024-01-01&end_date=2024-12-31&item_id=SKU-001&limit=100
```

Requires API key prefixed with `admin-`. Returns all stored optimization requests matching the filters.

---

## Authentication

All endpoints require `X-API-Key` header.

```bash
# Set your keys (comma-separated for multiple)
export OPTISTOCK_API_KEYS="sk-prod-abc123,sk-prod-def456"

# Dev mode (no auth required)
export OPTISTOCK_API_KEYS=""
```

**Security features:**
- API keys are SHA-256 hashed — never stored or compared in plaintext
- Constant-time comparison via `hmac.compare_digest` (prevents timing attacks)
- Rate limiting: 100 requests per 60-second window per key
- Input validation: NaN/Inf rejection, SQL injection detection, null byte filtering, bounded numeric ranges
- Request body size limit: 10 MB max
- Global exception handler: never leaks stack traces to clients
- All stored API keys hashed in SQLite database

---

## How the Optimization Works

### The Core Question

*"What's the inventory level that maximizes my profit, given uncertain demand and lead times?"*

### Step 1: Demand Forecasting

**Basic tier:** Calculates Average Daily Sales (ADS) using outlier-cleaned rolling windows (7/30/60/180/365/730 days). Cleans anomalies (days with 0 sales due to stockouts are flagged via the `available` field). Returns a variance estimate from historical variability.

**Premium/Elite tier:** Trains an XGBoost time-series model on your historical data. The model learns seasonal patterns, day-of-week effects, trends, and rolling window features. Results are validated against the basic tier baseline — if the ML prediction deviates more than 3x (demand) or 5x (variance), it falls back to the baseline automatically. This guardrail prevents wild predictions from sparse or noisy data.

**Elite tier additionally:** Trains a separate XGBoost model on your historical lead time observations to predict future lead times AND their variance.

### Step 2: Monte Carlo Simulation

The engine runs 100,000 days of simulated inventory operations:

1. **Demand** is sampled from a Negative Binomial distribution (parameterized by ADS and variance) — this captures the bursty, overdispersed nature of real sales better than Poisson
2. **Lead times** are sampled from a Gamma distribution (parameterized by predicted lead time and its variance for elite tier, or a hardcoded exponent for basic/premium)
3. **Financials** track accounts receivable (sales terms), accounts payable (payment terms), inventory holding cost (cost of capital), and devaluation (products losing value after 548 days)

### Step 3: OUTP Sweep

The optimizer tries every reasonable Order Up To Point (OUTP) — the target inventory level you're trying to maintain. For each OUTP, it runs the simulation and calculates:

```
profit = (daily_sales × gross_margin) − (avg_inventory × cost × capital_rate/365) − devaluation_loss
```

The OUTP that maximizes profit is selected. Early stopping kicks in when profits consistently decline.

### Step 4: Order Recommendation

```
recommended_order_qty = max(0, optimal_outp − current_available − on_order_qty + back_order_qty)
```

This accounts for what you already have, what's coming, and what's already promised to customers.

### Batch Processing

When you send 3+ items, the API automatically switches to parallel batch mode using Numba's `prange`. This is 3–10x faster per item than sequential processing.

---

## Data Requirements

### Minimum Data Points

| Tier | Historical Sales Data | Historical Lead Times |
|------|----------------------|----------------------|
| Basic | 7+ days | Not used |
| Premium | 60+ days | Not used |
| Elite | 60+ days | 30+ observations |

### What "Good" Data Looks Like

- **Daily granularity** — one row per day per item
- **Consistent date ranges** — gaps reduce model accuracy
- **Accurate `available` values** — 0 when out of stock, actual count when in stock
- **365+ days** — needed for ML variance computation (below 365 days, premium tier still works but uses residual-based variance)

### What Happens with Insufficient Data

- **Below 7 days:** Returns `demand_source: "insufficient_data"`, OUTP = 0
- **7–59 days (premium/elite):** Falls back to basic tier (rolling averages)
- **ML deviation guardrail breached:** Falls back to basic tier automatically
- **No historical lead times (elite):** Uses provided `lead_time_days` with default variance

---

## Deployment

### Modal (Production)

```bash
# 1. Install dependencies
pip install modal
modal setup

# 2. Configure API keys
export OPTISTOCK_API_KEYS="your-production-key,admin-your-admin-key"
modal secret create optistock-api-keys OPTISTOCK_API_KEYS=$OPTISTOCK_API_KEYS

# 3. Deploy
modal deploy modal_app.py
```

On cold start, Modal runs the Numba warmup script (~4s) to pre-compile all JIT functions. After warmup, single-item requests complete in 1–3s depending on tier.

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPTISTOCK_API_KEYS` | Comma-separated valid API keys. Empty = dev mode (no auth). Prefix `admin-` = admin access. |

### Cold Start Behavior

- **First request after deploy:** ~10s (Numba warmup + first compilation)
- **Subsequent requests:** 50ms–3s depending on tier and data volume
- **Modal keeps warm** for a period after activity — cold starts only after inactivity

---

## Performance Benchmarks

| Scenario | Data | Tier | Time | Notes |
|----------|------|------|------|-------|
| Demand only | 180 days | basic | ~50ms | Rolling averages |
| Demand only | 365 days | premium | 1–3s | ML XGBoost |
| Single item optimize | 180 days | basic | ~1.5s | 100K-day sim |
| Single item optimize | 365 days | premium | 2–4s | ML + sim |
| Single item optimize | 365 days + LT | elite | 3–6s | ML demand + ML LT |
| 3 items (batch) | 365 days | premium | ~4s | 1.3s/item (parallel) |
| 10 items (batch) | 365 days | premium | ~6.5s | 0.65s/item (parallel) |

---

## Security

OptiStack takes API security seriously:

1. **Input validation at every layer:**
   - Pydantic validators reject NaN, Inf, negative costs, oversized payloads
   - SQL injection patterns detected and blocked in string fields
   - Null bytes filtered from all string inputs
   - Numeric bounds enforced (costs ≤ $1M, dimensions ≤ 10,000ft, dates 2020–2035)

2. **Authentication:**
   - Keys hashed with SHA-256, compared with constant-time `hmac.compare_digest`
   - Rate limiting: 100 requests / 60s per key
   - Admin endpoints require `admin-` prefix

3. **Error handling:**
   - Global exception handler catches everything — no stack traces leak
   - Storage failures are fire-and-forget — never block API responses
   - Structured error responses: `{"error": "message", "code": "ERROR_CODE"}`

4. **Data storage:**
   - API keys stored as SHA-256 hashes in SQLite
   - All requests logged for audit and future model retraining
   - Admin-only access to historical data

---

## Project Structure

```
optistock-api/
├── api/
│   ├── routes.py          # FastAPI endpoints, batch routing
│   ├── schemas.py         # Pydantic models with input validation
│   ├── auth.py            # API key validation, rate limiting
│   └── storage.py         # SQLite request logging (fire-and-forget)
├── demand/
│   ├── adjusted_demand.py # Basic tier: rolling averages, outlier cleaning
│   ├── ml_predictor.py    # Premium/elite tier: XGBoost demand prediction
│   └── lt_predictor.py    # Elite tier: XGBoost lead time prediction
├── simulation/
│   ├── demand_dist.py     # Negative binomial + Poisson distributions (Numba)
│   ├── day_sim.py         # Daily Monte Carlo simulation (Numba JIT)
│   └── outp_optimizer.py   # OUTP sweep optimizer (Numba parallel)
├── scripts/
│   └── warm_numba.py      # Pre-compile Numba functions for cold start
├── tests/
│   ├── test_all.py        # Unit tests (30 tests)
│   └── test_performance.py # Integration + performance benchmarks
├── modal_app.py           # Modal deployment entry point
├── pyproject.toml         # Dependencies and project config
└── README.md              # You are here
```

---

## Running Tests

```bash
# Unit tests
python3 -m pytest tests/test_all.py -v

# Performance + integration tests
python3 tests/test_performance.py

# Interactive test client
python3 test_client.py
```

---

Private — all rights reserved.
