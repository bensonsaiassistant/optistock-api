# OptiStock API

Profit-maximizing inventory purchase optimization, deployed on [Modal](https://modal.com).

## What It Does

OptiStock answers one question: **how many units of each item should I order right now?**

It combines demand forecasting with Monte Carlo simulation to find the optimal inventory level (PSL — Periodic Stock Level) for each item, then translates that into a concrete order quantity based on your current stock position.

## Quick Start

### Local Development

```bash
cd /home/node/workspace/projects/optistock-api
pip install -e .
OPTISTOCK_API_KEYS="" uvicorn modal_app:fastapi_app --reload
```

Visit `http://localhost:8000/health` to confirm it's running.

### Deploy to Modal

```bash
export OPTISTOCK_API_KEYS="your-secret-key-here"
modal deploy modal_app.py
```

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/optimize` | Full pipeline: demand → simulate → order recommendation |
| `POST` | `/v1/demand` | Demand prediction only (returns ADS/variance per item) |
| `GET` | `/health` | Liveness check |

### Authentication

All endpoints require an `X-API-Key` header. Set valid keys via the `OPTISTOCK_API_KEYS` environment variable (comma-separated). If empty, all requests are allowed (dev mode).

## Architecture

```
Request → Demand Forecast → Monte Carlo Simulation → Order Recommendation
```

### Pipeline

1. **Demand Forecasting** — Calculates average daily sales (ADS) and variance from historical data. Basic tier uses moving averages; premium tier uses ML regression (XGBoost).

2. **Simulation** — Uses Numba-accelerated Monte Carlo simulation with log-space negative binomial distributions to find the PSL that maximizes expected profit.

3. **Order Calculation** — Translates the abstract PSL into a concrete order quantity:
   ```
   recommended_order_qty = max(0, optimal_psl - current_available - on_order_qty + back_order_qty)
   ```

### Key Files

| Path | Purpose |
|------|---------|
| `modal_app.py` | Modal deployment entry point |
| `api/routes.py` | FastAPI route handlers |
| `api/schemas.py` | Pydantic request/response models |
| `api/auth.py` | API key validation |
| `simulation/psl_optimizer.py` | PSL optimization engine |
| `simulation/demand_dist.py` | Negative binomial distribution (Numba JIT) |
| `simulation/day_sim.py` | Daily demand simulation |
| `demand/adjusted_demand.py` | Demand forecasting (stub — moving average) |
