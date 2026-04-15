# OptiStock API

Profit-maximizing inventory purchase optimization, deployed on [Modal](https://modal.com).

## What It Does

OptiStock answers one question: **how many units of each item should I order right now?**

It combines demand forecasting with Monte Carlo simulation to find the optimal inventory level (PSL — Periodic Stock Level) for each item, then translates that into a concrete order quantity based on your current stock position.

## Quick Start

### Local Development

```bash
cd optistock-api
pip install -e .
OPTISTOCK_API_KEYS="" uvicorn modal_app:fastapi_app --reload
```

Visit `http://localhost:8000/health` to confirm it's running.

### Deploy to Modal

```bash
export OPTISTOCK_API_KEYS="your-secret-key-here"
modal secret create optistock-api-keys OPTISTOCK_API_KEYS=$OPTISTOCK_API_KEYS
modal deploy modal_app.py
```

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/optimize` | Full pipeline: demand → simulate → order recommendation |
| `POST` | `/v1/demand` | Demand prediction only (returns ADS/variance per item) |
| `POST` | `/v1/simulate` | Direct simulation (provide ads/var yourself) |
| `GET` | `/v1/requests` | Query stored requests (admin only) |
| `GET` | `/v1/requests/{id}` | Get single stored request (admin only) |
| `GET` | `/health` | Liveness check |

### Authentication

All endpoints require an `X-API-Key` header. Set valid keys via `OPTISTOCK_API_KEYS` env var (comma-separated). Empty = dev mode (allows all). Admin endpoints require key prefix `admin-`.

## Architecture

```
Request → Demand Forecast → Monte Carlo Simulation → Order Recommendation
                                    ↓
                            Request Storage (SQLite)
```

### Pipeline

1. **Demand Forecasting**
   - **Basic tier:** Adjusted demand with outlier-cleaned rolling averages (60/180/365/730 day windows)
   - **Premium tier:** ML regression via ml-regression library (XGBoost) — stubbed, ready for model integration

2. **Simulation** — Numba-accelerated Monte Carlo simulation with log-space negative binomial distributions
   - Batch mode (`get_all_psls`): Parallel across items via `prange` (3+ items auto-batch)
   - Single mode: Sequential optimization per item

3. **Order Calculation**
   ```
   recommended_order_qty = max(0, optimal_psl - current_available - on_order_qty + back_order_qty)
   ```

4. **Data Storage** — Every request logged to SQLite with SHA-256 hashed API keys for analysis and retraining

### Key Files

| Path | Purpose |
|------|---------|
| `modal_app.py` | Modal deployment entry point, Numba warmup, lifespan hooks |
| `api/routes.py` | FastAPI route handlers, batch/single routing |
| `api/schemas.py` | Pydantic request/response models |
| `api/auth.py` | API key validation |
| `api/storage.py` | SQLite request logging, admin query endpoints |
| `simulation/psl_optimizer.py` | PSL optimization engine (`calc_opti_psl_3`, `get_all_psls`) |
| `simulation/demand_dist.py` | Negative binomial + Poisson distributions (Numba JIT) |
| `simulation/day_sim.py` | Daily demand simulation with OOS tracking |
| `demand/adjusted_demand.py` | Adjusted demand forecasting (rolling windows, outlier cleaning) |
| `demand/ml_predictor.py` | ML regression predictor stub (premium tier) |
| `scripts/warm_numba.py` | Numba JIT warmup for cold-start elimination |

### Performance

- **Numba warmup:** ~10s on cold start (Modal lifespan handler handles this)
- **Single item:** ~1.3s (after warmup)
- **5 items (batch):** ~10.5s (parallel via `get_all_psls` with `prange`)

## License

Private — all rights reserved.
