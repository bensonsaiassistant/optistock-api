"""OptiStock API — deployed on Modal."""

import modal
from fastapi import FastAPI

# Define Modal image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "numba>=0.57.0",
        "xgboost>=2.0.0",
        "scikit-learn>=1.3.0",
        "modal>=0.60.0",
    )
)

app = modal.App("optistock-api")

# Create the FastAPI app
fastapi_app = FastAPI(title="OptiStock API", version="0.1.0")

# Import and mount routes
from api.routes import router

fastapi_app.include_router(router)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("optistock-api-keys")],
    cpu=2.0,
    memory=2048,
    timeout=300,
)
@modal.fastapi_endpoint(method="POST", label="optimize")
def optimize_endpoint(request):
    """POST /v1/optimize — Full pipeline optimization."""
    from api.routes import optimize

    return optimize(request)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("optistock-api-keys")],
    cpu=1.0,
    timeout=60,
)
@modal.fastapi_endpoint(method="POST", label="demand")
def demand_endpoint(request):
    """POST /v1/demand — Demand forecast only."""
    from api.routes import demand_forecast

    return demand_forecast(request)


@app.function(
    image=image,
    timeout=10,
)
@modal.fastapi_endpoint(method="GET", label="health")
def health_endpoint():
    """GET /health — Service health check."""
    from api.routes import health

    return health()
