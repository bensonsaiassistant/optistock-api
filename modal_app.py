"""OptiStock API — deployed on Modal."""

import modal
from contextlib import asynccontextmanager
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
        "aiosqlite>=0.19.0",
    )
)

app = modal.App("optistock-api")


def warmup_numba():
    """Pre-compile all Numba JIT functions on Modal cold start."""
    from scripts.warm_numba import warm_numba_functions
    results = warm_numba_functions()
    total_ms = sum(ms for _, ms in results)
    print(f"Numba warmup complete: {total_ms:.0f}ms total")


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """Initialize storage DB on startup, close on shutdown."""
    from api import storage
    await storage.init_db()

    # Warm up Numba JIT-compiled functions to avoid cold-start latency
    print("Warming up Numba JIT functions…")
    warmup_numba()

    yield
    await storage.close_db()


# Create the FastAPI app with lifespan hooks
fastapi_app = FastAPI(
    title="OptiStock API",
    version="0.1.0",
    lifespan=lifespan,
)

# Import and mount routes
from api.routes import router, register_security_handlers

register_security_handlers(fastapi_app)
fastapi_app.include_router(router)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("optistock-api-keys")],
    cpu=2.0,
    memory=2048,
    timeout=300,
)
@modal.asgi_app()
def api_server():
    """Serve the full FastAPI app on Modal.

    All routes (POST /v1/optimize, POST /v1/demand, POST /v1/simulate,
    GET /health, GET /v1/requests) are handled by the FastAPI app.
    """
    return fastapi_app
