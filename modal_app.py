"""OptiStock API — deployed on Modal.

Cost-optimized configuration:
- Scales to zero when idle (no fixed cost)
- Tight timeout (30s — requests never need more)
- Small CPU/memory (basic tier is ~125ms, premium/elite ~4s)
- Concurrency limits to prevent runaway scaling
- Source code baked into the image (no mount issues)
"""

import modal
from contextlib import asynccontextmanager
from fastapi import FastAPI
import pathlib

app = modal.App("optistock-api")

# ── Project root ──────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).parent

# ── Image definition ──────────────────────────────────────────────────────
# Build image with dependencies + project source code baked in.
# add_local_python_source copies Python packages into the image.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1")  # Needed for xgboost
    .pip_install(
        # Core runtime dependencies
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "numba>=0.57.0",
        "aiosqlite>=0.19.0",
        # ML tier dependencies
        "xgboost>=2.0.0",
        "scikit-learn>=1.3.0",
        "polars>=0.20.0",
        # Modal itself
        "modal>=0.60.0",
    )
    # Bake the project's Python packages into the image
    # This copies api/, demand/, simulation/, scripts/ as importable modules
    .add_local_python_source(
        "api", "demand", "simulation", "scripts",
        copy=True,
    )
)


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
    # Cost optimization:
    cpu=1.0,              # 1 vCPU is plenty (sim runs in ~125ms for basic)
    memory=1024,          # 1 GB RAM (reduced from 2 GB)
    timeout=30,           # 30s max (requests never need 5 minutes)
    min_containers=0,     # Scale to zero when idle (no fixed cost)
    max_containers=4,     # Max 4 concurrent containers (prevent runaway scaling)
    # ML models need to import ml-regression — mount it as a volume
    volumes={
        "/ml-regression": modal.Volume.from_name("ml-regression-vol", create_if_missing=True),
    },
)
@modal.asgi_app()
def api_server():
    """Serve the full FastAPI app on Modal.

    All routes (POST /v1/optimize, POST /v1/demand, POST /v1/simulate,
    GET /health, GET /v1/requests) are handled by the FastAPI app.
    """
    return fastapi_app
