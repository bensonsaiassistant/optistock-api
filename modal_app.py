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
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles
import pathlib

app = modal.App("optistock-api")

# ── Project root ──────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).parent
WEBUI_DIR = pathlib.Path(__file__).parent / "webui"

# ── Image definition ──────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1")  # Needed for xgboost
    .pip_install(
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "numba>=0.57.0",
        "aiosqlite>=0.19.0",
        "xgboost>=2.0.0",
        "scikit-learn>=1.3.0",
        "polars>=0.20.0",
        "stripe>=5.0.0",
        "modal>=0.60.0",
        # webui v2 - 2026-04-16
        "bcrypt>=4.0.0",
        "pyjwt>=2.0.0",
        "email-validator>=2.0.0",
    )
    .add_local_python_source(
        "api", "demand", "simulation", "scripts",
        copy=True,
    )
    .add_local_dir(
        WEBUI_DIR,
        remote_path="/root/webui",
        ignore=lambda x: False,  # force rebuild
    )
)

CONTAINER_WEBUI = pathlib.Path("/root/webui")


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
    print("Warming up Numba JIT functions…")
    warmup_numba()
    yield
    await storage.close_db()


fastapi_app = FastAPI(
    title="OptiStock API",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount API routes
from api.routes import router, register_security_handlers
from api.user_routes import router as user_router
from api.billing_routes import router as billing_router

register_security_handlers(fastapi_app)
fastapi_app.include_router(router)
fastapi_app.include_router(user_router)
fastapi_app.include_router(billing_router)

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("optistock-api-keys")],
    cpu=1.0,
    memory=1024,
    timeout=30,
    min_containers=0,
    max_containers=4,
    volumes={
        "/ml-regression": modal.Volume.from_name("ml-regression-vol", create_if_missing=True),
    },
)
@modal.asgi_app()
def api_server():
    """Serve the full FastAPI app on Modal."""
    # Mount web UI static files (deferred to container runtime)
    fastapi_app.mount("/static", StaticFiles(directory=str(CONTAINER_WEBUI)), name="static")
    @fastapi_app.get("/")
    async def serve_index():
        return FileResponse(str(CONTAINER_WEBUI / "index.html"))
    return fastapi_app
