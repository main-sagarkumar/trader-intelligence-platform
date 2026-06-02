"""
Trader Intelligence API
-----------------------
FastAPI application entry point.
"""

from fastapi import FastAPI

from src.api.routes import router


# ──────────────────────────────────────────────
# Application
# ──────────────────────────────────────────────

app = FastAPI(
    title="Trader Intelligence API",
    description="Trader segmentation and future behaviour prediction API",
    version="1.0.0",
)

app.include_router(router, prefix="/api/v1",)