"""
API Routes
----------
REST endpoints for trader intelligence.
"""

from fastapi import APIRouter

from src.api.schemas import (
    CurrentTraderRequest, FutureTraderRequest, 
    CurrentTraderResponse, FutureTraderResponse,)

from src.pipelines.inference_pipeline import (predict_trader_profile,)

from src.pipelines.future_prediction_pipeline import (predict_future_trader_profile,)


router = APIRouter()


# ──────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────

@router.get("/health", tags=["System"],)
def health_check():
    """
    Return API service health status.

    Returns:
        Dictionary confirming the service is healthy.
    """
    return {
        "status": "healthy"
    }

@router.get("/", tags=["System"],)
def home():
    """
    API root endpoint.
    """
    return {
        "application": "Trader Intelligence Platform",
        "version": "1.0.0",
        "status": "running"
    }


# ──────────────────────────────────────────────
# Current Trader Intelligence
# ──────────────────────────────────────────────

@router.post(
    "/predict-current-segment",
    response_model=CurrentTraderResponse,
    tags=["Current Trader Intelligence"],
    )
def predict_current_segment(
    request: CurrentTraderRequest,
):
    """
    Predict current trader segment.
    """
    return predict_trader_profile(
        request.model_dump()
    )


# ──────────────────────────────────────────────
# Future Trader Intelligence
# ──────────────────────────────────────────────

@router.post(
    "/predict-future-segment",
    response_model=FutureTraderResponse,
    tags=["Future Trader Intelligence"],
    )
def predict_future_segment(
    request: FutureTraderRequest,
):
    """
    Predict future trader segment.
    """
    return predict_future_trader_profile(
        request.model_dump()
    )
