"""
Prediction Tests
----------------
Tests future trader prediction pipeline.
"""

from src.pipelines.future_prediction_pipeline import (
    predict_future_trader_profile,
)


# ──────────────────────────────────────────────
# Future Trader Prediction
# ──────────────────────────────────────────────

def test_future_prediction_pipeline():
    """
    Verify future prediction pipeline returns
    expected response structure.
    """

    trader_features = {
        "early_total_trades": 20,
        "early_avg_pnl": 1200,
        "early_win_rate": 0.70,
        "early_avg_holding_minutes": 1200,
        "early_avg_leverage": 2.0,
        "early_avg_risk_pct": 0.03,
        "early_stop_loss_usage_rate": 0.85,
        "early_overnight_position_rate": 0.80,
    }

    result = predict_future_trader_profile(
        trader_features
    )

    assert "cluster" in result
    assert "confidence" in result
    assert "segment" in result
    assert "description" in result
    assert "recommendations" in result