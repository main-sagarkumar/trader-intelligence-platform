"""
API Tests
---------
Tests Trader Intelligence API endpoints.
"""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


# ──────────────────────────────────────────────
# Health Endpoint
# ──────────────────────────────────────────────

def test_health_endpoint():
    # Verify API health endpoint.

    response = client.get("/api/v1/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy"
    }

def test_future_prediction_endpoint():
    # Verify future prediction endpoint.

    payload = {
        "early_total_trades": 20,
        "early_avg_pnl": 1200,
        "early_win_rate": 0.70,
        "early_avg_holding_minutes": 1200,
        "early_avg_leverage": 2.0,
        "early_avg_risk_pct": 0.03,
        "early_stop_loss_usage_rate": 0.85,
        "early_overnight_position_rate": 0.80,
    }

    response = client.post(
        "/api/v1/predict-future-segment",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "cluster" in data
    assert "confidence" in data
    assert "segment" in data
    assert "description" in data
    assert "recommendations" in data