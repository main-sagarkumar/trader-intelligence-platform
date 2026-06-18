"""Shared PyTest fixtures for deterministic platform tests."""

from pathlib import Path
import shutil
import uuid
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from src.api.main import app
from src.clustering.cluster_definitions import (
    CLUSTER_DESCRIPTIONS,
    CLUSTER_MAPPING,
    CLUSTER_RECOMMENDATIONS,
)


@pytest.fixture
def sample_raw_trade_data() -> pd.DataFrame:
    """Small deterministic trade-level dataset shared by feature and pipeline tests."""
    return pd.DataFrame(
        [
            {
                "trade_id": "T1",
                "trader_id": "TRADER_1",
                "persona": "disciplined_option_seller",
                "account_size": 100_000,
                "balance_before_trade": 100_000,
                "balance_after_trade": 101_000,
                "trade_timestamp": "2025-01-01 09:30:00",
                "strategy_type": "short_strangle",
                "leverage_used": 2.0,
                "holding_minutes": 60,
                "stop_loss_used": 1,
                "pnl": 1_000.0,
                "trade_outcome": "WIN",
                "overnight_position": 0,
                "capital_risked": 2_000,
                "risk_percentage": 0.02,
            },
            {
                "trade_id": "T2",
                "trader_id": "TRADER_1",
                "persona": "disciplined_option_seller",
                "account_size": 100_000,
                "balance_before_trade": 101_000,
                "balance_after_trade": 100_500,
                "trade_timestamp": "2025-01-02 10:00:00",
                "strategy_type": "short_strangle",
                "leverage_used": 3.0,
                "holding_minutes": 120,
                "stop_loss_used": 0,
                "pnl": -500.0,
                "trade_outcome": "LOSS",
                "overnight_position": 1,
                "capital_risked": 3_000,
                "risk_percentage": 0.03,
            },
            {
                "trade_id": "T3",
                "trader_id": "TRADER_2",
                "persona": "scalper_option_buyer",
                "account_size": 50_000,
                "balance_before_trade": 50_000,
                "balance_after_trade": 50_250,
                "trade_timestamp": "2025-01-01 09:45:00",
                "strategy_type": "option_buying",
                "leverage_used": 5.0,
                "holding_minutes": 30,
                "stop_loss_used": 1,
                "pnl": 250.0,
                "trade_outcome": "WIN",
                "overnight_position": 0,
                "capital_risked": 2_500,
                "risk_percentage": 0.05,
            },
            {
                "trade_id": "T4",
                "trader_id": "TRADER_2",
                "persona": "scalper_option_buyer",
                "account_size": 50_000,
                "balance_before_trade": 50_250,
                "balance_after_trade": 49_750,
                "trade_timestamp": "2025-01-03 14:15:00",
                "strategy_type": "option_buying",
                "leverage_used": 4.0,
                "holding_minutes": 45,
                "stop_loss_used": 1,
                "pnl": -500.0,
                "trade_outcome": "LOSS",
                "overnight_position": 0,
                "capital_risked": 2_000,
                "risk_percentage": 0.04,
            },
        ]
    )


@pytest.fixture
def sample_feature_data() -> pd.DataFrame:
    """Trader-level data with enough variation for clustering and analytics tests."""
    return pd.DataFrame(
        [
            {
                "trader_id": f"TRADER_{idx}",
                "persona": "persona_a" if idx % 2 else "persona_b",
                "account_size": 100_000 + idx * 10_000,
                "final_balance": 101_000 + idx * 10_000,
                "total_trades": 20 + idx * 5,
                "total_pnl": pnl,
                "avg_pnl": pnl / 10,
                "win_rate": win_rate,
                "avg_holding_minutes": 30 + idx * 15,
                "avg_leverage": leverage,
                "avg_risk_pct": risk,
                "stop_loss_usage_rate": min(1.0, 0.3 + idx * 0.1),
                "overnight_position_rate": min(1.0, idx * 0.05),
                "roi_pct": roi,
                "cluster": idx % 5,
            }
            for idx, (pnl, win_rate, leverage, risk, roi) in enumerate(
                [
                    (-600, 0.30, 8.0, 0.070, -0.06),
                    (400, 0.55, 3.0, 0.030, 0.04),
                    (900, 0.70, 2.0, 0.020, 0.09),
                    (150, 0.45, 5.0, 0.050, 0.015),
                    (700, 0.65, 2.5, 0.025, 0.07),
                    (-200, 0.40, 6.0, 0.060, -0.02),
                    (1_200, 0.80, 1.5, 0.015, 0.12),
                    (300, 0.50, 4.0, 0.040, 0.03),
                    (500, 0.60, 3.5, 0.035, 0.05),
                    (-400, 0.35, 7.0, 0.065, -0.04),
                ],
                start=1,
            )
        ]
    )


@pytest.fixture
def sample_early_feature_payload() -> dict[str, float]:
    """Valid future-segment request payload used by API and prediction tests."""
    return {
        "early_total_trades": 20,
        "early_avg_pnl": 1200.0,
        "early_win_rate": 0.70,
        "early_avg_holding_minutes": 1200.0,
        "early_avg_leverage": 2.0,
        "early_avg_risk_pct": 0.03,
        "early_stop_loss_usage_rate": 0.85,
        "early_overnight_position_rate": 0.80,
    }


@pytest.fixture
def sample_current_feature_payload() -> dict[str, float]:
    """Valid current-segment request payload used by API and inference tests."""
    return {
        "total_trades": 150,
        "avg_pnl": 500.0,
        "roi_pct": 0.08,
        "avg_holding_minutes": 2400.0,
        "avg_leverage": 2.0,
        "win_rate": 0.70,
    }


@pytest.fixture
def trained_dummy_model(sample_feature_data: pd.DataFrame) -> DummyClassifier:
    """Fast classifier fixture that supports predict and predict_proba without retraining project models."""
    from configs.model_config import EARLY_FEATURES

    x = pd.DataFrame(
        np.tile(np.arange(1, len(EARLY_FEATURES) + 1), (10, 1)),
        columns=EARLY_FEATURES,
    )
    y = sample_feature_data["cluster"].to_numpy()
    model = DummyClassifier(strategy="constant", constant=0)
    model.fit(x, y)
    return model


@pytest.fixture
def api_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """FastAPI client with expensive model-backed route dependencies mocked."""

    def current_response(_: dict[str, Any]) -> dict[str, Any]:
        """Return a deterministic current-segment API response."""
        cluster = 1
        return {
            "cluster": cluster,
            "segment": CLUSTER_MAPPING[cluster],
            "description": CLUSTER_DESCRIPTIONS[cluster],
            "recommendations": CLUSTER_RECOMMENDATIONS[cluster],
        }

    def future_response(_: dict[str, Any]) -> dict[str, Any]:
        """Return a deterministic future-segment API response."""
        cluster = 2
        return {
            "cluster": cluster,
            "confidence": 0.91,
            "segment": CLUSTER_MAPPING[cluster],
            "description": CLUSTER_DESCRIPTIONS[cluster],
            "recommendations": CLUSTER_RECOMMENDATIONS[cluster],
        }

    monkeypatch.setattr("src.api.routes.predict_trader_profile", current_response)
    monkeypatch.setattr("src.api.routes.predict_future_trader_profile", future_response)
    return TestClient(app)


@pytest.fixture
def workspace_tmp_dir() -> Path:
    """Workspace-local temporary directory for environments where OS temp is restricted."""
    tmp_root = Path.cwd() / "tests" / ".tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_root / uuid.uuid4().hex
    output_dir.mkdir()
    try:
        yield output_dir
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


@pytest.fixture
def temp_output_directory(workspace_tmp_dir: Path) -> Path:
    """Temporary output directory for tests that need filesystem persistence."""
    output_dir = workspace_tmp_dir / "outputs"
    output_dir.mkdir()
    return output_dir
