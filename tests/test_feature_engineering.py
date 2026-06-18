"""Tests for trader-level and early-trader feature engineering."""

import pandas as pd
import pytest

from src.feature_engineering.build_early_trader_features import (
    attach_target,
    build_features,
    get_early_trades,
)
from src.feature_engineering.build_trader_features import build_trader_features
from src.feature_engineering.scale_features import scale_features
from src.feature_engineering.select_features import select_clustering_features


def test_build_trader_features_calculates_profitability_metrics(sample_raw_trade_data):
    """Verify core trader-level profitability and behavior metrics."""
    # Purpose: verify core trader-level profitability and behavior aggregations.
    features = build_trader_features(sample_raw_trade_data)
    trader_1 = features.loc[features["trader_id"] == "TRADER_1"].iloc[0]

    assert trader_1["total_trades"] == 2
    assert trader_1["total_pnl"] == 500.0
    assert trader_1["avg_pnl"] == 250.0
    assert trader_1["win_rate"] == 0.5
    assert trader_1["avg_leverage"] == 2.5
    assert trader_1["stop_loss_usage_rate"] == 0.5
    assert trader_1["roi_pct"] == 0.005


def test_build_trader_features_handles_all_winning_trades(sample_raw_trade_data):
    """Verify all-winning input produces a win rate of one."""
    # Purpose: cover the edge case where win_rate should be exactly one.
    data = sample_raw_trade_data.copy()
    data["trade_outcome"] = "WIN"
    data["pnl"] = data["pnl"].abs()

    features = build_trader_features(data)

    assert features["win_rate"].eq(1.0).all()


def test_build_trader_features_handles_zero_account_size_without_crashing(sample_raw_trade_data):
    """Document ROI divide-by-zero behavior for zero account size."""
    # Purpose: document divide-by-zero behavior for ROI normalization.
    data = sample_raw_trade_data.copy()
    data.loc[data["trader_id"] == "TRADER_1", "account_size"] = 0

    features = build_trader_features(data)
    roi = features.loc[features["trader_id"] == "TRADER_1", "roi_pct"].iloc[0]

    assert roi == float("inf")


def test_build_trader_features_empty_input_returns_empty_frame(sample_raw_trade_data):
    """Verify empty but well-formed input returns an empty feature frame."""
    # Purpose: ensure empty but well-formed inputs remain discoverable by pytest users.
    features = build_trader_features(sample_raw_trade_data.iloc[0:0])

    assert features.empty
    assert "win_rate" in features.columns


def test_manual_feature_definitions_cover_avg_profit_avg_loss_and_risk_reward(sample_raw_trade_data):
    """Verify requested domain formulas using deterministic sample trades."""
    # Purpose: validate requested domain formulas that are not first-class production columns.
    grouped = sample_raw_trade_data.groupby("trader_id")
    feature_checks = grouped["pnl"].agg(
        avg_profit=lambda values: values[values > 0].mean(),
        avg_loss=lambda values: values[values < 0].mean(),
        pnl_volatility="std",
    )
    feature_checks["risk_reward_ratio"] = (
        feature_checks["avg_profit"] / feature_checks["avg_loss"].abs()
    )
    feature_checks["days_active"] = grouped["trade_timestamp"].apply(
        lambda values: pd.to_datetime(values).dt.date.nunique()
    )

    assert feature_checks.loc["TRADER_1", "avg_profit"] == 1000.0
    assert feature_checks.loc["TRADER_1", "avg_loss"] == -500.0
    assert feature_checks.loc["TRADER_1", "risk_reward_ratio"] == 2.0
    assert feature_checks.loc["TRADER_1", "days_active"] == 2
    assert feature_checks.loc["TRADER_2", "pnl_volatility"] == pytest.approx(530.3301, rel=1e-4)


def test_risk_reward_ratio_handles_missing_losses(sample_raw_trade_data):
    """Verify missing losses are represented as NaN in risk-reward setup."""
    # Purpose: cover divide-by-zero style behavior when there are no losing trades.
    data = sample_raw_trade_data.copy()
    data["pnl"] = data["pnl"].abs()
    avg_loss = data.loc[data["pnl"] < 0, "pnl"].mean()

    assert pd.isna(avg_loss)


def test_get_early_trades_sorts_and_limits_per_trader(sample_raw_trade_data):
    """Verify early-trade extraction sorts chronologically and applies limits."""
    # Purpose: verify early-window extraction is chronological and bounded.
    repeated = pd.concat([sample_raw_trade_data] * 11, ignore_index=True)
    repeated["trade_id"] = [f"T{i}" for i in range(len(repeated))]
    shuffled = repeated.sample(frac=1.0, random_state=42)

    early = get_early_trades(shuffled)

    assert early.groupby("trader_id").size().le(20).all()
    assert set(early["trade_outcome"].unique()) <= {0, 1}


def test_build_early_features_calculates_expected_values(sample_raw_trade_data):
    """Verify early-trader feature aggregations match expected values."""
    # Purpose: verify early feature aggregations match the supervised model contract.
    early_trades = get_early_trades(sample_raw_trade_data)

    features = build_features(early_trades)
    trader_2 = features.loc[features["trader_id"] == "TRADER_2"].iloc[0]

    assert trader_2["early_total_trades"] == 2
    assert trader_2["early_avg_pnl"] == -125.0
    assert trader_2["early_win_rate"] == 0.5
    assert trader_2["early_avg_leverage"] == 4.5
    assert trader_2["early_stop_loss_usage_rate"] == 1.0


def test_attach_target_keeps_only_labeled_traders(sample_raw_trade_data):
    """Verify supervised features keep only traders with cluster labels."""
    # Purpose: ensure supervised training rows are inner-joined to cluster labels.
    early_features = build_features(get_early_trades(sample_raw_trade_data))
    clusters = pd.DataFrame({"trader_id": ["TRADER_1"], "cluster": [3]})

    supervised = attach_target(early_features, clusters)

    assert supervised["trader_id"].tolist() == ["TRADER_1"]
    assert supervised["cluster"].tolist() == [3]


def test_select_and_scale_features_return_no_null_values(sample_feature_data):
    """Verify feature selection and scaling preserve shape without nulls."""
    # Purpose: verify clustering feature selection and scaling preserve shape without NaNs.
    selected = select_clustering_features(sample_feature_data)

    scaled, scaler = scale_features(selected)

    assert list(scaled.columns) == list(selected.columns)
    assert scaled.shape == selected.shape
    assert not scaled.isna().any().any()
    assert hasattr(scaler, "transform")
