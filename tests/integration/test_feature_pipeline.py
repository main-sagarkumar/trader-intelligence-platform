"""Integration tests for feature engineering pipeline stages."""

from src.feature_engineering.build_early_trader_features import (
    attach_target,
    build_features,
    get_early_trades,
)
from src.feature_engineering.build_trader_features import build_trader_features


def test_raw_data_to_feature_pipeline_preserves_trader_rows(sample_raw_trade_data):
    """Verify raw trades produce expected trader-level feature rows."""
    # Purpose: verify raw trade data can be transformed into trader-level features.
    features = build_trader_features(sample_raw_trade_data)

    assert len(features) == sample_raw_trade_data["trader_id"].nunique()
    assert {"trader_id", "win_rate", "avg_pnl", "roi_pct"}.issubset(features.columns)
    assert not features[["win_rate", "avg_pnl", "roi_pct"]].isna().any().any()


def test_early_feature_pipeline_attaches_cluster_target(sample_raw_trade_data):
    """Verify early-trader features can be joined to cluster targets."""
    # Purpose: verify early features can be joined to clustering labels for supervised learning.
    early = get_early_trades(sample_raw_trade_data)
    features = build_features(early)
    clustered = features[["trader_id"]].copy()
    clustered["cluster"] = [0, 1]

    supervised = attach_target(features, clustered)

    assert len(supervised) == 2
    assert "cluster" in supervised.columns
    assert not supervised.isna().any().any()
