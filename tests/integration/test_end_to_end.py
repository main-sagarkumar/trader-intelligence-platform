"""Integration tests for raw data through clustering-ready outputs."""

import pandas as pd
from sklearn.decomposition import PCA

from src.clustering.kmeans_clustering import train_kmeans
from src.feature_engineering.build_trader_features import build_trader_features
from src.feature_engineering.scale_features import scale_features
from src.feature_engineering.select_features import select_clustering_features


def test_raw_feature_clustering_prediction_ready_flow(sample_raw_trade_data):
    """Verify raw trade data can be transformed and clustered end to end."""
    # Purpose: verify raw data can move through feature engineering and clustering without null outputs.
    raw_batches = []
    for idx in range(5):
        batch = sample_raw_trade_data.copy()
        batch["trader_id"] = f"TRADER_{idx + 1}"
        batch["trade_id"] = [f"T{idx}_{row}" for row in range(len(batch))]
        batch["account_size"] = batch["account_size"] + idx * 10_000
        batch["pnl"] = batch["pnl"] + idx * 25
        batch["balance_after_trade"] = batch["balance_before_trade"] + batch["pnl"]
        batch["leverage_used"] = batch["leverage_used"] + idx * 0.25
        raw_batches.append(batch)
    raw_data = pd.concat(raw_batches, ignore_index=True)

    trader_features = build_trader_features(raw_data)
    selected = select_clustering_features(trader_features)
    scaled, _ = scale_features(selected)
    model, labels = train_kmeans(scaled)

    trader_features["cluster"] = labels

    assert len(trader_features) == raw_data[["trader_id", "persona"]].drop_duplicates().shape[0]
    assert "cluster" in trader_features.columns
    assert model.n_clusters == 5
    assert not trader_features.isna().any().any()


def test_full_transformed_features_support_two_dimensional_projection(sample_feature_data):
    """Verify transformed trader features support two-dimensional projection."""
    # Purpose: validate clustered features support analytics projection after transformation.
    selected = select_clustering_features(sample_feature_data)
    scaled, _ = scale_features(selected)
    projection = PCA(n_components=2, random_state=42).fit_transform(scaled)

    assert projection.shape == (len(sample_feature_data), 2)
    assert not selected.isna().any().any()
