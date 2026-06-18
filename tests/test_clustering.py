"""Tests for clustering, PCA projection, and KMeans artifact persistence."""

import joblib
from sklearn.decomposition import PCA

from configs.model_config import OPTIMAL_CLUSTERS
from src.clustering.kmeans_clustering import (
    cluster_feature_profile,
    cluster_persona_comparison,
    train_kmeans,
)
from src.feature_engineering.scale_features import scale_features
from src.feature_engineering.select_features import select_clustering_features


def test_train_kmeans_generates_expected_cluster_labels(sample_feature_data):
    """Verify KMeans produces one valid cluster label per trader."""
    # Purpose: verify the clustering pipeline produces one label per trader.
    selected = select_clustering_features(sample_feature_data)
    scaled, _ = scale_features(selected)

    model, labels = train_kmeans(scaled)

    assert len(labels) == len(sample_feature_data)
    assert model.n_clusters == OPTIMAL_CLUSTERS
    assert set(labels).issubset(set(range(OPTIMAL_CLUSTERS)))


def test_cluster_count_matches_configuration(sample_feature_data):
    """Verify the trained KMeans cluster count matches configuration."""
    # Purpose: keep KMeans configuration aligned with the shared model config.
    selected = select_clustering_features(sample_feature_data)
    scaled, _ = scale_features(selected)

    model, _ = train_kmeans(scaled)

    assert model.n_clusters == OPTIMAL_CLUSTERS


def test_pca_output_dimensions_are_correct(sample_feature_data):
    """Verify PCA projection produces two-dimensional analytics output."""
    # Purpose: validate dimensionality reduction output expected by clustering analysis.
    selected = select_clustering_features(sample_feature_data)
    scaled, _ = scale_features(selected)
    pca = PCA(n_components=2, random_state=42)

    transformed = pca.fit_transform(scaled)

    assert transformed.shape == (len(sample_feature_data), 2)


def test_scaled_features_have_no_nan_values(sample_feature_data):
    """Verify scaled clustering features do not contain NaN values."""
    # Purpose: ensure clustering never receives missing transformed feature values.
    selected = select_clustering_features(sample_feature_data)
    scaled, _ = scale_features(selected)

    assert not scaled.isna().any().any()


def test_clustered_dataframe_contains_cluster_assignment(sample_feature_data):
    """Verify cluster labels can be attached to trader feature rows."""
    # Purpose: verify labels can be attached back to trader records.
    selected = select_clustering_features(sample_feature_data)
    scaled, _ = scale_features(selected)
    _, labels = train_kmeans(scaled)

    clustered = sample_feature_data.copy()
    clustered["cluster"] = labels

    assert "cluster" in clustered.columns
    assert len(clustered["cluster"]) == len(sample_feature_data)


def test_kmeans_model_serialization_round_trip(sample_feature_data, temp_output_directory):
    """Verify a fitted KMeans model can be saved and reloaded."""
    # Purpose: verify trained clustering models can be persisted and reloaded.
    selected = select_clustering_features(sample_feature_data)
    scaled, _ = scale_features(selected)
    model, labels = train_kmeans(scaled)
    model_file = temp_output_directory / "kmeans_model.pkl"

    joblib.dump(model, model_file)
    loaded_model = joblib.load(model_file)

    assert loaded_model.n_clusters == model.n_clusters
    assert loaded_model.predict(scaled).shape == labels.shape


def test_cluster_profile_and_persona_comparison_shapes(sample_feature_data):
    """Verify cluster reporting helpers return expected grouped outputs."""
    # Purpose: cover clustering reporting helpers with deterministic grouped data.
    profile = cluster_feature_profile(sample_feature_data)
    comparison = cluster_persona_comparison(sample_feature_data)

    assert profile.index.name == "cluster"
    assert set(["total_trades", "avg_pnl", "roi_pct", "avg_holding_minutes"]).issubset(profile.columns)
    assert comparison.index.name == "cluster"
