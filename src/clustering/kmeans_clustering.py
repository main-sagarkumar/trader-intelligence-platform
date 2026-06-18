"""
Train and profile the KMeans trader segmentation model.

This module loads engineered trader features, fits the configured clustering
model, saves model artifacts, and prints business-facing cluster summaries.
"""

import joblib
import pandas as pd

from sklearn.cluster import KMeans

from configs.paths_config import (
    FEATURE_STORE_FILE,
    FEATURE_STORE_DIR,
    MODEL_DIR
)

from configs.model_config import OPTIMAL_CLUSTERS


def load_data():
    """
    Load original and scaled trader feature datasets for clustering.

    Returns:
        Tuple of raw trader features and scaled clustering features.
    """

    # Load original trader features
    trader_features = pd.read_csv(FEATURE_STORE_FILE)

    # Load scaled features used for clustering
    scaled_features = pd.read_csv(
        FEATURE_STORE_DIR / "scaled_trader_features.csv"
    )

    return trader_features, scaled_features


def train_kmeans(scaled_features):
    """
    Fit the configured KMeans model on scaled trader behavior features.

    Args:
        scaled_features: Standardized feature matrix used for clustering.

    Returns:
        Tuple containing the fitted KMeans model and cluster labels.
    """

    # Optimal K selected using elbow and silhouette analysis
    kmeans = KMeans(
        n_clusters=OPTIMAL_CLUSTERS,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(
        scaled_features
    )

    return kmeans, cluster_labels


def cluster_summary(clustered_df):
    """
    Print the number of traders assigned to each cluster.

    Args:
        clustered_df: Trader-level DataFrame containing a cluster column.
    """

    # Number of traders in each cluster
    cluster_sizes = (
        clustered_df["cluster"]
        .value_counts()
        .sort_index()
    )

    print("\n" + "=" * 80)
    print("CLUSTER SIZES")
    print("=" * 80)

    print(cluster_sizes)


def cluster_feature_profile(clustered_df):
    """
    Build and print average behavioral metrics for each cluster.

    Args:
        clustered_df: Trader-level DataFrame containing features and clusters.

    Returns:
        DataFrame indexed by cluster with mean feature values.
    """

    # Average feature values per cluster
    profile = (
        clustered_df
        .groupby("cluster")
        .agg({
            "total_trades": "mean",
            "avg_pnl": "mean",
            "roi_pct": "mean",
            "avg_holding_minutes": "mean",
            "avg_leverage": "mean",
            "win_rate": "mean"
        })
        .round(2)
    )

    print("\n" + "=" * 80)
    print("CLUSTER FEATURE PROFILE")
    print("=" * 80)

    print(profile)

    return profile


def cluster_persona_comparison(clustered_df):
    """
    Compare discovered clusters against the synthetic source personas.

    Args:
        clustered_df: Trader-level DataFrame containing cluster and persona.

    Returns:
        Crosstab DataFrame showing persona counts per cluster.
    """

    # Compare discovered clusters with original personas
    comparison = pd.crosstab(
        clustered_df["cluster"],
        clustered_df["persona"]
    )

    print("\n" + "=" * 80)
    print("CLUSTER VS PERSONA")
    print("=" * 80)

    print(comparison)

    return comparison


if __name__ == "__main__":

    trader_features, scaled_features = load_data()

    kmeans, cluster_labels = train_kmeans(
        scaled_features
    )

    # Save trained model
    joblib.dump(
        kmeans,
        MODEL_DIR / "kmeans_model.pkl"
    )

    # Attach cluster assignments
    trader_features["cluster"] = cluster_labels

    # Save clustered dataset
    trader_features.to_csv(
        FEATURE_STORE_DIR / "clustered_traders.csv",
        index=False
    )

    print("\n" + "=" * 80)
    print("KMEANS CLUSTERING COMPLETE")
    print("=" * 80)

    print(
        f"\nModel Saved: "
        f"{MODEL_DIR / 'kmeans_model.pkl'}"
    )

    print(
        f"\nClustered Dataset Saved: "
        f"{FEATURE_STORE_DIR / 'clustered_traders.csv'}"
    )

    cluster_summary(trader_features)

    cluster_feature_profile(
        trader_features
    )

    cluster_persona_comparison(
        trader_features
    )
