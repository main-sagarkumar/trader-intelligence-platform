"""
Cluster Visualization Module
-----------------------------
Generates visual diagnostics for trader clusters:
- Feature distribution boxplots per cluster
- Cluster profile heatmap
- Normalized cluster profile heatmap
- PCA 2D scatter plot
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from configs.paths_config import FEATURE_STORE_DIR, CLUSTERING_OUTPUT_DIR


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

CLUSTER_FEATURES = ["roi_pct", "win_rate", "avg_leverage"]

PROFILE_AGGREGATIONS = {
    "total_trades": "mean",
    "avg_pnl": "mean",
    "roi_pct": "mean",
    "avg_holding_minutes": "mean",
    "avg_leverage": "mean",
    "win_rate": "mean",
}


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

def load_clustered_data() -> pd.DataFrame:
    """Load the clustered trader dataset from the feature store."""
    return pd.read_csv(FEATURE_STORE_DIR / "clustered_traders.csv")


# ──────────────────────────────────────────────
# Plot Helpers
# ──────────────────────────────────────────────

def _save_and_show(filename: str) -> None:
    """Apply tight layout, save the current figure, and display it."""
    plt.tight_layout()
    plt.savefig(CLUSTERING_OUTPUT_DIR / filename)
    plt.show()


# ──────────────────────────────────────────────
# Visualization Functions
# ──────────────────────────────────────────────

def plot_cluster_distributions(df: pd.DataFrame) -> None:
    """
    Plot a boxplot for each feature in CLUSTER_FEATURES,
    grouped by cluster label.

    Args:
        df: Clustered trader DataFrame. Must contain a 'cluster' column
            and all columns listed in CLUSTER_FEATURES.
    """
    for feature in CLUSTER_FEATURES:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df, x="cluster", y=feature)
        plt.title(f"{feature} by Cluster")
        _save_and_show(f"{feature}_distribution.png")


def plot_cluster_profile_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-cluster mean statistics and render them as a heatmap.

    Args:
        df: Clustered trader DataFrame. Must contain a 'cluster' column
            and all keys in PROFILE_AGGREGATIONS.

    Returns:
        profile: DataFrame of mean feature values indexed by cluster.
    """
    # Aggregate mean values for each feature per cluster
    profile = df.groupby("cluster").agg(PROFILE_AGGREGATIONS).round(2)

    plt.figure(figsize=(10, 6))
    sns.heatmap(profile, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Cluster Profile Heatmap")
    _save_and_show("cluster_profile_heatmap.png")

    return profile


def plot_normalized_cluster_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-cluster mean statistics, normalize them using
    StandardScaler, and render the result as a heatmap.

    Normalization centres each feature around 0, making it easier to
    compare relative cluster behaviour across features with different
    scales (e.g. avg_pnl vs win_rate).

    Args:
        df: Clustered trader DataFrame. Must contain a 'cluster' column
            and all keys in PROFILE_AGGREGATIONS.

    Returns:
        normalized_profile: StandardScaler-normalized DataFrame of mean
            feature values indexed by cluster.
    """
    # Aggregate mean values for each feature per cluster
    cluster_profile = df.groupby("cluster").agg(PROFILE_AGGREGATIONS)

    # Normalize across clusters so features are on a comparable scale
    scaler = StandardScaler()
    normalized_profile = pd.DataFrame(
        scaler.fit_transform(cluster_profile),
        columns=cluster_profile.columns,
        index=cluster_profile.index,
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(normalized_profile, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Normalized Cluster Profile Heatmap")
    _save_and_show("normalized_cluster_profile_heatmap.png")

    return normalized_profile


def plot_pca_clusters() -> None:
    """
    Reduce scaled trader features to 2 principal components and
    plot the result, colour-coded by cluster assignment.

    Reads two CSVs from FEATURE_STORE_DIR:
        - scaled_trader_features.csv  →  input to PCA
        - clustered_traders.csv       →  cluster labels
    """
    # Load scaled features and cluster labels
    scaled_df = pd.read_csv(FEATURE_STORE_DIR / "scaled_trader_features.csv")
    clustered_df = pd.read_csv(FEATURE_STORE_DIR / "clustered_traders.csv")

    # Fit PCA and project features onto 2 components
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(scaled_df)

    pca_df = pd.DataFrame(pca_coords, columns=["PC1", "PC2"])
    pca_df["cluster"] = clustered_df["cluster"]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="cluster", palette="tab10")
    plt.title("PCA Cluster Visualization")
    _save_and_show("pca_cluster_visualization.png")


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    clustered_df = load_clustered_data()

    plot_cluster_distributions(clustered_df)

    profile = plot_cluster_profile_heatmap(clustered_df)
    print("\n" + "=" * 80)
    print("CLUSTER PROFILE")
    print("=" * 80)
    print(profile)

    normalized_profile = plot_normalized_cluster_heatmap(clustered_df)
    print("\n" + "=" * 80)
    print("NORMALIZED CLUSTER PROFILE")
    print("=" * 80)
    print(normalized_profile.round(2))

    plot_pca_clusters()

    print(f"\nPlots saved to: {CLUSTERING_OUTPUT_DIR}")