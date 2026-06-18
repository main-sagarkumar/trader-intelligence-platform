"""
Evaluate candidate KMeans cluster counts for trader segmentation.

This module computes elbow and silhouette diagnostics to support selecting the
number of behavioral trader clusters used by the production clustering model.
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from configs.paths_config import (FEATURE_STORE_DIR, CLUSTERING_OUTPUT_DIR)


def load_scaled_features():
    """
    Load standardized clustering features from the feature store.

    Returns:
        DataFrame containing scaled trader behavior features.
    """

    # Load scaled feature dataset used for clustering
    return pd.read_csv(
        FEATURE_STORE_DIR / "scaled_trader_features.csv"
    )


def evaluate_clusters(df):
    """
    Evaluate KMeans quality metrics across candidate cluster counts.

    Args:
        df: Scaled clustering feature DataFrame.

    Returns:
        Tuple containing K values, inertia scores, and silhouette scores.
    """

    inertia_scores = []
    silhouette_scores = []

    k_values = range(2, 9)

    for k in k_values:

        # Train KMeans for current K
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        cluster_labels = kmeans.fit_predict(df)

        # Elbow metric
        inertia_scores.append(
            kmeans.inertia_
        )

        # Silhouette metric
        silhouette_scores.append(
            silhouette_score(
                df,
                cluster_labels
            )
        )

    return (
        k_values,
        inertia_scores,
        silhouette_scores
    )


def plot_elbow_curve(k_values, inertia_scores):
    """
    Save and display an elbow curve for KMeans inertia.

    Args:
        k_values: Candidate cluster counts.
        inertia_scores: KMeans inertia values for each K.
    """

    plt.figure(figsize=(8, 5))

    plt.plot(
        k_values,
        inertia_scores,
        marker="o"
    )

    plt.title("Elbow Curve")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")

    plt.tight_layout()

    plt.savefig(
        CLUSTERING_OUTPUT_DIR /
        "elbow_curve.png"
    )

    plt.show()


def plot_silhouette_scores(
    k_values,
    silhouette_scores
):
    """
    Save and display silhouette scores for candidate cluster counts.

    Args:
        k_values: Candidate cluster counts.
        silhouette_scores: Silhouette scores calculated for each K.
    """

    plt.figure(figsize=(8, 5))

    plt.plot(
        k_values,
        silhouette_scores,
        marker="o"
    )

    plt.title("Silhouette Scores")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")

    plt.tight_layout()

    plt.savefig(
        CLUSTERING_OUTPUT_DIR /
        "silhouette_scores.png"
    )

    plt.show()


if __name__ == "__main__":

    scaled_df = load_scaled_features()

    (
        k_values,
        inertia_scores,
        silhouette_scores
    ) = evaluate_clusters(
        scaled_df
    )

    results = pd.DataFrame({
        "k": list(k_values),
        "inertia": inertia_scores,
        "silhouette_score": silhouette_scores
    })

    print("\n" + "=" * 80)
    print("CLUSTER EVALUATION")
    print("=" * 80)

    print(results)

    plot_elbow_curve(
        k_values,
        inertia_scores
    )

    plot_silhouette_scores(
        k_values,
        silhouette_scores
    )

    print(
        f"\nPlots saved to: "
        f"{CLUSTERING_OUTPUT_DIR}"
    )
