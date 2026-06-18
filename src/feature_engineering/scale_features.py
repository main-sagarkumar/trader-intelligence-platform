"""
Scale clustering features before KMeans segmentation.

This module selects configured behavior features, standardizes them to remove
magnitude bias, and saves the fitted scaler for inference.
"""

import pandas as pd

import joblib

from sklearn.preprocessing import StandardScaler

from configs.paths_config import FEATURE_STORE_FILE, FEATURE_STORE_DIR, MODEL_DIR


def select_clustering_features(df):
    """
    Select the configured feature columns used by the clustering model.

    Args:
        df: Trader-level feature DataFrame.

    Returns:
        DataFrame containing only clustering input features.
    """

    # Final feature set used for clustering
    from configs.model_config import CLUSTERING_FEATURES

    return df[CLUSTERING_FEATURES].copy()


def scale_features(df):
    """
    Standardize clustering features for distance-based modeling.

    Args:
        df: Unscaled clustering feature DataFrame.

    Returns:
        Tuple of scaled feature DataFrame and fitted StandardScaler.
    """

    # Standardize features to mean=0 and std=1
    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(df)

    scaled_df = pd.DataFrame(
        scaled_features,
        columns=df.columns
    )

    return scaled_df, scaler


if __name__ == "__main__":

    trader_features = pd.read_csv(FEATURE_STORE_FILE)

    clustering_df = select_clustering_features(trader_features)

    # Scale selected clustering features
    scaled_df, scaler = scale_features(clustering_df)

    # Save fitted scaler for future inference
    joblib.dump(
        scaler,
        MODEL_DIR / "scaler.pkl"
    )

    print("\n" + "=" * 80)
    print("SCALED FEATURE DATASET")
    print("=" * 80)

    print("\nShape:")
    print(scaled_df.shape)

    print("\nMeans:")
    print(scaled_df.mean().round(4))

    print("\nStandard Deviations:")
    print(scaled_df.std().round(4))

    scaled_df.to_csv(
        FEATURE_STORE_DIR / "scaled_trader_features.csv",
        index=False
    )

    print(
        "\nScaled features saved to:"
    )

    print(
        FEATURE_STORE_DIR /
        "scaled_trader_features.csv"
    )
