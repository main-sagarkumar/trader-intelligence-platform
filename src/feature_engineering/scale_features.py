import pandas as pd

import joblib

from sklearn.preprocessing import StandardScaler

from configs.paths_config import FEATURE_STORE_FILE, FEATURE_STORE_DIR, MODEL_DIR


def select_clustering_features(df):

    # Final feature set used for clustering
    from configs.model_config import CLUSTERING_FEATURES

    return df[CLUSTERING_FEATURES].copy()


def scale_features(df):

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