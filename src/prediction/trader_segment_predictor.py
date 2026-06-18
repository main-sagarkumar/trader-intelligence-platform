"""
Predict a current trader's behavioral segment.

This module loads the saved scaler and KMeans model, transforms incoming
trader features, and returns the predicted cluster ID.
"""

import joblib
import pandas as pd

from configs.paths_config import MODEL_DIR
from configs.model_config import CLUSTERING_FEATURES



def load_models():
    """
    Load persisted current-segment inference artifacts.

    Returns:
        Tuple containing fitted scaler and trained KMeans model.
    """

    # Load trained scaler
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    # Load trained KMeans model
    kmeans = joblib.load(MODEL_DIR / "kmeans_model.pkl")

    return scaler, kmeans


def predict_cluster(trader_features):
    """
    Predict the behavioral cluster for a trader feature payload.

    Args:
        trader_features: Dictionary containing configured clustering features.

    Returns:
        Integer cluster assignment.
    """

    scaler, kmeans = load_models()

    # Preserve training-time feature order before scaling and prediction.
    feature_df = pd.DataFrame([trader_features], columns=CLUSTERING_FEATURES)

    scaled_features = pd.DataFrame(scaler.transform(feature_df),
                                   columns=CLUSTERING_FEATURES)

    cluster = kmeans.predict(scaled_features)[0]

    return int(cluster)


if __name__ == "__main__":

    sample_trader = {
        "total_trades": 150,
        "avg_pnl": 500,
        "roi_pct": 0.08,
        "avg_holding_minutes": 2400,
        "avg_leverage": 2,
        "win_rate": 0.70
    }

    cluster = predict_cluster(
        sample_trader
    )

    print("\nPredicted Cluster:", cluster)
