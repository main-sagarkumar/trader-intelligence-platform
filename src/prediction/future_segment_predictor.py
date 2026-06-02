"""
Future Segment Predictor
------------------------
Predicts a trader's future segment using
behavioural features derived from their
first 20 trades.
"""

import joblib
import pandas as pd

from configs.model_config import EARLY_FEATURES
from configs.paths_config import MODEL_DIR


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

MODEL_FILE = MODEL_DIR / "segment_classifier.pkl"


# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────

def load_model():
    """
    Load trained segment classifier.

    Returns:
        Trained Random Forest model.
    """
    return joblib.load(MODEL_FILE)


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────

def predict_future_profile(
    trader_features: dict,
) -> dict:
    """
    Predict a trader's future segment and confidence.

    Args:
        trader_features:
            Early trader behavioural features.

    Returns:
        Dictionary containing predicted
        cluster and confidence score.
    """
    model = load_model()

    feature_df = pd.DataFrame([trader_features], columns=EARLY_FEATURES,)

    cluster = int(model.predict(feature_df)[0])

    probabilities = model.predict_proba(feature_df)[0]

    confidence = float(probabilities.max())

    return {
        "cluster": cluster,
        "confidence": round(confidence, 4),
    }


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":

    sample_trader = {
        "early_total_trades": 20,
        "early_avg_pnl": 1200,
        "early_win_rate": 0.70,
        "early_avg_holding_minutes": 1200,
        "early_avg_leverage": 2.0,
        "early_avg_risk_pct": 0.03,
        "early_stop_loss_usage_rate": 0.85,
        "early_overnight_position_rate": 0.80,
    }

    result = predict_future_profile(sample_trader)

    print("\n" + "=" * 80)
    print("FUTURE SEGMENT PREDICTION")
    print("=" * 80)

    print(result)