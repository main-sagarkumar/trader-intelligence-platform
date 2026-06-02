"""
Future Trader Prediction Pipeline
---------------------------------
Predicts a trader's future segment using
behavioural features derived from their
first 20 trades.

Pipeline:
1. Predict future segment using Random Forest
2. Retrieve segment profile and recommendations
3. Generate a business-friendly intelligence report
"""

from src.prediction.future_segment_predictor import predict_future_profile

from src.prediction.recommendation_engine import get_trader_recommendations


# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────

def predict_future_trader_profile(trader_features: dict,) -> dict:
    """
    Generate a future trader intelligence report.

    Args:
        trader_features:
            Early behavioural features derived
            from a trader's first 20 trades.

    Returns:
        Future trader profile containing:
        - Predicted cluster
        - Confidence score
        - Segment
        - Description
        - Recommendations
    """
    prediction = predict_future_profile(trader_features)

    cluster = prediction["cluster"]
    confidence = prediction["confidence"]

    profile = get_trader_recommendations(cluster)

    return {
        "cluster": cluster,
        "confidence": confidence,
        **profile,
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

    report = predict_future_trader_profile(sample_trader)

    print("\n" + "=" * 80)
    print("FUTURE TRADER INTELLIGENCE REPORT")
    print("=" * 80)

    print(f"\nCluster: {report['cluster']}")

    print(
        f"\nConfidence: "
        f"{report['confidence']:.2%}"
    )

    print(f"\nSegment:\n{report['segment']}")

    print(
        f"\nDescription:\n"
        f"{report['description']}"
    )

    print("\nRecommendations:")

    for recommendation in report["recommendations"]:
        print(f"• {recommendation}")