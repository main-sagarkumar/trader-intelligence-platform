"""
Orchestrate current-trader segment inference.

This pipeline combines cluster prediction with business-facing segment
recommendations for API and dashboard consumers.
"""

from src.prediction.trader_segment_predictor import predict_cluster

from src.prediction.recommendation_engine import get_trader_recommendations


def predict_trader_profile(trader_features):
    """
    Predict a current trader profile from engineered behavior features.

    Args:
        trader_features: Dictionary containing current trader model features.

    Returns:
        Dictionary with cluster, segment description, and recommendations.
    """

    # Predict behavioral cluster
    cluster = predict_cluster(trader_features)

    # Get segment insights and recommendations
    trader_profile = (get_trader_recommendations(cluster))

    return {"cluster": cluster, **trader_profile}


if __name__ == "__main__":

    sample_trader = {
        "total_trades": 150,
        "avg_pnl": 500,
        "roi_pct": 0.08,
        "avg_holding_minutes": 2400,
        "avg_leverage": 2,
        "win_rate": 0.70
    }

    profile = predict_trader_profile(
        sample_trader
    )

    print("\n" + "=" * 80)
    print("TRADER INTELLIGENCE REPORT")
    print("=" * 80)

    print(f"\nCluster: "f"{profile['cluster']}")

    print(f"\nSegment: "f"{profile['segment']}")

    print(f"\nDescription:\n"
        f"{profile['description']}"
    )

    print("\nRecommendations:")

    for recommendation in profile["recommendations"]:
        print(
            f"• {recommendation}"
        )
