"""
Map trader clusters to business recommendations.

This module translates numeric model outputs into segment names, explanations,
and actionable recommendations for product and risk teams.
"""

from src.clustering.cluster_definitions import (
    CLUSTER_MAPPING,
    CLUSTER_DESCRIPTIONS,
    CLUSTER_RECOMMENDATIONS
)


def get_trader_recommendations(cluster):
    """
    Return business metadata and recommendations for a cluster.

    Args:
        cluster: Numeric cluster ID produced by a segmentation model.

    Returns:
        Dictionary containing segment name, description, and recommendations.
    """

    return {
        "segment": CLUSTER_MAPPING[cluster],
        "description": CLUSTER_DESCRIPTIONS[cluster],
        "recommendations": CLUSTER_RECOMMENDATIONS[cluster]
    }


if __name__ == "__main__":

    cluster = 4

    insights = get_trader_recommendations(
        cluster
    )

    print("\n" + "=" * 80)
    print("TRADER INSIGHTS")
    print("=" * 80)

    print(
        f"\nSegment: "
        f"{insights['segment']}"
    )

    print(
        f"\nDescription:\n"
        f"{insights['description']}"
    )

    print("\nRecommendations:")

    for recommendation in insights[
        "recommendations"
    ]:
        print(
            f"• {recommendation}"
        )
