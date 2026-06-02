from src.clustering.cluster_definitions import (
    CLUSTER_MAPPING,
    CLUSTER_DESCRIPTIONS,
    CLUSTER_RECOMMENDATIONS
)


def get_trader_recommendations(cluster):

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