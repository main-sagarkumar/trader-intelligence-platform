"""
Generate business-readable insights from trader analytics data.

This module converts aggregate cluster and persona statistics into short
dashboard insight statements for non-technical users.
"""


def generate_insights(df):
    """
    Create automated insight messages from trader feature data.

    Args:
        df: Trader-level DataFrame containing cluster, persona, ROI,
            and leverage columns.

    Returns:
        List of human-readable insight strings.
    """

    insights = []

    # Compare average ROI by cluster to identify strongest and weakest segments.
    cluster_roi = (
        df.groupby("cluster")["roi_pct"]
        .mean()
    )

    best_cluster = cluster_roi.idxmax()
    worst_cluster = cluster_roi.idxmin()

    insights.append(
        f"🏆 Cluster {best_cluster} delivers the highest average ROI."
    )

    insights.append(
        f"⚠️ Cluster {worst_cluster} shows the weakest profitability."
    )

    # Persona-level ROI gives a business label for the best-performing behavior.
    best_persona = (
        df.groupby("persona")["roi_pct"]
        .mean()
        .idxmax()
    )

    insights.append(
        f"📈 {best_persona} traders generate the strongest average returns."
    )

    # Correlation highlights whether leverage is helping or hurting returns.
    leverage_corr = (
        df["avg_leverage"]
        .corr(df["roi_pct"])
    )

    insights.append(
        f"📊 Leverage-to-ROI correlation: {leverage_corr:.2f}"
    )

    return insights
