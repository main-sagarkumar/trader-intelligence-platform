"""
Build cluster-level analytics for trader behavior segments.

This module aggregates trader feature data by cluster so dashboards can compare
segment profitability, risk, leverage, and population size.
"""


def get_cluster_summary(df):
    """
    Aggregate trader metrics at the cluster level.

    Args:
        df: Trader-level feature DataFrame with cluster assignments.

    Returns:
        DataFrame with one row per cluster and summary metrics.
    """

    return (
        df.groupby("cluster")
        .agg(
            trader_count=("trader_id", "count"),
            avg_roi=("roi_pct", "mean"),
            avg_win_rate=("win_rate", "mean"),
            avg_leverage=("avg_leverage", "mean"),
            avg_risk=("avg_risk_pct", "mean"),
            avg_pnl=("total_pnl", "mean")
        )
        .reset_index()
    )
