"""
Compute high-level dashboard KPIs for the Trader Intelligence Platform.

This module summarizes trader-level feature data into business metrics used
by the Streamlit executive dashboard and analytics views.
"""


def get_kpis(df):
    """
    Calculate platform-wide trader KPIs.

    Args:
        df: Trader-level feature DataFrame containing ROI, win rate, PnL,
            and leverage columns.

    Returns:
        Dictionary with aggregate KPI values for dashboard display.
    """

    return {
        "total_traders": len(df),
        "avg_roi": df["roi_pct"].mean(),
        "avg_win_rate": df["win_rate"].mean(),
        "total_pnl": df["total_pnl"].sum(),
        "avg_leverage": df["avg_leverage"].mean()
    }
