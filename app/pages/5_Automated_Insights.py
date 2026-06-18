"""
Render automated business insights in the Streamlit dashboard.

This page reads clustered trader features and converts aggregate analytics into
short insight cards for business users.
"""

import streamlit as st
import pandas as pd


def generate_insights(df):
    """
    Generate dashboard insight strings from clustered trader data.

    Args:
        df: Clustered trader feature DataFrame.

    Returns:
        List of insight strings summarizing ROI, persona, and leverage patterns.
    """

    insights = []

    # Identify the cluster with the strongest average profitability.
    best_cluster = (
        df.groupby("cluster")["roi_pct"]
        .mean()
        .idxmax()
    )

    best_roi = (
        df.groupby("cluster")["roi_pct"]
        .mean()
        .max()
    )

    insights.append(
        f"🏆 Cluster {best_cluster} delivers the highest average ROI ({best_roi:.2%})."
    )

    worst_cluster = (
        df.groupby("cluster")["roi_pct"]
        .mean()
        .idxmin()
    )

    insights.append(
        f"⚠️ Cluster {worst_cluster} shows the weakest profitability."
    )

    best_persona = (
        df.groupby("persona")["roi_pct"]
        .mean()
        .idxmax()
    )

    insights.append(
        f"📈 {best_persona} traders generate the strongest average returns."
    )

    # Use leverage-to-ROI correlation to explain risk behavior directionally.
    corr = df["avg_leverage"].corr(df["roi_pct"])

    if corr > 0:
        insights.append(
            f"📊 Leverage shows a positive correlation ({corr:.2f}) with ROI."
        )
    else:
        insights.append(
            f"📊 Higher leverage is associated with lower ROI ({corr:.2f})."
        )

    return insights

st.set_page_config(
    page_title="Automated Insights",
    page_icon="💡",
    layout="wide"
)

st.title("💡 Automated Insights")

df = pd.read_csv(
    "data/feature_store/clustered_traders.csv"
)

insights = generate_insights(df)

for insight in insights:

    st.markdown(
        f"""
        <div style="
        background:#1E293B;
        padding:20px;
        border-radius:15px;
        margin-bottom:15px;
        border-left:5px solid #3B82F6;
        ">
        {insight}
        </div>
        """,
        unsafe_allow_html=True
    )
