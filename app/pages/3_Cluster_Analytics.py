"""
Render cluster-level analytics in the Streamlit dashboard.

This page compares trader segments by profitability, win rate, leverage, and
risk behavior to support business interpretation of clusters.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

# ------------------------------
# Page Config
# ------------------------------

st.set_page_config(
    page_title="Cluster Analytics",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Cluster Analytics")

# ------------------------------
# Load Data
# ------------------------------

df = pd.read_csv(
    "data/feature_store/clustered_traders.csv"
)

st.sidebar.markdown("""
# 📈 Trader Intelligence

### Analytics Platform

Behavior Analytics  
Risk Analytics  
Trader Segmentation  
Business Intelligence
""")

# ------------------------------
# Cluster Summary
# ------------------------------

cluster_summary = (
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

st.subheader("Cluster Summary")

st.dataframe(
    cluster_summary,
    use_container_width=True
)

# ------------------------------
# Cluster Size
# ------------------------------

st.subheader("Cluster Distribution")

fig = px.bar(
    cluster_summary,
    x="cluster",
    y="trader_count",
    title="Number of Traders per Cluster"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------------
# Average ROI
# ------------------------------

st.subheader("Average ROI by Cluster")

fig = px.bar(
    cluster_summary,
    x="cluster",
    y="avg_roi",
    title="Cluster Profitability"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------------
# Average Win Rate
# ------------------------------

st.subheader("Average Win Rate by Cluster")

fig = px.bar(
    cluster_summary,
    x="cluster",
    y="avg_win_rate",
    title="Cluster Win Rate"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------------
# Risk vs ROI
# ------------------------------

st.subheader("Risk vs ROI")

fig = px.scatter(
    cluster_summary,
    x="avg_risk",
    y="avg_roi",
    size="trader_count",
    color="cluster",
    hover_name="cluster",
    title="Risk vs ROI by Cluster"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------------
# Top Cluster
# ------------------------------

best_cluster = (
    cluster_summary
    .sort_values("avg_roi", ascending=False)
    .iloc[0]
)

st.success(
    f"""
    Best Performing Cluster:
    Cluster {int(best_cluster['cluster'])}

    Average ROI: {best_cluster['avg_roi']:.2%}

    Average Win Rate: {best_cluster['avg_win_rate']:.2%}
    """
)
