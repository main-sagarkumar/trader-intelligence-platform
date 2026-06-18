"""
Render the Streamlit executive dashboard page.

This page summarizes trader intelligence KPIs and high-level business findings
from the clustered trader feature store.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from styles import load_css, metric_card

load_css()

CLUSTER_NAMES = {
    0: "🔥 High-Risk Gamblers",
    1: "🛡 Conservative Traders",
    2: "⚡ Aggressive Swing Traders",
    3: "🏆 Elite Performers",
    4: "💀 Capital Destroyers"
}


# ------------------------------
# Page Config
# ------------------------------
st.markdown("""
## 📋 Executive Summary

- Elite Performers generate the highest ROI.
- Conservative Traders maintain the highest win rates.
- High leverage clusters show significantly lower profitability.
- Risk-managed traders outperform aggressive traders.
""")

st.sidebar.markdown("""
# 📈 Trader Intelligence

### Analytics Platform

Behavior Analytics  
Risk Analytics  
Trader Segmentation  
Business Intelligence
""")

st.set_page_config(
    page_title="Executive Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Executive Dashboard")

# ------------------------------
# Load Data
# ------------------------------

df = pd.read_csv(
    "data/feature_store/clustered_traders.csv"
)

df["cluster_name"] = (
    df["cluster"]
    .map(CLUSTER_NAMES)
)


# ------------------------------
# KPI Cards
# ------------------------------

total_traders = len(df)
avg_roi = df["roi_pct"].mean()
avg_win_rate = df["win_rate"].mean()
total_pnl = df["total_pnl"].sum()

col1, col2, col3, col4 = st.columns(4)

with col1:
    metric_card(
        "Total Traders",
        f"{total_traders:,}"
    )

with col2:
    metric_card(
        "Average ROI",
        f"{avg_roi:.2%}"
    )

with col3:
    metric_card(
        "Win Rate",
        f"{avg_win_rate:.2%}"
    )

with col4:
    metric_card(
        "Total PnL",
        f"${total_pnl:,.0f}"
    )

st.markdown("---")

# ------------------------------
# Cluster Distribution
# ------------------------------

cluster_counts = (
    df["cluster_name"]
    .value_counts()
    .sort_index()
    .reset_index()
)

cluster_counts.columns = [
    "cluster_name",
    "count"
]

fig = px.bar(
    cluster_counts,
    x="cluster_name",
    y="count",
    title="Trader Distribution by Cluster"
)

fig.update_layout(
    template="plotly_dark",
    height=500,
    title_x=0.05,
    font=dict(size=14)
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------------
# Persona Distribution
# ------------------------------

persona_counts = (
    df["cluster_name"]
    .value_counts()
    .reset_index()
)

persona_counts.columns = [
    "cluster_name",
    "count"
]

fig = px.pie(
    persona_counts,
    names="cluster_name",
    values="count",
    title="Trader Persona Distribution"
)

fig.update_layout(
    template="plotly_dark",
    height=500,
    title_x=0.05,
    font=dict(size=14)
)

st.plotly_chart(
    fig,
    use_container_width=True
)
