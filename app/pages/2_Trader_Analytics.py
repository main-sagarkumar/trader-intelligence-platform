"""
Render trader-level analytics in the Streamlit dashboard.

This page visualizes ROI, leverage, and persona-level behavior patterns from
the clustered trader feature dataset.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

# ------------------------------
# Page Config
# ------------------------------

st.set_page_config(
    page_title="Trader Analytics",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Trader Analytics")

st.sidebar.markdown("""
# 📈 Trader Intelligence

### Analytics Platform

Behavior Analytics  
Risk Analytics  
Trader Segmentation  
Business Intelligence
""")

# ------------------------------
# Load Data
# ------------------------------

df = pd.read_csv(
    "data/feature_store/clustered_traders.csv"
)

# ------------------------------
# ROI Distribution
# ------------------------------

st.subheader("ROI Distribution")

fig = px.histogram(
    df,
    x="roi_pct",
    nbins=40,
    title="Distribution of ROI"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------------
# Win Rate Distribution
# ------------------------------

st.subheader("Win Rate Distribution")

fig = px.histogram(
    df,
    x="win_rate",
    nbins=30,
    title="Distribution of Win Rate"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------------
# Leverage vs ROI
# ------------------------------

st.subheader("Leverage vs ROI")

fig = px.scatter(
    df,
    x="avg_leverage",
    y="roi_pct",
    color="cluster",
    hover_data=[
        "trader_id",
        "persona"
    ],
    title="Leverage vs ROI"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------------
# Top Traders
# ------------------------------

st.subheader("Top 20 Traders by ROI")

top_traders = (
    df.sort_values(
        "roi_pct",
        ascending=False
    )
    .head(20)
)

st.dataframe(
    top_traders[
        [
            "trader_id",
            "persona",
            "roi_pct",
            "win_rate",
            "avg_leverage",
            "cluster"
        ]
    ],
    use_container_width=True
)

# ------------------------------
# Persona Performance
# ------------------------------

st.subheader("Average ROI by Persona")

persona_roi = (
    df.groupby("persona")["roi_pct"]
      .mean()
      .reset_index()
      .sort_values(
          "roi_pct",
          ascending=False
      )
)

fig = px.bar(
    persona_roi,
    x="persona",
    y="roi_pct",
    title="Average ROI by Persona"
)

st.plotly_chart(
    fig,
    use_container_width=True
)
