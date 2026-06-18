"""
Render data quality checks for dashboard source data.

This page inspects clustered trader features for missing values, duplicates,
and invalid metric ranges before analytics are interpreted.
"""

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Data Quality",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Data Quality Dashboard")

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

# --------------------------
# Quality Checks
# --------------------------

missing_values = df.isnull().sum().sum()

duplicate_traders = (
    df["trader_id"]
    .duplicated()
    .sum()
)

invalid_win_rate = (
    (df["win_rate"] < 0) |
    (df["win_rate"] > 1)
).sum()

invalid_roi = (
    df["roi_pct"].isnull()
).sum()

invalid_leverage = (
    df["avg_leverage"] < 0
).sum()

# --------------------------
# Metrics
# --------------------------

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Missing Values",
    missing_values
)

col2.metric(
    "Duplicate Traders",
    duplicate_traders
)

col3.metric(
    "Invalid Win Rate",
    invalid_win_rate
)

col4.metric(
    "Invalid ROI",
    invalid_roi
)

col5.metric(
    "Invalid Leverage",
    invalid_leverage
)

# --------------------------
# Detailed Report
# --------------------------

st.markdown("---")

report = pd.DataFrame({
    "Check": [
        "Missing Values",
        "Duplicate Traders",
        "Invalid Win Rate",
        "Invalid ROI",
        "Invalid Leverage"
    ],
    "Count": [
        missing_values,
        duplicate_traders,
        invalid_win_rate,
        invalid_roi,
        invalid_leverage
    ]
})

st.dataframe(
    report,
    use_container_width=True
)

# --------------------------
# Quality Score
# --------------------------

total_issues = report["Count"].sum()

quality_score = max(
    0,
    100 - total_issues
)

st.success(
    f"Data Quality Score: {quality_score}%"
)
