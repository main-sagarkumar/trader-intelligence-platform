"""
Main Streamlit dashboard for trader intelligence.

This page collects trader behavior inputs, calls the FastAPI prediction
service, and displays segment insights and recommendations.
"""

import requests
import streamlit as st

CLUSTER_NAMES = {
    0: "🔥 High-Risk Gamblers",
    1: "🛡 Conservative Traders",
    2: "⚡ Aggressive Swing Traders",
    3: "🏆 Elite Performers",
    4: "💀 Capital Destroyers"
}


# ==================================================
# Configuration
# ==================================================

API_URL = (
    "https://trader-intelligence-api-532641891308."
    "asia-south1.run.app/api/v1"
)

st.set_page_config(
    page_title="Home",
    page_icon="📈",
    layout="wide"
)


# ==================================================
# Hero Section
# ==================================================

st.title("📈 Trader Intelligence Platform")

st.markdown("""
### AI-Powered Trader Segmentation & Behavioral Analysis

Analyze trader behavior using machine learning models and discover
which trader segment a user belongs to based on trading activity,
profitability, leverage usage, and risk-taking behavior.
""")

st.divider()

# ==================================================
# Input Section
# ==================================================

st.subheader("Trader Analysis")

with st.container():

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Trading Activity")

        total_trades = st.number_input(
            "Total Trades",
            min_value=1,
            value=100
        )

        avg_holding_minutes = st.number_input(
            "Average Holding Minutes",
            min_value=1,
            value=1000
        )

    with col2:
        st.markdown("#### Performance Metrics")

        avg_pnl = st.number_input(
            "Average PnL",
            value=500.0
        )

        roi_input = st.number_input(
            "ROI (%)",
            value=5.0,
            step=0.1
        )

    with col3:
        st.markdown("#### Risk Metrics")

        avg_leverage = st.number_input(
            "Average Leverage",
            min_value=1.0,
            value=2.0,
            step=0.1
        )

        win_rate = st.slider(
            "Win Rate (%)",
            min_value=0,
            max_value=100,
            value=60
        )

st.markdown("")

analyze_clicked = st.button(
    "🔍 Analyze Trader",
    use_container_width=True,
    type="primary"
)

# ==================================================
# Prediction
# ==================================================

if analyze_clicked:

    trader_features = {
        "total_trades": total_trades,
        "avg_pnl": avg_pnl,
        "roi_pct": roi_input / 100,
        "avg_holding_minutes": avg_holding_minutes,
        "avg_leverage": avg_leverage,
        "win_rate": win_rate / 100
    }

    with st.spinner("Analyzing trader behavior..."):

        response = requests.post(
            f"{API_URL}/predict-current-segment",
            json=trader_features
        )

        result = response.json()

    st.divider()

    # ==================================================
    # Results Dashboard
    # ==================================================

    st.subheader("📊 Trader Intelligence Report")

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric(
            label="Predicted Cluster",
            value=result["cluster"]
        )

    with metric_col2:
        st.metric(
            label="Trader Segment",
            value=result["segment"]
        )

    st.markdown("")

    # ==================================================
    # Segment Description
    # ==================================================

    st.subheader("🧠 Segment Insight")

    st.info(
        result["description"]
    )

    # ==================================================
    # Recommendations
    # ==================================================

    st.subheader("✅ Recommendations")

    for recommendation in result["recommendations"]:
        st.success(recommendation)

# ==================================================
# Footer
# ==================================================

st.divider()

with st.expander("About This Platform"):
    st.write(
        """
        The Trader Intelligence Platform uses machine learning models
        trained on trader behavior data to identify trading styles,
        risk profiles, and trader segments.

        The system combines clustering and predictive analytics to
        generate actionable trader insights and recommendations.
        """
    )
