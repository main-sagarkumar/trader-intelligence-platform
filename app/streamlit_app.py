from pathlib import Path
import sys
import streamlit as st

from src.dashboard.dashboard_service import analyze_trader

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Trader Intelligence Platform",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Trader Intelligence Platform")
st.markdown(
    "Analyze trader behavior using machine learning-based segmentation."
)


# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------

st.sidebar.header("Trader Inputs")

total_trades = st.sidebar.number_input(
    "Total Trades",
    min_value=1,
    value=100
)

avg_pnl = st.sidebar.number_input(
    "Average PnL",
    value=500.0
)

roi_input = st.sidebar.number_input(
    "ROI (%)",
    value=5.0,
    step=0.1
)

avg_holding_minutes = st.sidebar.number_input(
    "Average Holding Minutes",
    min_value=1,
    value=1000
)

avg_leverage = st.sidebar.number_input(
    "Average Leverage",
    min_value=1.0,
    value=2.0,
    step=0.1
)

win_rate = st.sidebar.slider(
    "Win Rate (%)",
    min_value=0,
    max_value=100,
    value=60
)


# --------------------------------------------------
# Analyze Trader
# --------------------------------------------------

if st.sidebar.button("Analyze Trader"):

    # Convert percentage inputs to model format
    trader_features = {
        "total_trades": total_trades,
        "avg_pnl": avg_pnl,
        "roi_pct": roi_input / 100,
        "avg_holding_minutes": avg_holding_minutes,
        "avg_leverage": avg_leverage,
        "win_rate": win_rate / 100
    }

    result = analyze_trader(
        trader_features
    )

    # ----------------------------------------------
    # Prediction Results
    # ----------------------------------------------

    st.subheader("Trader Intelligence Report")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Cluster",
            result["cluster"]
        )

    with col2:
        st.metric(
            "Segment",
            result["segment"]
        )

    st.markdown("---")

    st.subheader("Segment Description")

    st.info(
        result["description"]
    )

    st.subheader("Recommendations")

    for recommendation in result[
        "recommendations"
    ]:
        st.success(
            recommendation
        )