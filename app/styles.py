"""
Provide reusable styling helpers for the Streamlit dashboard.

This module centralizes custom CSS and metric-card rendering so dashboard pages
can share a consistent visual treatment.
"""

import streamlit as st

def load_css():
    """
    Inject dashboard-wide CSS styles into the Streamlit page.

    Returns:
        None. Styles are rendered directly through Streamlit markdown.
    """

    st.markdown("""
    <style>

    .main {
        padding-top: 1rem;
    }

    .metric-card {
        background: #1E293B;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #334155;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    }

    .metric-title {
        color: #94A3B8;
        font-size: 14px;
    }

    .metric-value {
        color: white;
        font-size: 32px;
        font-weight: bold;
    }

    .insight-card {
        background: #111827;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 10px;
    }

    </style>
    """, unsafe_allow_html=True)

def metric_card(title, value):
    """
    Render a styled metric card in the Streamlit dashboard.

    Args:
        title: Metric label shown above the value.
        value: Display value for the metric.
    """

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
