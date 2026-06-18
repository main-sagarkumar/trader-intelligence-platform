"""
Read analytics database views for dashboard consumption.

This module centralizes SQL queries used to retrieve precomputed warehouse or
database metrics for the Streamlit analytics layer.
"""

import pandas as pd

from src.database.db import engine


def get_executive_kpis():
    """
    Load executive KPI rows from the analytics database.

    Returns:
        DataFrame containing records from the executive_kpis view.
    """

    query = """
    SELECT *
    FROM executive_kpis
    """

    return pd.read_sql(
        query,
        engine
    )
