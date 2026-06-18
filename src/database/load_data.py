"""
Load clustered trader metrics into the analytics database.

This script reads the feature-store output and replaces the database table used
by downstream analytics views and dashboards.
"""

import pandas as pd

from src.database.db import engine

df = pd.read_csv(
    "data/feature_store/clustered_traders.csv"
)

print(f"Rows loaded from CSV: {len(df)}")

# Replace keeps the local analytics table aligned with the latest feature store.
df.to_sql(
    "trader_metrics",
    engine,
    if_exists="replace",
    index=False
)

print(
    f"Successfully loaded {len(df)} rows into trader_metrics."
)
