"""
Validate raw synthetic trade data before feature engineering.

This module prints a data quality report covering schema health, balance logic,
time validity, and risk distributions used by downstream ML pipelines.
"""

import pandas as pd

from configs.paths_config import RAW_TRADES_FILE


def validate_dataset(df):
    """
    Print validation diagnostics for a raw trade dataset.

    Args:
        df: Raw trade-level DataFrame generated or ingested by the platform.
    """

    print("\n" + "=" * 50)
    print("DATASET VALIDATION REPORT")
    print("=" * 50)

    # Verify dataset dimensions
    print("\nShape:", df.shape)

    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Ensure trade IDs are unique
    duplicate_trade_ids = df["trade_id"].duplicated().sum()

    print("\nDuplicate Trade IDs:", duplicate_trade_ids)

    # Verify balance evolution logic
    balance_errors = (
        (df["balance_before_trade"] + df["pnl"]).round(2)
        !=
        df["balance_after_trade"]
        ).sum()

    print("\nBalance Errors:", balance_errors)

    # Ensure holding period is positive
    invalid_holding_minutes = (df["holding_minutes"] <= 0).sum()

    print("\nInvalid Holding Minutes:", invalid_holding_minutes)

    # Ensure leverage is positive
    invalid_leverage = (df["leverage_used"] <= 0).sum()

    print("\nInvalid Leverage:", invalid_leverage)

    # Verify no weekend trades exist
    df["trade_timestamp"] = pd.to_datetime(df["trade_timestamp"])

    weekend_trades = (df["trade_timestamp"].dt.dayofweek > 4).sum()

    print("\nWeekend Trades:", weekend_trades)

    # Review risk distribution
    print("\nRisk Percentage Summary:")
    print(df["risk_percentage"].describe())

    print("\nValidation Complete")


if __name__ == "__main__":

    df = pd.read_csv(RAW_TRADES_FILE)

    validate_dataset(df)
