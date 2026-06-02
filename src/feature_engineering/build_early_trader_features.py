"""
Early Trader Feature Engineering
--------------------------------
Builds a supervised learning dataset by:

1. Extracting each trader's first N trades (early trade window)
2. Aggregating behavioural features from those trades
3. Attaching cluster labels as the prediction target

Output:
    data/feature_store/early_trader_features.csv
"""

import pandas as pd

from configs.paths_config import RAW_TRADES_FILE, FEATURE_STORE_DIR


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

# Number of earliest trades used to predict future behaviour
EARLY_TRADE_WINDOW = 20

OUTPUT_FILE = FEATURE_STORE_DIR / "early_trader_features.csv"


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw trade history and trader cluster labels.

    Returns:
        raw_trades:
            Trade-level dataset.

        clustered_traders:
            Trader-level dataset containing
            final cluster assignments.
    """
    raw_trades = pd.read_csv(RAW_TRADES_FILE)
    clustered_traders = pd.read_csv(FEATURE_STORE_DIR / "clustered_traders.csv")

    return raw_trades, clustered_traders


# ──────────────────────────────────────────────
# Early Trade Extraction
# ──────────────────────────────────────────────

def get_early_trades(raw_trades: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the first EARLY_TRADE_WINDOW trades for each trader.

    Traders with fewer than EARLY_TRADE_WINDOW trades
    are retained and all available trades are used.

    Args:
        raw_trades:
            Complete trade history.

    Returns:
        DataFrame containing only early trades.
    """
    raw_trades["trade_timestamp"] = pd.to_datetime(raw_trades["trade_timestamp"])

    # Convert outcome labels to binary values
    raw_trades["trade_outcome"] = (
        raw_trades["trade_outcome"].map({"WIN": 1, "LOSS": 0})
    )

    # Ensure trades are ordered chronologically
    raw_trades = raw_trades.sort_values(["trader_id", "trade_timestamp"])

    return (
        raw_trades
        .groupby("trader_id")
        .head(EARLY_TRADE_WINDOW)
    )


# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────

def build_features(early_trades: pd.DataFrame) -> pd.DataFrame:
    """
    Create behavioural features using only early trades.

    Features:
        - Trade activity
        - Profitability
        - Win rate
        - Holding behaviour
        - Leverage usage
        - Risk-taking behaviour
        - Stop-loss discipline
        - Overnight exposure

    Args:
        early_trades:
            First N trades per trader.

    Returns:
        Trader-level feature dataset.
    """
    trader_features = (
        early_trades
        .groupby("trader_id")
        .agg(
            early_total_trades=("trade_id", "count"),
            early_avg_pnl=("pnl", "mean"),
            early_win_rate=("trade_outcome", "mean"),
            early_avg_holding_minutes=("holding_minutes", "mean"),
            early_avg_leverage=("leverage_used", "mean"),
            early_avg_risk_pct=("risk_percentage", "mean"),
            early_stop_loss_usage_rate=("stop_loss_used", "mean"),
            early_overnight_position_rate=("overnight_position", "mean"),
        )
        .reset_index()
        .round(4)
    )
    return trader_features


# ──────────────────────────────────────────────
# Target Creation
# ──────────────────────────────────────────────

def attach_target(
    features_df: pd.DataFrame,
    clustered_traders: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach future trader segment labels.

    The cluster discovered during unsupervised learning
    becomes the target variable for supervised learning.

    Args:
        features_df:
            Early behavioural features.

        clustered_traders:
            Trader cluster assignments.

    Returns:
        Supervised learning dataset.
    """
    return features_df.merge(
        clustered_traders[["trader_id", "cluster"]],
        on="trader_id",
        how="inner",
    )


# ──────────────────────────────────────────────
# Dataset Saving
# ──────────────────────────────────────────────

def save_dataset(features_df: pd.DataFrame) -> None:
    """
    Persist engineered dataset to disk.

    Args:
        features_df:
            Final supervised learning dataset.
    """
    features_df.to_csv(OUTPUT_FILE, index=False)


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":

    raw_trades, clustered_traders = load_data()

    early_trades = get_early_trades(raw_trades)

    features_df = build_features(early_trades)

    features_df = attach_target(
        features_df,
        clustered_traders,
    )

    save_dataset(features_df)

    print("\n" + "=" * 80)
    print("EARLY TRADER FEATURE DATASET")
    print("=" * 80)

    print(f"\nShape: {features_df.shape}")

    print("\nColumns:")
    print(features_df.columns.tolist())

    print("\nSample:")
    print(features_df.head())

    print(f"\nSaved to:\n{OUTPUT_FILE}")