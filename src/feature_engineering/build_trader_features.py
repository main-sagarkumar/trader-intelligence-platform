import pandas as pd

from configs.paths_config import (
    RAW_TRADES_FILE,
    FEATURE_STORE_DIR
)


def build_trader_features(df):

    trader_features = (
    df.groupby(["trader_id", "persona"])
    .agg(
        account_size=("account_size", "first"),
        final_balance=("balance_after_trade", "last"),
        total_trades=("trade_id", "count"),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
        win_rate=("trade_outcome", lambda x: (x == "WIN").mean()),
        avg_holding_minutes=("holding_minutes", "mean"),
        avg_leverage=("leverage_used", "mean"),
        avg_risk_pct=("risk_percentage", "mean"),
        stop_loss_usage_rate=("stop_loss_used", "mean"),
        overnight_position_rate=("overnight_position", "mean")
    ).reset_index())

    # Normalize profitability across account sizes
    trader_features["roi_pct"] = (
    trader_features["total_pnl"] / trader_features["account_size"])

    trader_features = trader_features.round(4)

    return trader_features


if __name__ == "__main__":

    df = pd.read_csv(RAW_TRADES_FILE)

    trader_features = build_trader_features(df)

    print("\n" + "=" * 80)
    print("TRADER FEATURE DATASET")
    print("=" * 80)

    print("\nShape:")
    print(trader_features.shape)

    print("\nColumns:")
    print(trader_features.columns.tolist())

    print("\nSample Features:")
    print(trader_features.head())

    print("\nFeature Summary:")
    print(trader_features.describe())

    trader_features.to_csv(FEATURE_STORE_DIR / "trader_features.csv", index=False)

    print("\nTrader features saved to:")
    print(FEATURE_STORE_DIR / "trader_features.csv")