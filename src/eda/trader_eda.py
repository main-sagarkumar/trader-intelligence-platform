import pandas as pd

from configs.paths_config import (RAW_TRADES_FILE, FEATURE_STORE_FILE)


def persona_summary(df):

    summary = (
        df.groupby("persona")
        .agg(
            total_trades=("trade_id", "count"),
            avg_pnl=("pnl", "mean"),
            median_pnl=("pnl", "median"),
            avg_holding_minutes=("holding_minutes", "mean"),
            avg_leverage=("leverage_used", "mean"),
            avg_risk_pct=("risk_percentage", "mean")
        ).round(2).sort_values(by="avg_pnl",ascending=False)
    )

    print("\n" + "=" * 80)
    print("PERSONA SUMMARY")
    print("=" * 80)

    print(summary)

    return summary


def trader_summary(df):

    summary = (
        df.groupby(["trader_id", "persona"])
        .agg(
            total_trades=("trade_id", "count"),
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            avg_holding_minutes=("holding_minutes", "mean"),
            avg_leverage=("leverage_used", "mean"),
            avg_risk_pct=("risk_percentage", "mean")
        ).round(2).reset_index()
    )

    print("\n" + "=" * 80)
    print("TRADER SUMMARY")
    print("=" * 80)

    print(summary.head())

    return summary

def trader_persona_analysis(trader_df):

    persona_analysis = (
        trader_df.groupby("persona")
        .agg(
            trader_count=("trader_id", "count"),
            avg_total_pnl=("total_pnl", "mean"),
            median_total_pnl=("total_pnl", "median"),
            avg_total_trades=("total_trades", "mean")
        ).round(2).sort_values(by="avg_total_pnl", ascending=False)
    )

    print("\n" + "=" * 80)
    print("TRADER LEVEL PERSONA ANALYSIS")
    print("=" * 80)

    print(persona_analysis)

    return persona_analysis

def roi_analysis(trader_features):

    roi_summary = (
        trader_features
        .groupby("persona")
        .agg(
            avg_roi=("roi_pct", "mean"),
            median_roi=("roi_pct", "median")
        )
        .round(4)
        .sort_values(
            by="avg_roi",
            ascending=False
        )
    )

    print("\n" + "=" * 80)
    print("ROI ANALYSIS")
    print("=" * 80)

    print(roi_summary)

    return roi_summary


if __name__ == "__main__":

    df = pd.read_csv(RAW_TRADES_FILE)
    persona_summary(df)

    trader_df = trader_summary(df)
    trader_persona_analysis(trader_df)

    trader_features = pd.read_csv(FEATURE_STORE_FILE)
    roi_analysis(trader_features)