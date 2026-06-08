from pathlib import Path

import matplotlib.pyplot as plt
import shap


OUTPUT_DIR = Path("outputs/explainability")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_DISPLAY_NAMES = {
    "early_avg_holding_minutes":
        "Holding Duration",

    "early_avg_risk_pct":
        "Risk Percentage",

    "early_avg_leverage":
        "Leverage Usage",

    "early_overnight_position_rate":
        "Overnight Position Rate",

    "early_stop_loss_usage_rate":
        "Stop Loss Usage",

    "early_avg_pnl":
        "Average PnL",

    "early_win_rate":
        "Win Rate",

    "early_total_trades":
        "Trade Count"
}



def generate_summary_plot(shap_values, X):
    """
    Global SHAP summary plot.
    """
    X_display = X.rename(
    columns=FEATURE_DISPLAY_NAMES
    )
    shap.summary_plot(
        shap_values.values,
        X_display,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_summary.png",
        bbox_inches="tight"
    )

    plt.close()


def generate_bar_plot(shap_values, X):
    """
    SHAP feature importance bar chart.
    """
    X_display = X.rename(
    columns=FEATURE_DISPLAY_NAMES
    )

    shap.summary_plot(
        shap_values.values,
        X_display,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_bar_importance.png",
        bbox_inches="tight"
    )

    plt.close()