import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from configs.paths_config import FEATURE_STORE_FILE, EDA_OUTPUT_DIR


def load_features():

    # Load trader-level feature dataset
    return pd.read_csv(FEATURE_STORE_FILE)


def plot_feature_histograms(df):

    # Core behavioral features for distribution analysis
    features = [
        "total_trades",
        "total_pnl",
        "avg_pnl",
        "win_rate",
        "avg_holding_minutes",
        "avg_leverage",
        "avg_risk_pct"
    ]

    # Visualize feature distributions and outliers
    df[features].hist(figsize=(14, 10), bins=20)

    plt.suptitle("Trader Feature Distributions")
    plt.tight_layout()

    plt.savefig(EDA_OUTPUT_DIR / "feature_histograms.png")
    plt.show()


def plot_correlation_heatmap(df):

    # Analyze relationships between numerical features
    numeric_df = df.select_dtypes(include="number")

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm"
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    plt.savefig(EDA_OUTPUT_DIR / "correlation_heatmap.png")
    plt.show()


def plot_persona_boxplots(df):

    # Compare behavioral features across personas
    features = [
        "avg_holding_minutes",
        "avg_leverage",
        "avg_risk_pct",
        "win_rate"
    ]

    for feature in features:

        plt.figure(figsize=(10, 5))

        sns.boxplot(
            data=df,
            x="persona",
            y=feature
        )

        plt.xticks(rotation=15)
        plt.title(f"{feature} by Persona")
        plt.tight_layout()

        plt.savefig(EDA_OUTPUT_DIR / f"{feature}_boxplot.png")

        plt.show()


def feature_summary(df):

    # Quick statistical overview of engineered features
    print("\n" + "=" * 80)
    print("FEATURE SUMMARY")
    print("=" * 80)

    print(df.describe())


if __name__ == "__main__":

    trader_features = load_features()

    print("\nDataset Shape:", trader_features.shape)

    feature_summary(trader_features)

    plot_feature_histograms(trader_features)

    plot_correlation_heatmap(trader_features)

    plot_persona_boxplots(trader_features)

    print("\nFeature EDA Complete")
    print(f"Plots saved to: {EDA_OUTPUT_DIR}")