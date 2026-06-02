import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from configs.paths_config import FEATURE_STORE_FILE, EDA_OUTPUT_DIR


def select_clustering_features(df):

    # Behavioral features that describe trading style

    '''
    Final clustering features selected after EDA and correlation analysis.

    Dropped Features:
        avg_risk_pct
            - Highly correlated with avg_leverage (0.94)
            - Both represent trader aggressiveness
            - Kept avg_leverage as the primary aggressiveness metric

        stop_loss_usage_rate
            - Highly correlated with win_rate (0.93)
            - Kept win_rate because it directly measures trading performance

        overnight_position_rate
            - Highly correlated with stop_loss_usage_rate (0.97)
            - Also strongly correlated with win_rate (0.91)
            - Added limited new information to the feature set

    Excluded Non-Behavioral Features:
        trader_id
            - Identifier only
        persona
            - Ground truth label used later for cluster evaluation
        account_size
            - Capital size, not trading behavior
        final_balance
            - Derived from account size and profitability
        total_pnl
            - Scale dependent
            - Replaced by roi_pct which normalizes performance across account sizes

    Final Features:
        total_trades
        avg_pnl
        roi_pct
        avg_holding_minutes
        avg_leverage
        win_rate
'''

    from configs.model_config import CLUSTERING_FEATURES

    return df[CLUSTERING_FEATURES].copy()


def feature_correlation_analysis(df):

    # Measure linear relationships between features
    correlation_matrix = df.corr().round(2)

    print("\n" + "=" * 80)
    print("FEATURE CORRELATION MATRIX")
    print("=" * 80)

    print(correlation_matrix)

    return correlation_matrix


def plot_correlation_heatmap(correlation_matrix):

    # Visualize feature relationships before clustering
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm"
    )

    plt.title("Selected Feature Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        EDA_OUTPUT_DIR / "selected_feature_heatmap.png"
    )

    plt.show()


if __name__ == "__main__":

    # Load engineered trader features
    trader_features = pd.read_csv(FEATURE_STORE_FILE)

    # Keep only clustering features
    clustering_df = select_clustering_features(
        trader_features
    )

    print("\n" + "=" * 80)
    print("CLUSTERING FEATURE DATASET")
    print("=" * 80)

    print("\nShape:")
    print(clustering_df.shape)

    print("\nColumns:")
    print(clustering_df.columns.tolist())

    print("\nSample:")
    print(clustering_df.head())

    correlation_matrix = feature_correlation_analysis(
        clustering_df
    )

    plot_correlation_heatmap(
        correlation_matrix
    )

    print(
        f"\nHeatmap saved to: "
        f"{EDA_OUTPUT_DIR / 'selected_feature_heatmap.png'}"
    )