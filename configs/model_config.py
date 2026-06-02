''' Model configuration shared across feature engineering,
clustering, and prediction modules. '''

CLUSTERING_FEATURES = [
    "total_trades",
    "avg_pnl",
    "roi_pct",
    "avg_holding_minutes",
    "avg_leverage",
    "win_rate"
]

# Optimal number of clusters selected using Elbow Method and Silhouette Score
OPTIMAL_CLUSTERS = 5

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Features derived from a trader's first 20 trades.
# Used to predict the trader's future segment.
EARLY_FEATURES = [
    "early_total_trades",
    "early_avg_pnl",
    "early_win_rate",
    "early_avg_holding_minutes",
    "early_avg_leverage",
    "early_avg_risk_pct",
    "early_stop_loss_usage_rate",
    "early_overnight_position_rate",
]

# Future trader segment discovered using KMeans clustering
TARGET_COLUMN = "cluster"

# Random seed used throughout the project
RANDOM_STATE = 42

# Number of trees in Random Forest
RF_N_ESTIMATORS = 100

# Percentage of data reserved for testing
TEST_SIZE = 0.20

