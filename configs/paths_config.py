"""
Define filesystem paths used across the Trader Intelligence Platform.

This configuration module centralizes data, feature-store, model, dashboard,
and output directories and ensures required local folders exist.
"""

from pathlib import Path

# Project Root
ROOT_DIR = Path.cwd()

# Data Directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_TRADES_FILE = RAW_DATA_DIR / "raw_data_trades.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

FEATURE_STORE_DIR = DATA_DIR / "feature_store"
FEATURE_STORE_FILE = FEATURE_STORE_DIR / "trader_features.csv"

# Model Directories
MODEL_DIR = ROOT_DIR / "saved_models"

# Output Directories
OUTPUT_DIR = ROOT_DIR / "outputs"

# Dashboard Directory
DASHBOARD_DIR = ROOT_DIR / "dashboards"

# EDA Output Directory
EDA_OUTPUT_DIR = OUTPUT_DIR / "eda"

# Clustering Output Directory
CLUSTERING_OUTPUT_DIR = OUTPUT_DIR / "clustering"

# Report Output Directory
REPORTS_DIR = OUTPUT_DIR / "reports"


# Create Directories If Missing
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLUSTERING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
