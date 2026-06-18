# Trader Intelligence Platform Project Structure

This document explains what each folder in the repository contains and how the main pieces fit together.

## High-Level Layout

```text
trader-intelligence-platform/
|-- app/
|-- configs/
|-- data/
|-- dashboards/
|-- docs/
|-- notebooks/
|-- outputs/
|-- saved_models/
|-- src/
|-- tests/
|-- .env
|-- Dockerfile
|-- Dockerfile.streamlit
|-- pytest.ini
|-- requirements.txt
|-- requirements-prod.txt
|-- requirements-streamlit.txt
|-- requirements-lock.txt
`-- README.md
```

## Root Files

### `README.md`

Main project overview. It explains the business problem, ML pipeline, API, dashboard, Docker usage, and basic setup.

### `.env`

Local environment variables. This file is used for configuration such as database connection values or runtime settings.

Do not commit real secrets, passwords, API keys, or production credentials.

### `pytest.ini`

PyTest configuration.

Current purpose:

- Makes the project root importable during tests with `pythonpath = .`

### `requirements.txt`

Main development dependency list.

Used for local development and testing. It includes core libraries such as:

- pandas
- numpy
- scikit-learn
- FastAPI
- Streamlit
- MLflow
- pytest
- pytest-cov

### `requirements-prod.txt`

Production-focused dependency file.

This is normally used for API deployment containers where only runtime dependencies should be installed.

### `requirements-streamlit.txt`

Streamlit-specific dependency file.

This is useful when deploying the dashboard separately from the API.

### `requirements-lock.txt`

Pinned dependency snapshot.

This records exact installed package versions for reproducibility.

### `Dockerfile`

Docker image definition for the FastAPI service.

Expected purpose:

- Install API/runtime dependencies
- Copy project code
- Start the API server

### `Dockerfile.streamlit`

Docker image definition for the Streamlit dashboard.

Expected purpose:

- Install dashboard dependencies
- Copy app/dashboard files
- Start Streamlit

## `app/`

Contains the Streamlit dashboard application.

```text
app/
|-- streamlit_app.py
|-- styles.py
`-- pages/
    |-- 1_Executive_Dashboard.py
    |-- 2_Trader_Analytics.py
    |-- 3_Cluster_Analytics.py
    |-- 4_Data_Quality.py
    `-- 5_Automated_Insights.py
```

### `app/streamlit_app.py`

Main Streamlit application entry point.

It provides the interactive user interface for trader intelligence workflows, such as entering trader metrics and viewing predictions or recommendations.

### `app/styles.py`

Dashboard styling helpers.

Used for reusable visual components such as metric cards and CSS injection.

### `app/pages/`

Multi-page Streamlit dashboard pages.

#### `1_Executive_Dashboard.py`

High-level business dashboard.

Likely includes:

- Total traders
- Average ROI
- Average win rate
- Total PnL
- Executive summary metrics

#### `2_Trader_Analytics.py`

Trader-level analytics page.

Likely includes:

- ROI distribution
- Trader ranking
- Persona-level comparison
- Leverage versus ROI analysis

#### `3_Cluster_Analytics.py`

Cluster-level analytics page.

Likely includes:

- Average ROI by cluster
- Cluster size distribution
- Risk versus ROI
- Best and weakest cluster insights

#### `4_Data_Quality.py`

Data quality dashboard page.

Likely includes:

- Missing values
- Invalid ROI checks
- Duplicate checks
- Null checks
- Schema health indicators

#### `5_Automated_Insights.py`

Automated narrative insight page.

Uses analytics logic to turn trader/cluster data into business-friendly observations.

## `configs/`

Contains project configuration constants.

```text
configs/
|-- model_config.py
`-- paths_config.py
```

### `configs/model_config.py`

Central model configuration.

Contains:

- Clustering feature list
- Number of KMeans clusters
- Early-trader feature list
- Target column name
- Random seed
- Random Forest settings
- Train/test split settings

Important constants:

- `CLUSTERING_FEATURES`
- `OPTIMAL_CLUSTERS`
- `EARLY_FEATURES`
- `TARGET_COLUMN`
- `RANDOM_STATE`

### `configs/paths_config.py`

Central path configuration.

Defines project directories such as:

- `DATA_DIR`
- `RAW_DATA_DIR`
- `FEATURE_STORE_DIR`
- `MODEL_DIR`
- `OUTPUT_DIR`
- `DASHBOARD_DIR`
- `EDA_OUTPUT_DIR`
- `CLUSTERING_OUTPUT_DIR`
- `REPORTS_DIR`

It also creates required directories if they do not exist.

## `data/`

Stores local project data.

This folder may not always appear in `git status` if ignored or generated locally.

Expected structure:

```text
data/
|-- raw/
|-- processed/
`-- feature_store/
```

### `data/raw/`

Raw generated or ingested trade-level data.

Example:

- `raw_data_trades.csv`

### `data/processed/`

Cleaned or transformed intermediate datasets.

### `data/feature_store/`

Feature datasets used by clustering, supervised training, dashboards, and inference.

Expected examples:

- `trader_features.csv`
- `scaled_trader_features.csv`
- `clustered_traders.csv`
- `early_trader_features.csv`

## `dashboards/`

Dashboard output or exported dashboard assets.

This appears to be a generated or support directory for dashboard-related artifacts.

## `docs/`

Project documentation.

```text
docs/
|-- architecture.png
|-- analytics_warehouse.md
|-- data_dictionary.md
|-- interview_preparation_guide.md
|-- project_structure.md
|-- startup_guide.md
`-- notes/
    `-- Streamlit Application layer onwards.md
```

### `docs/architecture.png`

Architecture diagram used in the README.

### `docs/analytics_warehouse.md`

Documentation for analytics/data warehouse concepts or reporting layer.

### `docs/data_dictionary.md`

Explains dataset columns, metrics, and business meaning.

### `docs/interview_preparation_guide.md`

Interview preparation material for explaining the project.

### `docs/project_structure.md`

This document.

Explains what each folder and major file does.

### `docs/startup_guide.md`

Quick local startup commands.

Includes:

- Activating the virtual environment
- Running tests
- Starting FastAPI
- Starting Streamlit

### `docs/notes/`

Working notes and deeper design notes.

## `notebooks/`

Exploratory notebooks.

```text
notebooks/
`-- 01_data_generation/
    `-- trader_persona_design.ipynb
```

### `notebooks/01_data_generation/trader_persona_design.ipynb`

Notebook for exploring synthetic trader personas and data-generation design.

Use notebooks for exploration, not production pipeline logic.

## `outputs/`

Generated outputs from EDA, clustering, reports, or experiments.

Expected structure:

```text
outputs/
|-- eda/
|-- clustering/
`-- reports/
```

### `outputs/eda/`

EDA plots or summary files.

### `outputs/clustering/`

Clustering analysis outputs, charts, or reports.

### `outputs/reports/`

Generated reports.

## `saved_models/`

Serialized trained model artifacts.

```text
saved_models/
|-- kmeans_model.pkl
|-- scaler.pkl
`-- segment_classifier.pkl
```

### `saved_models/kmeans_model.pkl`

Trained KMeans clustering model.

Used for assigning a current trader to a behavioral cluster.

### `saved_models/scaler.pkl`

Fitted feature scaler.

Used to transform trader features before KMeans prediction.

### `saved_models/segment_classifier.pkl`

Trained supervised classifier.

Used to predict a trader's future segment from early trading behavior.

## `src/`

Main production source code.

```text
src/
|-- analytics/
|-- api/
|-- clustering/
|-- dashboard/
|-- database/
|-- eda/
|-- feature_engineering/
|-- ingestion/
|-- modeling/
|-- monitoring/
|-- pipelines/
|-- prediction/
|-- utils/
`-- validation/
```

## `src/ingestion/`

Data generation and raw-data definitions.

```text
src/ingestion/
|-- personas.py
|-- raw_data_generator.py
`-- schema.py
```

### `personas.py`

Defines synthetic trader personas.

Each persona includes behavior assumptions such as:

- Average trades per day
- Win probability
- Leverage range
- Risk-per-trade range
- Holding time range
- Stop-loss discipline
- Overnight holding probability
- Preferred strategies

### `raw_data_generator.py`

Generates synthetic trade-level data.

Main class:

- `SyntheticTradeGenerator`

Responsibilities:

- Create trader IDs
- Assign personas
- Generate account sizes
- Generate trading days
- Generate trade counts
- Generate trade timestamps
- Generate PnL
- Generate complete trade events

### `schema.py`

Contains schema dictionaries for trade transactions and trade legs.

This acts as a reference for expected raw data structure.

## `src/validation/`

Dataset validation logic.

```text
src/validation/
`-- validate_dataset.py
```

### `validate_dataset.py`

Prints a validation report for raw trade datasets.

Checks include:

- Dataset shape
- Missing values
- Duplicate trade IDs
- Balance calculation errors
- Invalid holding minutes
- Invalid leverage values
- Weekend trades
- Risk percentage summary

## `src/feature_engineering/`

Feature creation and transformation logic.

```text
src/feature_engineering/
|-- build_trader_features.py
|-- build_early_trader_features.py
|-- select_features.py
`-- scale_features.py
```

### `build_trader_features.py`

Aggregates trade-level data into trader-level features.

Creates features such as:

- Account size
- Final balance
- Total trades
- Total PnL
- Average PnL
- Win rate
- Average holding minutes
- Average leverage
- Average risk percentage
- Stop-loss usage rate
- Overnight position rate
- ROI percentage

### `build_early_trader_features.py`

Builds supervised-learning features from the first N trades per trader.

Main flow:

```text
raw trades
-> first 20 trades per trader
-> early behavior features
-> attach cluster label as target
-> save early_trader_features.csv
```

Important functions:

- `load_data`
- `get_early_trades`
- `build_features`
- `attach_target`
- `save_dataset`

### `select_features.py`

Selects final clustering features from trader-level features.

Also includes feature correlation analysis and heatmap plotting helpers.

### `scale_features.py`

Scales clustering features using `StandardScaler`.

Outputs:

- Scaled feature DataFrame
- Fitted scaler object

The scaler is saved for inference.

## `src/clustering/`

Unsupervised trader segmentation logic.

```text
src/clustering/
|-- cluster_analysis.py
|-- cluster_definitions.py
|-- find_optimal_clusters.py
`-- kmeans_clustering.py
```

### `find_optimal_clusters.py`

Analyzes possible cluster counts.

Likely uses:

- Elbow method
- Silhouette score

Purpose:

- Help decide the best value for `OPTIMAL_CLUSTERS`

### `kmeans_clustering.py`

Trains KMeans clustering model.

Main responsibilities:

- Load trader features
- Load scaled features
- Train KMeans
- Save model
- Attach cluster labels
- Save clustered trader dataset
- Print cluster summaries

Important functions:

- `load_data`
- `train_kmeans`
- `cluster_summary`
- `cluster_feature_profile`
- `cluster_persona_comparison`

### `cluster_analysis.py`

Cluster analysis and profiling logic.

Used to understand and explain discovered trader segments.

### `cluster_definitions.py`

Business definitions for clusters.

Contains:

- `CLUSTER_MAPPING`
- `CLUSTER_DESCRIPTIONS`
- `CLUSTER_RECOMMENDATIONS`

This file connects numeric cluster IDs to business-friendly segment names and advice.

## `src/modeling/`

Supervised model training.

```text
src/modeling/
`-- train_segment_classifier.py
```

### `train_segment_classifier.py`

Trains a Random Forest classifier to predict future trader segment from early-trader features.

Main responsibilities:

- Load early-trader feature dataset
- Split train/test data
- Train classifier
- Evaluate accuracy
- Print classification report
- Print confusion matrix
- Show feature importance
- Save trained classifier

Important functions:

- `load_data`
- `prepare_data`
- `train_model`
- `evaluate_model`
- `show_feature_importance`
- `save_model`

## `src/prediction/`

Prediction and recommendation logic.

```text
src/prediction/
|-- future_segment_predictor.py
|-- recommendation_engine.py
`-- trader_segment_predictor.py
```

### `trader_segment_predictor.py`

Predicts a current trader's behavioral cluster using full trader-level features.

Main flow:

```text
input trader features
-> load scaler and KMeans model
-> scale features
-> predict cluster
```

Important functions:

- `load_models`
- `predict_cluster`

### `future_segment_predictor.py`

Predicts a trader's future segment from early trading features.

Main flow:

```text
early trader features
-> load classifier
-> predict cluster
-> predict probabilities
-> return cluster and confidence
```

Important functions:

- `load_model`
- `predict_future_profile`

### `recommendation_engine.py`

Converts a cluster ID into business-facing trader recommendations.

Returns:

- Segment name
- Segment description
- Recommendation list

## `src/pipelines/`

Higher-level orchestration flows.

```text
src/pipelines/
|-- data_generation_pipeline.py
|-- future_prediction_pipeline.py
`-- inference_pipeline.py
```

### `data_generation_pipeline.py`

Reserved for or intended to contain the end-to-end data generation workflow.

### `inference_pipeline.py`

Current-trader inference pipeline.

Main flow:

```text
trader features
-> predict cluster
-> fetch recommendations
-> return trader profile
```

### `future_prediction_pipeline.py`

Future-trader prediction pipeline.

Main flow:

```text
early trader features
-> predict future cluster and confidence
-> fetch recommendations
-> return future trader profile
```

## `src/api/`

FastAPI application layer.

```text
src/api/
|-- main.py
|-- routes.py
`-- schemas.py
```

### `main.py`

FastAPI app entry point.

Creates the app and includes the versioned router under:

```text
/api/v1
```

### `routes.py`

Defines API endpoints.

Current endpoints:

- `GET /api/v1/health`
- `GET /api/v1/`
- `POST /api/v1/predict-current-segment`
- `POST /api/v1/predict-future-segment`

### `schemas.py`

Pydantic request and response models.

Contains:

- `CurrentTraderRequest`
- `FutureTraderRequest`
- `CurrentTraderResponse`
- `FutureTraderResponse`

## `src/analytics/`

Analytics helpers used by dashboards and reports.

```text
src/analytics/
|-- cluster_analytics.py
|-- insights.py
`-- kpi_service.py
```

### `kpi_service.py`

Computes high-level KPIs.

Examples:

- Total traders
- Average ROI
- Average win rate
- Total PnL
- Average leverage

### `cluster_analytics.py`

Computes cluster-level aggregates.

Examples:

- Trader count by cluster
- Average ROI by cluster
- Average win rate by cluster
- Average leverage by cluster
- Average risk by cluster
- Average PnL by cluster

### `insights.py`

Generates automated insight strings from trader/cluster data.

Examples:

- Best ROI cluster
- Weakest profitability cluster
- Best persona by returns
- Leverage-to-ROI correlation

## `src/dashboard/`

Dashboard-facing service layer.

```text
src/dashboard/
`-- dashboard_service.py
```

### `dashboard_service.py`

Provides dashboard-friendly wrapper functions.

Currently includes:

- `analyze_trader`

This delegates prediction to the inference pipeline.

## `src/database/`

Database connection and loading helpers.

```text
src/database/
|-- db.py
|-- load_data.py
|-- read_views.py
`-- test_connection.py
```

### `db.py`

Database engine/session setup.

Usually reads connection configuration from environment variables.

### `load_data.py`

Loads DataFrame data into a database table.

Expected use:

- Push trader metrics or feature outputs into a relational database.

### `read_views.py`

Reads database views or tables for analytics/dashboard use.

### `test_connection.py`

Small connection-check helper.

Despite the name, this is production/helper code under `src`, not part of the PyTest suite.

## `src/eda/`

Exploratory data analysis scripts.

```text
src/eda/
|-- feature_eda.py
`-- trader_eda.py
```

### `trader_eda.py`

EDA over trader-level features.

Likely includes:

- ROI analysis
- Persona analysis
- Feature summaries
- Visualizations

### `feature_eda.py`

EDA over engineered features.

Likely includes:

- Feature distributions
- Correlation checks
- Plotting helpers

## `src/monitoring/`

Reserved for monitoring functionality.

Expected future responsibilities:

- Model drift checks
- Data quality monitoring
- Prediction monitoring
- Alerting

## `src/utils/`

Utility modules.

```text
src/utils/
|-- env_loader.py
`-- logger.py
```

### `env_loader.py`

Loads environment variables, likely through `python-dotenv`.

### `logger.py`

Central logging configuration/helper.

## `tests/`

PyTest test suite.

```text
tests/
|-- conftest.py
|-- test_api.py
|-- test_clustering.py
|-- test_data_generation.py
|-- test_data_validation.py
|-- test_doc.md
|-- test_feature_engineering.py
|-- test_mlflow.py
|-- test_model_loading.py
|-- test_prediction.py
|-- test_segmentation.py
|-- test_streamlit_helpers.py
`-- integration/
    |-- test_end_to_end.py
    |-- test_feature_pipeline.py
    `-- test_prediction_pipeline.py
```

### `tests/conftest.py`

Shared test fixtures.

Includes:

- Sample raw trade data
- Sample trader feature data
- Sample API payloads
- Dummy model fixture
- FastAPI test client
- Workspace-local temporary directory fixture

### `tests/test_data_generation.py`

Tests synthetic trade generation.

### `tests/test_data_validation.py`

Tests raw dataset contract and validation report behavior.

### `tests/test_feature_engineering.py`

Tests trader-level and early-trader feature calculations.

### `tests/test_clustering.py`

Tests KMeans clustering, PCA dimensions, no-NaN transformed data, and serialization.

### `tests/test_prediction.py`

Tests current and future prediction functions and pipeline schema.

### `tests/test_model_loading.py`

Tests model-loading paths and model serialization behavior.

### `tests/test_mlflow.py`

Tests local MLflow experiment, parameter, metric, and artifact logging.

### `tests/test_api.py`

Tests FastAPI endpoints with mocked model-backed dependencies.

### `tests/test_streamlit_helpers.py`

Tests dashboard helper functions without launching Streamlit.

### `tests/test_segmentation.py`

Placeholder noting that segmentation behavior is covered through clustering and prediction tests.

### `tests/test_doc.md`

Documentation for the PyTest suite.

### `tests/integration/`

Integration tests for cross-module flows.

Covered flows:

- Raw data to features
- Early features to supervised target
- Current prediction pipeline
- Future prediction pipeline
- Raw data to clustering-ready output

## Main Runtime Flows

## Flow 1: Generate Synthetic Trade Data

```text
src/ingestion/personas.py
-> src/ingestion/raw_data_generator.py
-> data/raw/raw_data_trades.csv
```

Purpose:

- Create realistic synthetic trader activity data.

## Flow 2: Build Trader Features

```text
data/raw/raw_data_trades.csv
-> src/feature_engineering/build_trader_features.py
-> data/feature_store/trader_features.csv
```

Purpose:

- Convert trade-level data into trader-level ML features.

## Flow 3: Train Clustering Model

```text
trader_features.csv
-> select clustering features
-> scale features
-> train KMeans
-> saved_models/kmeans_model.pkl
-> data/feature_store/clustered_traders.csv
```

Purpose:

- Discover trader behavioral segments.

## Flow 4: Build Early Trader Dataset

```text
raw trade data
-> first 20 trades per trader
-> early features
-> attach cluster label
-> data/feature_store/early_trader_features.csv
```

Purpose:

- Create supervised-learning data for future segment prediction.

## Flow 5: Train Future Segment Classifier

```text
early_trader_features.csv
-> Random Forest classifier
-> saved_models/segment_classifier.pkl
```

Purpose:

- Predict long-term trader segment using early behavior.

## Flow 6: Current Trader Prediction

```text
API/dashboard input
-> src/pipelines/inference_pipeline.py
-> src/prediction/trader_segment_predictor.py
-> scaler + KMeans
-> recommendation engine
-> response profile
```

Purpose:

- Segment a trader using full current behavior features.

## Flow 7: Future Trader Prediction

```text
API/dashboard input
-> src/pipelines/future_prediction_pipeline.py
-> src/prediction/future_segment_predictor.py
-> Random Forest classifier
-> recommendation engine
-> response profile
```

Purpose:

- Predict future segment from early trading behavior.

## Flow 8: API Serving

```text
src/api/main.py
-> src/api/routes.py
-> pipelines
-> prediction/recommendation output
```

Purpose:

- Serve model predictions over HTTP.

## Flow 9: Streamlit Dashboard

```text
app/streamlit_app.py
-> app/pages/
-> src/analytics/
-> src/dashboard/
-> src/pipelines/
```

Purpose:

- Provide interactive analytics and prediction UI.

## Recommended Reading Order

For understanding the project quickly:

1. `README.md`
2. `docs/startup_guide.md`
3. `docs/data_dictionary.md`
4. `configs/model_config.py`
5. `src/ingestion/raw_data_generator.py`
6. `src/feature_engineering/build_trader_features.py`
7. `src/clustering/kmeans_clustering.py`
8. `src/modeling/train_segment_classifier.py`
9. `src/pipelines/inference_pipeline.py`
10. `src/pipelines/future_prediction_pipeline.py`
11. `src/api/routes.py`
12. `app/streamlit_app.py`
13. `tests/test_doc.md`

## What Is Production Code Versus Support Code?

Production-style runtime code:

- `src/api/`
- `src/pipelines/`
- `src/prediction/`
- `src/feature_engineering/`
- `src/clustering/`
- `src/analytics/`
- `src/dashboard/`
- `src/database/`
- `app/`

Experimentation and analysis:

- `notebooks/`
- `src/eda/`
- Some scripts under `src/clustering/`

Configuration:

- `configs/`
- `.env`
- requirements files

Documentation:

- `README.md`
- `docs/`
- `tests/test_doc.md`

Tests:

- `tests/`

Generated artifacts:

- `data/`
- `outputs/`
- `saved_models/`
- `dashboards/`

