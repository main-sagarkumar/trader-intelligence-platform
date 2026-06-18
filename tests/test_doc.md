# PyTest Test Suite Documentation

## Purpose

This test suite validates the Trader Intelligence Platform from data generation through feature engineering, clustering, prediction, API responses, dashboard helper logic, MLflow logging, and integration flows.

The tests are designed to be:

- Deterministic
- Fast
- Independent of external services
- Safe to run locally or in CI
- Focused on behavior rather than implementation details

Production code is not modified by the tests. Expensive or external behavior, such as loading trained models or calling external services, is mocked where appropriate.

## How To Run Tests

From the project root:

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

To run with coverage:

```powershell
.\venv\Scripts\python.exe -m pytest --cov=src --cov=app --cov-report=term-missing
```

To collect tests without executing them:

```powershell
.\venv\Scripts\python.exe -m pytest --collect-only -q
```

## Test Layout

```text
tests/
|-- conftest.py
|-- test_api.py
|-- test_clustering.py
|-- test_data_generation.py
|-- test_data_validation.py
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

## Shared Fixtures

Shared fixtures live in `tests/conftest.py`.

### `sample_raw_trade_data`

Provides a small deterministic trade-level dataset with:

- Two traders
- Winning and losing trades
- Trade timestamps
- Leverage usage
- Stop-loss behavior
- PnL values
- Risk percentages

This fixture is used by validation, feature engineering, and integration tests.

### `sample_feature_data`

Provides trader-level feature data with enough variation for:

- Clustering tests
- KPI tests
- Cluster summary tests
- PCA dimensionality tests

It includes expected model features such as `total_trades`, `avg_pnl`, `roi_pct`, `avg_holding_minutes`, `avg_leverage`, and `win_rate`.

### `sample_early_feature_payload`

Represents a valid future-segment prediction request payload.

Used by:

- Future prediction tests
- API tests
- Integration tests

### `sample_current_feature_payload`

Represents a valid current-segment prediction request payload.

Used by:

- Current prediction tests
- API tests
- Dashboard helper tests

### `trained_dummy_model`

Creates a lightweight scikit-learn `DummyClassifier` that supports:

- `predict`
- `predict_proba`

This avoids retraining or loading expensive production models during tests.

### `api_client`

Creates a FastAPI `TestClient` and mocks route-level prediction dependencies.

This keeps API tests focused on:

- Routing
- Request validation
- Response schemas
- HTTP status codes

The API tests do not depend on serialized model artifacts.

### `temp_output_directory`

Provides a temporary directory for tests that need safe file persistence, such as model serialization tests.

## Test Areas

## Data Generation Tests

File: `tests/test_data_generation.py`

These tests validate synthetic trade generation behavior.

Covered behavior:

- Trader profile generation returns expected schema
- Generated trade events contain required fields
- Balance after trade is calculated from PnL
- Trade outcome matches PnL direction
- Generated leverage is positive
- Trade generation is reproducible for seeded random fields
- UUID trade IDs remain unique
- Generated trade counts are always positive
- Trading days exclude weekends

Important detail:

Trade IDs are generated with UUIDs, so they are intentionally not deterministic. The deterministic test compares all generated fields except `trade_id`, then separately verifies trade ID uniqueness.

## Data Validation Tests

File: `tests/test_data_validation.py`

These tests validate the expected raw trade dataset contract.

Covered behavior:

- Required columns exist
- `trader_id` is not null
- `trade_id` values are unique
- Leverage values are within an expected range
- Timestamps are parseable
- Invalid datasets are detected
- The production validation report surfaces duplicate trade IDs and balance errors

The project currently has a reporting-style validation function rather than a strict validator that raises exceptions. For that reason, some validation rules are encoded as test-side assertions to document the expected data contract.

## Feature Engineering Tests

File: `tests/test_feature_engineering.py`

These tests validate trader-level and early-trader feature calculations.

Covered production features:

- `total_trades`
- `total_pnl`
- `avg_pnl`
- `win_rate`
- `avg_leverage`
- `stop_loss_usage_rate`
- `roi_pct`
- Early-trader feature aggregation
- Early-trader target attachment
- Feature selection
- Feature scaling

Covered requested domain formulas:

- Average profit
- Average loss
- Risk-reward ratio
- PnL volatility
- Days active

Some requested features are not direct production columns today. Those formulas are tested as domain expectations using the sample data, so future production implementation has clear target behavior.

Edge cases covered:

- All winning trades
- Zero account size divide-by-zero behavior
- Empty input
- Missing losses for risk-reward calculations

## Clustering Tests

File: `tests/test_clustering.py`

These tests validate KMeans clustering and clustering helper behavior.

Covered behavior:

- KMeans pipeline executes successfully
- One cluster label is generated per trader
- Cluster count matches `OPTIMAL_CLUSTERS`
- Labels are within the expected cluster range
- PCA output has two dimensions
- Scaled features contain no NaN values
- Cluster labels can be attached to trader rows
- KMeans model serialization and deserialization works
- Cluster profile and persona comparison helpers return expected shapes

The tests use small deterministic data, not production-scale training data.

## Prediction Tests

File: `tests/test_prediction.py`

These tests validate current and future prediction behavior.

Covered behavior:

- Future predictor returns `cluster` and `confidence`
- Confidence is between `0` and `1`
- Future predictor uses the configured `EARLY_FEATURES` column order
- Current segment prediction returns an integer cluster
- Current prediction pipeline returns business profile schema
- Future prediction pipeline merges model output with recommendations
- Recommendation engine returns segment metadata
- Model-like objects expose `predict_proba`
- Probability output dimensions are valid

Model loading is mocked to avoid relying on serialized artifacts or retraining.

## Model Loading Tests

File: `tests/test_model_loading.py`

These tests validate model-loading behavior without requiring real model files.

Covered behavior:

- Future segment model is loaded from `segment_classifier.pkl`
- Current segment loader loads `scaler.pkl`
- Current segment loader loads `kmeans_model.pkl`
- Serialized dummy classifiers preserve `predict` and `predict_proba`

`joblib.load` is mocked where the test is about path correctness.

## MLflow Tests

File: `tests/test_mlflow.py`

These tests validate MLflow usage with a local temporary tracking directory.

Covered behavior:

- Local experiment creation
- Parameter logging
- Metric logging
- Artifact logging
- Logged data can be retrieved through `MlflowClient`

No external MLflow tracking server is required.

## API Tests

File: `tests/test_api.py`

These tests validate FastAPI behavior through `TestClient`.

Covered endpoints:

- `GET /api/v1/health`
- `GET /api/v1/`
- `POST /api/v1/predict-current-segment`
- `POST /api/v1/predict-future-segment`

Covered behavior:

- Health endpoint returns `200`
- Health endpoint returns `{"status": "healthy"}`
- Root endpoint returns application metadata
- Valid current prediction payload succeeds
- Valid future prediction payload succeeds
- Missing fields fail validation
- Invalid datatypes fail validation
- Boundary numeric values are accepted at schema level
- Response schemas contain expected keys

Prediction functions are mocked at the API route layer so these tests validate API behavior without loading models.

## Streamlit Helper Tests

File: `tests/test_streamlit_helpers.py`

These tests validate helper/service logic used by Streamlit-facing components.

Covered behavior:

- KPI aggregation
- Average ROI calculation
- Average win rate calculation
- Total PnL aggregation
- Cluster-level summaries
- Automated insight generation
- Dashboard analysis service delegation

The tests do not launch Streamlit and do not perform browser UI testing.

## Integration Tests

Integration tests live under `tests/integration/`.

### Feature Pipeline

File: `tests/integration/test_feature_pipeline.py`

Covered flow:

```text
Raw trade data
-> trader-level feature engineering
-> early-trader feature engineering
-> target attachment
```

Assertions:

- Expected trader row counts
- Expected feature columns
- Cluster target exists
- No null outputs

### Prediction Pipeline

File: `tests/integration/test_prediction_pipeline.py`

Covered flow:

```text
Feature payload
-> mocked model prediction
-> recommendation profile
```

Assertions:

- Current prediction returns cluster profile
- Future prediction returns cluster, confidence, and recommendations
- Confidence is bounded between `0` and `1`

### End-To-End Flow

File: `tests/integration/test_end_to_end.py`

Covered flow:

```text
Raw trade data
-> feature engineering
-> feature selection
-> scaling
-> KMeans clustering
-> prediction-ready clustered features
```

Assertions:

- Expected row count at the production feature grain
- Cluster column exists
- Cluster model uses configured cluster count
- No null outputs
- PCA projection supports two-dimensional analytics

## Mocking Strategy

The suite mocks expensive or external dependencies.

Mocked areas:

- API route prediction calls
- Current model loading
- Future model loading
- Dashboard prediction delegation

Not mocked:

- Feature calculations
- Scaling
- KMeans training on small test data
- Local MLflow file-store logging
- FastAPI request validation

This gives useful behavioral coverage while keeping the tests fast.

## Determinism Strategy

The tests use:

- Small hand-built DataFrames
- Fixed NumPy/random seeds where generation is tested
- Dummy models instead of trained production models
- Temporary directories for artifacts
- Local MLflow tracking URIs

The test suite avoids:

- Network calls
- Cloud dependencies
- Database dependencies
- Full model retraining
- Browser-based Streamlit testing

## Current Coverage Summary

The suite currently contains 60 collected tests.

Covered modules include:

- `src.ingestion.raw_data_generator`
- `src.validation.validate_dataset`
- `src.feature_engineering.build_trader_features`
- `src.feature_engineering.build_early_trader_features`
- `src.feature_engineering.select_features`
- `src.feature_engineering.scale_features`
- `src.clustering.kmeans_clustering`
- `src.prediction.future_segment_predictor`
- `src.prediction.trader_segment_predictor`
- `src.prediction.recommendation_engine`
- `src.pipelines.inference_pipeline`
- `src.pipelines.future_prediction_pipeline`
- `src.api.routes`
- `src.api.main`
- `src.analytics.kpi_service`
- `src.analytics.cluster_analytics`
- `src.analytics.insights`
- `src.dashboard.dashboard_service`

Areas intentionally not covered by this suite:

- Browser-level Streamlit UI
- Docker deployment behavior
- Real external database connectivity
- Cloud or network services
- Full production model training runs
- Notebook execution

## Warnings Seen During Test Runs

Some warnings may appear during local runs:

- MLflow file-store deprecation warning
- Pandas timestamp parsing warning for intentionally invalid timestamp tests
- Joblib CPU-core detection warning in constrained environments
- Pytest cache warnings if the environment cannot write `.pytest_cache`

These warnings do not indicate failing tests.

## Maintenance Notes

When production code changes, update tests in the closest matching file:

- Data generator changes: `test_data_generation.py`
- Raw schema changes: `test_data_validation.py`
- Feature logic changes: `test_feature_engineering.py`
- Cluster config or helpers: `test_clustering.py`
- Prediction contract changes: `test_prediction.py`
- Model artifact loading changes: `test_model_loading.py`
- API schema or route changes: `test_api.py`
- Dashboard helper changes: `test_streamlit_helpers.py`
- Cross-module behavior changes: `tests/integration/`

Prefer adding or updating fixtures in `conftest.py` when the same setup is needed in more than one test file.
