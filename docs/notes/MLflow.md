# Trader Intelligence Platform – MLflow Integration Notes

## Goal

Until now, our project could:

```text
Train Model
    ↓
Evaluate Model
    ↓
Save segment_classifier.pkl
```

Problem:

```text
❌ No experiment history
❌ No model comparison
❌ No versioning
❌ No artifact tracking
```

We introduced **MLflow** to solve these problems.

---

# What is MLflow?

MLflow is an MLOps platform used to:

```text
Track Experiments
Track Metrics
Track Parameters
Track Artifacts
Store Models
Version Models
Manage Model Lifecycle
```

Think:

```text
GitHub → Code Versioning

MLflow → Model Versioning
```

---

# Architecture Before MLflow

```text
Trader Intelligence Platform

Feature Engineering
        ↓

Random Forest
        ↓

segment_classifier.pkl
        ↓

Overwrite every run
```

---

# Architecture After MLflow

```text
Feature Engineering
        ↓

Random Forest
        ↓

MLflow Run
        │
        ├── Parameters
        ├── Metrics
        ├── Artifacts
        └── Model
```

---

# MLflow Components Learned

## 1. Experiment

An experiment is a collection of runs.

Example:

```python
future_segment_prediction
```

Think:

```text
Experiment
    ↓

All model training attempts
```

---

## 2. Run

Each training execution creates a run.

Example:

```text
Run 1
Run 2
Run 3
```

Each run stores:

```text
Parameters
Metrics
Artifacts
Model
```

---

## 3. Parameters

Inputs to training.

Example:

```python
RF_N_ESTIMATORS = 200

TEST_SIZE = 0.20
```

Logged using:

```python
mlflow.log_param(
    "n_estimators",
    RF_N_ESTIMATORS,
)

mlflow.log_param(
    "test_size",
    TEST_SIZE,
)
```

---

## Why Track Parameters?

Without MLflow:

```text
Which model used 100 trees?
```

Impossible to know.

With MLflow:

```text
Run A → 100 trees
Run B → 200 trees
```

Easy comparison.

---

# Metrics

Metrics measure performance.

Logged:

```python
accuracy
precision
recall
f1_score
```

---

## Code

```python
mlflow.log_metrics(
    {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
)
```

---

## Why Metrics?

Allows comparison:

```text
100 trees → Accuracy 92.5%

200 trees → Accuracy 92.5%
```

Result:

```text
Same performance

100 trees preferred
```

Less compute.

---

# Artifacts

Artifacts are files generated during training.

Examples:

```text
classification_report.txt

feature_importance.csv

confusion_matrix.png
```

---

# Metric vs Artifact

Metric:

```text
accuracy = 0.925
```

Artifact:

```text
classification_report.txt
```

Metrics:

```text
Numeric
Searchable
Comparable
```

Artifacts:

```text
Files
Reports
Visualizations
```

---

# Classification Report Artifact

## Why?

Console output disappears.

We want:

```text
Permanent record
```

---

## Flow

```text
Predictions
      ↓

classification_report()
      ↓

classification_report.txt
      ↓

MLflow Artifact
```

---

## Code

```python
report = classification_report(
    y_test,
    predictions,
    zero_division=0,
)

with open(report_file, "w") as f:
    f.write(report)

mlflow.log_artifact(
    report_file
)
```

---

# Feature Importance Artifact

## Goal

Understand:

```text
Why model predicts segments
```

---

## Flow

```text
Random Forest
        ↓

feature_importances_
        ↓

DataFrame
        ↓

feature_importance.csv
        ↓

MLflow
```

---

## Code

```python
importance_df = pd.DataFrame(
    {
        "feature": EARLY_FEATURES,
        "importance": model.feature_importances_,
    }
)
```

---

## Top Features Found

```text
early_avg_holding_minutes

early_avg_risk_pct

early_avg_leverage

early_overnight_position_rate

early_stop_loss_usage_rate
```

---

# Business Interpretation

Model learned:

```text
Trader Behavior
      ↓

Future Segment
```

instead of:

```text
Profit
      ↓

Future Segment
```

This is a very valuable insight.

---

# Confusion Matrix Artifact

## Why?

Accuracy hides mistakes.

Example:

```text
Accuracy = 92.5%
```

Looks great.

But:

```text
Cluster 0 completely failed
```

Accuracy alone doesn't show this.

---

# Flow

```text
Predictions
      ↓

Confusion Matrix
      ↓

Heatmap
      ↓

confusion_matrix.png
      ↓

MLflow
```

---

## Code

```python
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
)
```

---

# Confusion Matrix Analysis

Result:

```text
Cluster 0 → Cluster 4

2 mistakes
```

and

```text
Cluster 4 → Cluster 2

1 mistake
```

---

# Why Cluster 0 Failed?

Support:

```text
Only 2 samples
```

Likely:

```text
Class imbalance issue
```

not a Random Forest issue.

---

# Feature Importance Interpretation

Top Features:

```text
Holding Time
Risk %
Leverage
Overnight Positions
Stop Loss Usage
```

Meaning:

```text
Behavior predicts future segment
```

more than:

```text
Profitability
```

---

# Model Logging

Initially:

```text
segment_classifier.pkl
```

was saved locally.

---

## Code

```python
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="segment_classifier",
)
```

---

# Why Log Model?

Allows:

```text
Reproducibility
Deployment
Versioning
```

---

# Problem with Local Model

Current:

```text
saved_models/

segment_classifier.pkl
```

Every run:

```text
Overwrite
Overwrite
Overwrite
```

---

# Model Registry

## Purpose

Version models.

---

# Flow

```text
Train Model
      ↓

Log Model
      ↓

Register Model
      ↓

Version 1
```

Next training:

```text
Version 2
```

---

## Code

```python
model_info = mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="segment_classifier",
)

mlflow.register_model(
    model_uri=model_info.model_uri,
    name="segment_classifier",
)
```

---

# Result

MLflow Registry:

```text
segment_classifier

Version 1
Version 2
Version 3
...
```

---

# Why Registry?

Without Registry:

```text
Only latest model exists
```

With Registry:

```text
Model history

Rollback

Version tracking
```

---

# PostgreSQL Integration

Initially:

```text
MLflow
    ↓

Local mlruns
```

---

## Problem

No backend database.

Limited lifecycle management.

---

# New Architecture

```text
MLflow Server
        ↓

PostgreSQL
        ↓

Experiments
Runs
Metrics
Parameters
Registry
```

---

# Databases

Business DB:

```text
trader_intelligence
```

MLflow DB:

```text
mlflow_db
```

---

# Why Separate Databases?

Cleaner design.

```text
trader_intelligence
    ↓
Business Data

mlflow_db
    ↓
ML Metadata
```

---

# MLflow Server

Started using:

```powershell
mlflow server `
--backend-store-uri "postgresql://postgres:<password>@localhost:5432/mlflow_db" `
--default-artifact-root "./mlruns" `
--host 127.0.0.1 `
--port 5000
```

---

# Tracking URI

Added:

```python
mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)
```

---

# Why?

Before:

```text
Python
    ↓

Local Files
```

After:

```text
Python
    ↓

MLflow Server
    ↓

PostgreSQL
```

---

# Backend Store vs Artifact Store

## Backend Store

Stores:

```text
Experiments
Runs
Metrics
Parameters
Registry
```

Used:

```text
PostgreSQL
```

---

## Artifact Store

Stores:

```text
CSV
PNG
TXT
Model Files
```

Used:

```text
mlruns/
```

---

# Complete Final Architecture

```text
Trader Intelligence Platform

                     PostgreSQL
                    (mlflow_db)
                           ↑
                           │
                    MLflow Server
                           ↑
                           │
Train Model → MLflow Client
                           │
                           ▼
                      Artifacts
                           │
                           ▼
                        mlruns/

Artifacts:
-----------
classification_report.txt
feature_importance.csv
confusion_matrix.png

Models:
--------
segment_classifier
Version 1
Version 2
Version 3
```

---

# Interview Questions & Answers

## What is MLflow?

Open-source MLOps platform used for experiment tracking, artifact management, model logging, and model versioning.

---

## What is an Experiment?

Collection of related runs.

Example:

```text
future_segment_prediction
```

---

## What is a Run?

Single model training execution.

---

## What is a Metric?

Numeric performance measure.

Examples:

```text
Accuracy
Precision
Recall
F1
```

---

## What is an Artifact?

Generated file.

Examples:

```text
CSV
PNG
TXT
```

---

## What is Model Registry?

Version-controlled repository for models.

Supports:

```text
Versioning
Rollback
Lifecycle Management
```

---

## Why PostgreSQL Instead of SQLite?

```text
Production-grade
Scalable
Multi-user
Already used in project
```

---

## Biggest Learning

The model revealed:

```text
Trader behavior
```

is a stronger predictor of future trader segment than:

```text
Early profitability
```

which is exactly the type of business insight this platform was designed to uncover.

---

# Milestone Achieved

```text
MLFLOW + MLOPS INTEGRATION COMPLETE

✅ Experiment Tracking
✅ Parameter Tracking
✅ Metric Tracking
✅ Artifact Tracking
✅ Feature Importance Analysis
✅ Confusion Matrix Visualization
✅ PostgreSQL Backend
✅ Model Logging
✅ Model Registry
✅ Model Versioning
```
