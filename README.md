# Trader Intelligence Platform

> Behavioral segmentation, future trader prediction, and personalized recommendations, end-to-end.

---

## Overview

Most trading platforms treat all traders the same, despite significant differences in risk appetite, discipline, leverage usage, and profitability. This platform addresses that gap with a full ML pipeline that:

1. **Segments traders** into behavioral personas using unsupervised clustering.
2. **Predicts a new trader's future segment** from just their first 20 trades.
3. **Delivers personalized recommendations** based on predicted behavior.

The project integrates unsupervised learning, supervised learning, a REST API, an interactive dashboard, automated testing, and Docker, built as a production-style system.

---

## Features

| Area | What it does |
|---|---|
| **Trader Segmentation** | KMeans clustering to discover behavioral personas from historical trade data |
| **Future Segment Prediction** | Random Forest Classifier predicts long-term segment from early activity (first 20 trades) |
| **Recommendation Engine** | Segment-specific, explainable recommendations tailored to each trader |
| **API Layer** | FastAPI with versioned endpoints, Swagger docs, and Pydantic validation |
| **Dashboard** | Streamlit app with interactive prediction interface and business-friendly visuals |
| **Engineering** | Dockerized, config-driven, modular, with PyTest coverage |

---

## Architecture

```
Raw Trade Data
     ↓
Feature Engineering
     ↓
Trader Feature Matrix
     ↓
KMeans Clustering ──────────────────────→ Trader Segments (labels)
     ↓                                              ↓
Early Feature Extraction             Segment Profiles & Insights
(first 20 trades)
     ↓
Random Forest Classifier
     ↓
Future Segment Prediction + Confidence Score
     ↓
Recommendation Engine
     ↓
FastAPI  /  Streamlit
```

---

## ML Pipeline

### Stage 1: Unsupervised Learning (Segmentation)

Discovers naturally occurring trader personas from complete trade history.

- **Model:** KMeans Clustering
- **Input:** Aggregated trader behavior features
- **Output:** Trader segment labels + cluster profiles

### Stage 2: Supervised Learning (Future Prediction)

Predicts a new trader's long-term segment using only their first 20 trades.

- **Model:** Random Forest Classifier
- **Input Features:**

  | Feature | Description |
  |---|---|
  | Average PnL | Mean profit/loss per trade |
  | Win Rate | Percentage of winning trades |
  | Holding Duration | Avg time a position is held |
  | Leverage Usage | Avg leverage applied |
  | Risk Percentage | Capital risked per trade |
  | Stop Loss Usage | Frequency of stop-loss orders |
  | Overnight Exposure | Frequency of positions held overnight |

- **Output:** Predicted future segment + confidence score

---

## Project Structure

```
trader-intelligence-platform/
│
├── app/                        # Streamlit application entry point
├── configs/                    # Config files (model params, paths, etc.)
├── notebooks/                  # Exploratory notebooks
├── saved_models/               # Serialized trained models
│
├── src/
│   ├── api/                    # FastAPI routes and schemas
│   ├── clustering/             # KMeans training and profiling
│   ├── dashboard/              # Streamlit UI components
│   ├── eda/                    # Exploratory data analysis scripts
│   ├── feature_engineering/    # Feature extraction and transformation
│   ├── ingestion/              # Data loading and validation
│   ├── modeling/               # Classifier training and evaluation
│   ├── pipelines/              # End-to-end pipeline orchestration
│   ├── prediction/             # Inference logic
│   └── validation/             # Input/output schema validation
│
├── tests/                      # PyTest test suite
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## API Reference

Base URL: `http://localhost:8000/api/v1`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `POST` | `/predict-segment` | Segment a trader based on full trade history |
| `POST` | `/predict-future-segment` | Predict future segment from first 20 trades |

Full Swagger documentation is available at `http://localhost:8000/docs` when the server is running.

---

## Local Setup

**Prerequisites:** Python 3.8+

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn src.api.main:app --reload

# Start the Streamlit dashboard (separate terminal)
streamlit run app/streamlit_app.py
```

- API: `http://localhost:8000`
- Dashboard: `http://localhost:8501`

---

## Docker

```bash
# Build the image
docker build -t trader-intelligence-api .

# Run the container
docker run -p 8000:8000 trader-intelligence-api
```

---

## Testing

```bash
pytest -v
```

Current test coverage includes:

- Prediction pipeline (input → output correctness)
- API endpoints (request/response validation)
- Response schema validation

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-Learn |
| API | FastAPI |
| Dashboard | Streamlit |
| Testing | PyTest |
| Containerization | Docker |
| Cloud (planned) | Google Cloud Platform |

---

## Roadmap

- [ ] Model monitoring and alerting
- [ ] Data drift detection
- [ ] Feature store integration
- [ ] CI/CD pipeline
- [ ] Cloud Run deployment
- [ ] Real trading platform integration

---

## Author

**Sagar Kumar**
