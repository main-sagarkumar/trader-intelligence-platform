"""Tests for current and future trader prediction workflows."""

import numpy as np
import pandas as pd

from configs.model_config import EARLY_FEATURES
from src.pipelines.future_prediction_pipeline import predict_future_trader_profile
from src.pipelines.inference_pipeline import predict_trader_profile
from src.prediction.future_segment_predictor import predict_future_profile
from src.prediction.recommendation_engine import get_trader_recommendations
from src.prediction.trader_segment_predictor import predict_cluster


def test_future_predictor_returns_probability_between_zero_and_one(
    monkeypatch,
    trained_dummy_model,
    sample_early_feature_payload,
):
    """Verify future prediction returns a bounded confidence score."""
    # Purpose: verify future prediction schema and bounded confidence using a mocked model load.
    monkeypatch.setattr("src.prediction.future_segment_predictor.load_model", lambda: trained_dummy_model)

    result = predict_future_profile(sample_early_feature_payload)

    assert result["cluster"] == 0
    assert 0.0 <= result["confidence"] <= 1.0


def test_future_predictor_uses_expected_feature_order(monkeypatch, sample_early_feature_payload):
    """Verify future predictor preserves configured model feature order."""
    # Purpose: ensure model input columns match the supervised training feature order.
    captured_columns = []

    class RecordingModel:
        def predict(self, frame):
            """Capture model input columns and return a deterministic class."""
            captured_columns.extend(frame.columns.tolist())
            return np.array([1])

        def predict_proba(self, frame):
            """Return deterministic class probabilities for confidence calculation."""
            return np.array([[0.1, 0.9]])

    monkeypatch.setattr("src.prediction.future_segment_predictor.load_model", lambda: RecordingModel())

    predict_future_profile(sample_early_feature_payload)

    assert captured_columns == EARLY_FEATURES


def test_predict_cluster_returns_integer_label(monkeypatch, sample_current_feature_payload):
    """Verify current-segment prediction returns a plain integer label."""
    # Purpose: verify current segment prediction scales features and returns a plain integer.
    class DummyScaler:
        def transform(self, frame):
            """Return raw numeric values while preserving scaler interface."""
            return frame.to_numpy()

    class DummyKMeans:
        def predict(self, frame):
            """Return a deterministic cluster label for current inference."""
            return np.array([4])

    monkeypatch.setattr(
        "src.prediction.trader_segment_predictor.load_models",
        lambda: (DummyScaler(), DummyKMeans()),
    )

    cluster = predict_cluster(sample_current_feature_payload)

    assert cluster == 4
    assert isinstance(cluster, int)


def test_predict_trader_profile_returns_expected_schema(monkeypatch, sample_current_feature_payload):
    """Verify current inference pipeline returns the business profile schema."""
    # Purpose: validate current inference pipeline response composition.
    monkeypatch.setattr("src.pipelines.inference_pipeline.predict_cluster", lambda _: 1)

    result = predict_trader_profile(sample_current_feature_payload)

    assert set(result) == {"cluster", "segment", "description", "recommendations"}
    assert result["cluster"] == 1
    assert isinstance(result["recommendations"], list)


def test_future_prediction_pipeline_merges_prediction_and_recommendations(
    monkeypatch,
    sample_early_feature_payload,
):
    """Verify future pipeline merges model output with recommendations."""
    # Purpose: verify future pipeline output schema without loading a real model.
    monkeypatch.setattr(
        "src.pipelines.future_prediction_pipeline.predict_future_profile",
        lambda _: {"cluster": 2, "confidence": 0.81},
    )

    result = predict_future_trader_profile(sample_early_feature_payload)

    assert result["cluster"] == 2
    assert result["confidence"] == 0.81
    assert {"segment", "description", "recommendations"}.issubset(result)


def test_recommendation_engine_returns_known_cluster_metadata():
    """Verify recommendation metadata exists for configured clusters."""
    # Purpose: ensure cluster definitions are available for API and dashboards.
    result = get_trader_recommendations(0)

    assert result["segment"]
    assert result["description"]
    assert isinstance(result["recommendations"], list)
    assert len(result["recommendations"]) > 0


def test_predict_proba_output_dimensions(trained_dummy_model):
    """Verify model probability output dimensions and bounds."""
    # Purpose: verify classifier fixtures and model-like objects expose probabilities.
    frame = pd.DataFrame([dict(zip(EARLY_FEATURES, range(1, len(EARLY_FEATURES) + 1)))])

    probabilities = trained_dummy_model.predict_proba(frame)

    assert probabilities.shape == (1, len(trained_dummy_model.classes_))
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
