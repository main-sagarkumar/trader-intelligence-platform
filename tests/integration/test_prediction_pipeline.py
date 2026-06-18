"""Integration tests for current and future prediction pipelines."""

import numpy as np

from src.pipelines.future_prediction_pipeline import predict_future_trader_profile
from src.pipelines.inference_pipeline import predict_trader_profile


def test_current_prediction_pipeline_end_to_end_with_mocked_models(monkeypatch, sample_current_feature_payload):
    """Verify current prediction pipeline returns a complete trader profile."""
    # Purpose: verify current prediction pipeline flow from features to business profile.
    class DummyScaler:
        def transform(self, frame):
            """Return raw numeric values while matching the scaler API."""
            return frame.to_numpy()

    class DummyKMeans:
        def predict(self, frame):
            """Return a deterministic cluster label for integration testing."""
            return np.array([3])

    monkeypatch.setattr(
        "src.prediction.trader_segment_predictor.load_models",
        lambda: (DummyScaler(), DummyKMeans()),
    )

    result = predict_trader_profile(sample_current_feature_payload)

    assert result["cluster"] == 3
    assert {"segment", "description", "recommendations"}.issubset(result)


def test_future_prediction_pipeline_end_to_end_with_mocked_model(
    monkeypatch,
    trained_dummy_model,
    sample_early_feature_payload,
):
    """Verify future prediction pipeline returns confidence and recommendations."""
    # Purpose: verify future prediction flow from early features to recommendations.
    monkeypatch.setattr("src.prediction.future_segment_predictor.load_model", lambda: trained_dummy_model)

    result = predict_future_trader_profile(sample_early_feature_payload)

    assert result["cluster"] == 0
    assert 0.0 <= result["confidence"] <= 1.0
    assert {"segment", "description", "recommendations"}.issubset(result)
