"""Tests for model loading paths and serialized model behavior."""

import joblib

from src.prediction.future_segment_predictor import load_model
from src.prediction.trader_segment_predictor import load_models


def test_future_model_loads_from_configured_path(monkeypatch, trained_dummy_model):
    """Verify the future classifier loader uses the configured model path."""
    # Purpose: verify model loading delegates to joblib with the configured model file.
    seen_paths = []

    def fake_load(path):
        """Return a dummy classifier while capturing the requested path."""
        seen_paths.append(path)
        return trained_dummy_model

    monkeypatch.setattr("src.prediction.future_segment_predictor.joblib.load", fake_load)

    model = load_model()

    assert model is trained_dummy_model
    assert seen_paths[0].name == "segment_classifier.pkl"


def test_current_model_loader_loads_scaler_and_kmeans(monkeypatch):
    """Verify current inference loads scaler and KMeans artifacts."""
    # Purpose: verify both current inference artifacts are loaded in order.
    loaded_names = []

    def fake_load(path):
        """Return artifact names while recording load order."""
        loaded_names.append(path.name)
        return path.name

    monkeypatch.setattr("src.prediction.trader_segment_predictor.joblib.load", fake_load)

    scaler, kmeans = load_models()

    assert scaler == "scaler.pkl"
    assert kmeans == "kmeans_model.pkl"
    assert loaded_names == ["scaler.pkl", "kmeans_model.pkl"]


def test_joblib_round_trip_preserves_predict_proba(trained_dummy_model, temp_output_directory):
    """Verify a serialized classifier preserves prediction methods."""
    # Purpose: ensure serialized classifier artifacts retain prediction methods.
    model_file = temp_output_directory / "segment_classifier.pkl"

    joblib.dump(trained_dummy_model, model_file)
    loaded_model = joblib.load(model_file)

    assert hasattr(loaded_model, "predict")
    assert hasattr(loaded_model, "predict_proba")
