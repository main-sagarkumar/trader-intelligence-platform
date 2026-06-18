"""Tests for FastAPI routes, request validation, and response schemas."""


def test_health_endpoint_returns_healthy_status(api_client):
    """Verify the health endpoint returns a healthy status payload."""
    # Purpose: verify API health endpoint.
    response = api_client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_home_endpoint_returns_application_metadata(api_client):
    """Verify the API root endpoint exposes service metadata."""
    # Purpose: verify API root metadata remains available for clients.
    response = api_client.get("/api/v1/")

    assert response.status_code == 200
    assert response.json()["application"] == "Trader Intelligence Platform"


def test_current_segment_valid_payload_succeeds(api_client, sample_current_feature_payload):
    """Verify a valid current-segment payload returns the expected schema."""
    # Purpose: verify current segment prediction endpoint response schema.
    response = api_client.post("/api/v1/predict-current-segment", json=sample_current_feature_payload)

    assert response.status_code == 200
    assert set(response.json()) == {"cluster", "segment", "description", "recommendations"}


def test_current_segment_missing_field_fails_validation(api_client, sample_current_feature_payload):
    """Verify missing current-segment fields are rejected by validation."""
    # Purpose: ensure missing required fields are rejected by Pydantic validation.
    payload = sample_current_feature_payload.copy()
    payload.pop("win_rate")

    response = api_client.post("/api/v1/predict-current-segment", json=payload)

    assert response.status_code == 422


def test_current_segment_invalid_datatype_fails_validation(api_client, sample_current_feature_payload):
    """Verify invalid current-segment datatypes are rejected before inference."""
    # Purpose: ensure invalid datatypes do not reach model inference.
    payload = sample_current_feature_payload.copy()
    payload["total_trades"] = "many"

    response = api_client.post("/api/v1/predict-current-segment", json=payload)

    assert response.status_code == 422


def test_current_segment_boundary_values_are_accepted(api_client, sample_current_feature_payload):
    """Verify schema-level numeric boundary values are accepted."""
    # Purpose: document schema-level acceptance for numeric boundary values.
    payload = sample_current_feature_payload.copy()
    payload.update({"total_trades": 0, "win_rate": 0.0, "roi_pct": -1.0})

    response = api_client.post("/api/v1/predict-current-segment", json=payload)

    assert response.status_code == 200


def test_future_segment_valid_payload_succeeds(api_client, sample_early_feature_payload):
    """Verify a valid future-segment payload returns the expected schema."""
    # Purpose: verify future segment endpoint response schema.
    response = api_client.post("/api/v1/predict-future-segment", json=sample_early_feature_payload)

    assert response.status_code == 200
    assert set(response.json()) == {"cluster", "confidence", "segment", "description", "recommendations"}
    assert 0.0 <= response.json()["confidence"] <= 1.0


def test_future_segment_missing_field_fails_validation(api_client, sample_early_feature_payload):
    """Verify missing future-segment fields are rejected by validation."""
    # Purpose: ensure future prediction rejects incomplete payloads.
    payload = sample_early_feature_payload.copy()
    payload.pop("early_win_rate")

    response = api_client.post("/api/v1/predict-future-segment", json=payload)

    assert response.status_code == 422


def test_future_segment_invalid_datatype_fails_validation(api_client, sample_early_feature_payload):
    """Verify invalid future-segment datatypes are rejected before inference."""
    # Purpose: ensure invalid future payload values are caught before pipeline execution.
    payload = sample_early_feature_payload.copy()
    payload["early_avg_pnl"] = "profitable"

    response = api_client.post("/api/v1/predict-future-segment", json=payload)

    assert response.status_code == 422
