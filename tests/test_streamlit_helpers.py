"""Tests for dashboard and Streamlit-facing helper functions."""

import pytest

from src.analytics.cluster_analytics import get_cluster_summary
from src.analytics.insights import generate_insights
from src.analytics.kpi_service import get_kpis
from src.dashboard.dashboard_service import analyze_trader


def test_get_kpis_returns_expected_metric_aggregations(sample_feature_data):
    """Verify KPI helper returns expected aggregate metrics."""
    # Purpose: validate dashboard KPI helper calculations.
    kpis = get_kpis(sample_feature_data)

    assert kpis["total_traders"] == len(sample_feature_data)
    assert kpis["avg_roi"] == pytest.approx(sample_feature_data["roi_pct"].mean())
    assert kpis["avg_win_rate"] == pytest.approx(sample_feature_data["win_rate"].mean())
    assert kpis["total_pnl"] == sample_feature_data["total_pnl"].sum()


def test_get_cluster_summary_aggregates_by_cluster(sample_feature_data):
    """Verify cluster summary helper aggregates trader rows by cluster."""
    # Purpose: validate cluster dashboard aggregation helper output.
    summary = get_cluster_summary(sample_feature_data)

    assert set(["cluster", "trader_count", "avg_roi", "avg_win_rate", "avg_leverage"]).issubset(summary.columns)
    assert summary["trader_count"].sum() == len(sample_feature_data)


def test_generate_insights_returns_dashboard_strings(sample_feature_data):
    """Verify automated insight helper returns non-empty strings."""
    # Purpose: ensure automated insight helper produces user-facing insight entries.
    insights = generate_insights(sample_feature_data)

    assert len(insights) == 4
    assert all(isinstance(item, str) and item for item in insights)


def test_dashboard_analyze_trader_delegates_to_prediction_pipeline(monkeypatch, sample_current_feature_payload):
    """Verify dashboard service delegates trader analysis to prediction."""
    # Purpose: verify Streamlit-facing helper can be tested without launching UI.
    expected = {"cluster": 1, "segment": "Conservative", "description": "desc", "recommendations": []}
    monkeypatch.setattr("src.dashboard.dashboard_service.predict_trader_profile", lambda _: expected)

    result = analyze_trader(sample_current_feature_payload)

    assert result == expected
