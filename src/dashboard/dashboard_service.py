"""
Expose dashboard-facing trader analysis helpers.

This module keeps Streamlit pages thin by delegating prediction workflows to
the reusable inference pipeline.
"""

from src.pipelines.inference_pipeline import predict_trader_profile


def analyze_trader(trader_features):

    '''
    Dashboard-facing service for trader analysis.

    Accepts trader features and returns
    segment prediction, description,
    and recommendations.
    '''

    return predict_trader_profile(trader_features)
