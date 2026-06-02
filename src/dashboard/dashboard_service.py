from src.pipelines.inference_pipeline import predict_trader_profile


def analyze_trader(trader_features):

    '''
    Dashboard-facing service for trader analysis.

    Accepts trader features and returns
    segment prediction, description,
    and recommendations.
    '''

    return predict_trader_profile(trader_features)