import pandas as pd
import joblib
import shap

from configs.model_config import (
    EARLY_FEATURES
)

from src.explainability.global_explainer import (
    calculate_global_importance
)

clf = joblib.load(
    "saved_models/segment_classifier.pkl"
)

df = pd.read_csv(
    "data/feature_store/early_trader_features.csv"
)

X = df[EARLY_FEATURES]

explainer = shap.TreeExplainer(clf)

shap_values = explainer(X)

importance = calculate_global_importance(
    shap_values
)

print(importance)