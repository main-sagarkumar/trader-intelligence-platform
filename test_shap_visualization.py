import pandas as pd
import joblib
import shap

from configs.model_config import EARLY_FEATURES

from src.explainability.shap_visualizer import (
    generate_summary_plot,
    generate_bar_plot
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

generate_summary_plot(
    shap_values,
    X
)

generate_bar_plot(
    shap_values,
    X
)

print("SHAP visualizations saved.")