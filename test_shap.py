import pandas as pd
import joblib
import shap

from configs.model_config import EARLY_FEATURES
from src.explainability.explanation_service import (
    build_detailed_explanations
)

# Load model
clf = joblib.load(
    "saved_models/segment_classifier.pkl"
)

# Load early features
df = pd.read_csv(
    "data/feature_store/early_trader_features.csv"
)

X = df[EARLY_FEATURES]

# Take one trader
sample = X.iloc[[0]]

# Create SHAP explainer
explainer = shap.TreeExplainer(clf)

# Generate SHAP values
values = explainer(sample)

# Predict cluster
prediction = clf.predict(sample)[0]

# Generate explanations
explanations = build_detailed_explanations(
    values,
    sample,
    prediction
)

print("Predicted Cluster:", prediction)
print("Explanations:")
print(explanations)