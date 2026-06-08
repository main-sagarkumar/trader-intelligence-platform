from pathlib import Path

import numpy as np
import pandas as pd

from configs.model_config import EARLY_FEATURES

OUTPUT_DIR = Path(
    "outputs/explainability"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def calculate_global_importance(
    shap_values
):

    importance = np.abs(
        shap_values.values
    ).mean(axis=(0, 2))

    results = pd.DataFrame({
        "feature": EARLY_FEATURES,
        "importance": importance
    })

    results = results.sort_values(
        "importance",
        ascending=False
    )

    results.to_csv(
        OUTPUT_DIR /
        "global_feature_importance.csv",
        index=False
    )

    return results