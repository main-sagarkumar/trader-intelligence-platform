"""
Segment Classifier Training
---------------------------
Trains a supervised learning model to predict a trader's
future segment using only their first 20 trades.

Pipeline:
1. Load early trader features
2. Split into train and test sets
3. Train Random Forest classifier
4. Evaluate model performance
5. Analyze feature importance
6. Save trained model
"""

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from configs.model_config import (
    EARLY_FEATURES,
    TARGET_COLUMN,
    RANDOM_STATE,
    RF_N_ESTIMATORS,
    TEST_SIZE,
)
from configs.paths_config import FEATURE_STORE_DIR, MODEL_DIR



# Constants

INPUT_FILE = FEATURE_STORE_DIR / "early_trader_features.csv"
MODEL_FILE = MODEL_DIR / "segment_classifier.pkl"



# Data Loading

def load_data() -> pd.DataFrame:
    """
    Load supervised learning dataset.

    Returns:
        Trader-level feature dataset containing
        early behaviour features and cluster labels.
    """
    return pd.read_csv(INPUT_FILE)



# Train-Test Split

def prepare_data(df: pd.DataFrame):
    """
    Split dataset into training and testing sets.

    Args:
        df:
            Early trader feature dataset.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df[EARLY_FEATURES]
    y = df[TARGET_COLUMN]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )



# Model Training

def train_model(X_train, y_train) -> RandomForestClassifier:
    """
    Train Random Forest classifier.

    Args:
        X_train:
            Training features.

        y_train:
            Training labels.

    Returns:
        Trained model.
    """
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    return model



# Model Evaluation

def evaluate_model(model, X_test, y_test) -> None:
    """
    Evaluate model performance on unseen data.

    Args:
        model:
            Trained classifier.

        X_test:
            Test features.

        y_test:
            Test labels.
    """
    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE")
    print("=" * 80)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )



# Feature Importance

def show_feature_importance(model) -> None:
    """
    Display feature importance rankings.

    Args:
        model:
            Trained Random Forest model.
    """
    importance_df = (
        pd.DataFrame(
            {
                "feature": EARLY_FEATURES,
                "importance": model.feature_importances_,
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)

    print(importance_df)


# Model Persistence

def save_model(model) -> None:
    """
    Save trained model for future inference.

    Args:
        model:
            Trained classifier.
    """
    joblib.dump(
        model,
        MODEL_FILE,
    )


# Entry Point

if __name__ == "__main__":

    df = load_data()

    X_train, X_test, y_train, y_test = prepare_data(df)

    model = train_model(
        X_train,
        y_train,
    )

    evaluate_model(
        model,
        X_test,
        y_test,
    )

    show_feature_importance(model)

    save_model(model)

    print(f"\nModel saved to:\n{MODEL_FILE}")