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
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                            precision_score,recall_score,f1_score,)

from sklearn.model_selection import train_test_split

from configs.model_config import (
    EARLY_FEATURES,
    TARGET_COLUMN,
    RANDOM_STATE,
    RF_N_ESTIMATORS,
    TEST_SIZE,
)
from configs.paths_config import FEATURE_STORE_DIR, MODEL_DIR

# Adding ML FLow
import mlflow
from mlflow.sklearn import log_model

from src.monitoring.mlflow_tracker import start_experiment

import matplotlib.pyplot as plt
import seaborn as sns

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

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test,predictions,)

    precision = precision_score(
        y_test,predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE")
    print("=" * 80)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(report)

    # Save report for MLflow 

    report_file = (
        FEATURE_STORE_DIR.parent
        / "reports"
        / "classification_report.txt"
    )

    report_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(type(report))

    with open(report_file, "w") as f:
        f.write(str(report)) 

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    print("\nConfusion Matrix:")
    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "report_file": report_file,
        "confusion_matrix": cm,
    }

def save_confusion_matrix(cm):
    """Save confusion matrix image."""

    output_file = (
        FEATURE_STORE_DIR.parent
        / "reports"
        / "confusion_matrix.png"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(output_file)

    plt.close()

    return output_file


# Feature Importance

def show_feature_importance(model):
    """
    Display feature importance rankings..
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

    output_file = (
        FEATURE_STORE_DIR.parent /
        "reports" /
        "feature_importance.csv"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance_df.to_csv(
        output_file,
        index=False,
    )
    return output_file


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

    with start_experiment(
        "future_segment_prediction"
    ):

        df = load_data()

        X_train, X_test, y_train, y_test = prepare_data(df)

        model = train_model(
            X_train,
            y_train,
        )

        # Model settings
        mlflow.log_param(
            "n_estimators",
            RF_N_ESTIMATORS,
        )

        mlflow.log_param(
            "test_size",
            TEST_SIZE,
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        # Performance metrics
        mlflow.log_metrics(
            {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
            }
        )
    
        file = show_feature_importance(model)

        cm_png = save_confusion_matrix(metrics["confusion_matrix"])

        # Upload the artifacts
        mlflow.log_artifact(metrics['report_file'])
        mlflow.log_artifact(str(file)) 
        mlflow.log_artifact(str(cm_png))  

        save_model(model)

        # Log Model
        model_info = log_model(
            sk_model=model,
            name="segment_classifier",
        )

        # Register model version
        mlflow.register_model(
            model_uri=model_info.model_uri,
            name = "segment_classifier"
        )

        print(f"\nModel saved to:\n{MODEL_FILE}")