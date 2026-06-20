"""
Central MLflow utility.

Purpose:
- Create experiments
- Start runs
- Keep MLflow logic reusable
"""

import mlflow

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)


def start_experiment(experiment_name: str):
    """
    Create experiment if it doesn't exist
    and start a new run.
    """

    mlflow.set_experiment(experiment_name)

    return mlflow.start_run()