"""
Central MLflow utility.

Purpose:
- Create experiments
- Start runs
- Keep MLflow logic reusable
"""

import mlflow


def start_experiment(experiment_name: str):
    """
    Create experiment if it doesn't exist
    and start a new run.
    """

    mlflow.set_experiment(experiment_name)

    return mlflow.start_run()