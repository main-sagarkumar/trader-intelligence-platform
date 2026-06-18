"""Tests for local MLflow experiment and run logging behavior."""

import pytest

mlflow = pytest.importorskip("mlflow")


def test_mlflow_experiment_creation_uses_local_tracking_uri(workspace_tmp_dir):
    """Verify MLflow can create experiments using a local tracking directory."""
    # Purpose: verify experiment creation without requiring an external MLflow server.
    mlflow.set_tracking_uri((workspace_tmp_dir / "mlruns").as_uri())

    experiment_id = mlflow.create_experiment("pytest-local-experiment")

    assert experiment_id is not None
    assert mlflow.get_experiment(experiment_id).name == "pytest-local-experiment"


def test_mlflow_logs_metrics_params_and_artifacts(workspace_tmp_dir):
    """Verify MLflow logs params, metrics, and artifacts locally."""
    # Purpose: validate local MLflow logging for metrics, parameters, and artifacts.
    mlflow.set_tracking_uri((workspace_tmp_dir / "mlruns").as_uri())
    mlflow.set_experiment("pytest-logging")
    artifact = workspace_tmp_dir / "artifact.txt"
    artifact.write_text("model-card", encoding="utf-8")

    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "dummy")
        mlflow.log_metric("accuracy", 0.75)
        mlflow.log_artifact(str(artifact))
        run_id = run.info.run_id

    client = mlflow.tracking.MlflowClient()
    stored = client.get_run(run_id)
    artifacts = client.list_artifacts(run_id)

    assert stored.data.params["model_type"] == "dummy"
    assert stored.data.metrics["accuracy"] == 0.75
    assert [item.path for item in artifacts] == ["artifact.txt"]
