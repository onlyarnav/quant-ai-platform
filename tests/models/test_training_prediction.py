"""Training, local tracking, registry, and prediction smoke tests."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone
from config.settings import settings
from src.models.predict import predict
from src.models.registry import get_latest_model
from src.models.train import (
    drift_check,
    retrain_if_due,
    should_retrain,
    train_all_models,
    train_model,
    walk_forward_validate,
)


def feature_file(path):
    """Write a compact deterministic feature panel for model tests."""
    rng = np.random.default_rng(5)
    rows = []
    for symbol in ("AAA", "BBB", "CCC"):
        x = rng.normal(size=100)
        for i, value in enumerate(x):
            rows.append(
                {
                    "symbol": symbol,
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                    "rsi_14": value,
                    "roc_5": value * 0.2,
                    "future_return_1d": value * 0.01 + rng.normal(scale=0.001),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_parquet(path, index=False)
    return frame


def test_train_registry_prediction_and_due(tmp_path, monkeypatch):
    """Local MLflow tracking and saved model reproduce predictions exactly."""
    data_path = tmp_path / "features.parquet"
    frame = feature_file(data_path)
    monkeypatch.setattr(
        settings, "MLFLOW_TRACKING_URI", f"file://{tmp_path / 'mlruns'}"
    )
    monkeypatch.setattr(settings, "MODEL_ARTIFACT_DIR", tmp_path / "models")
    monkeypatch.setattr(settings, "TRAIN_RATIO", 0.65)
    monkeypatch.setattr(settings, "VALIDATION_RATIO", 0.15)
    monkeypatch.setattr(settings, "TEST_RATIO", 0.20)
    result = train_model(data_path, "ridge", params={"alpha": 0.1})
    loaded = get_latest_model("ridge", tmp_path / "models")
    assert loaded["feature_columns"] == ["rsi_14", "roc_5"]
    assert loaded["dataset_hash"]
    assert loaded["target_column"] == "future_return_1d"
    assert loaded["train_date_range"]
    probe = frame.iloc[-4:-1][loaded["feature_columns"]]
    np.testing.assert_allclose(
        result["model"].predict(probe), loaded["model"].predict(probe)
    )
    assert should_retrain(loaded)[0] is False
    output = predict(
        data_path,
        "ridge",
        output_dir=tmp_path / "predictions",
        artifact_dir=tmp_path / "models",
    )
    assert output.columns.tolist() == [
        "symbol",
        "date",
        "predicted_return",
        "model_name",
        "model_version",
    ]
    assert len(output) == len(frame)
    future_rows = frame.loc[frame["date"].eq(frame["date"].max())].copy()
    future_rows["date"] += pd.Timedelta(days=1)
    future_rows["rsi_14"] += 100
    extended_path = tmp_path / "extended.parquet"
    pd.concat([frame, future_rows], ignore_index=True).to_parquet(
        extended_path, index=False
    )
    extended = predict(
        extended_path,
        "ridge",
        output_dir=tmp_path / "extended_predictions",
        artifact_dir=tmp_path / "models",
    )
    compare = output.merge(extended, on=["symbol", "date"], suffixes=("_old", "_new"))
    np.testing.assert_allclose(
        compare["predicted_return_old"], compare["predicted_return_new"]
    )
    future_rows = future_rows.copy()
    future_rows.loc[future_rows["symbol"] == "AAA", "date"] += pd.Timedelta(days=1)
    uneven_path = tmp_path / "uneven.parquet"
    pd.concat([frame, future_rows], ignore_index=True).to_parquet(
        uneven_path, index=False
    )
    latest = predict(
        uneven_path,
        "ridge",
        output_dir=tmp_path / "latest_predictions",
        artifact_dir=tmp_path / "models",
        latest_per_symbol=True,
    )
    assert len(latest) == 3
    latest_dates = latest.set_index("symbol")["date"]
    assert latest_dates["AAA"] > latest_dates["BBB"]
    tracking_path = tmp_path / "mlruns"
    assert tracking_path.exists()
    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=f"file://{tracking_path}")
    runs = client.search_runs(
        [client.get_experiment_by_name(settings.MLFLOW_EXPERIMENT_NAME).experiment_id]
    )
    assert runs
    assert "model_type" in runs[0].data.params
    assert "test_rmse" in runs[0].data.metrics
    assert client.list_artifacts(runs[0].info.run_id)


def test_all_registered_models_compare_and_retrain_when_missing(tmp_path, monkeypatch):
    """Comparison runs the registry; a missing artifact triggers retraining."""
    try:
        import xgboost  # noqa: F401
        import lightgbm  # noqa: F401
    except Exception as exc:
        pytest.skip(f"all-model comparison requires optional boosters: {exc}")
    data_path = tmp_path / "features.parquet"
    feature_file(data_path)
    monkeypatch.setattr(
        settings, "MLFLOW_TRACKING_URI", f"file://{tmp_path / 'all_models'}"
    )
    monkeypatch.setattr(settings, "MODEL_ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(
        settings,
        "RANDOM_FOREST_PARAMS",
        {"n_estimators": 3, "min_samples_leaf": 2},
    )
    monkeypatch.setattr(
        settings,
        "GRADIENT_BOOSTING_PARAMS",
        {"n_estimators": 3, "learning_rate": 0.05, "max_depth": 2},
    )
    monkeypatch.setattr(
        settings,
        "XGBOOST_PARAMS",
        {"n_estimators": 3, "max_depth": 2, "n_jobs": 1},
    )
    monkeypatch.setattr(
        settings,
        "LIGHTGBM_PARAMS",
        {"n_estimators": 3, "verbosity": -1},
    )
    comparison = train_all_models(data_path)
    assert set(comparison["model_name"]) == {
        "linear",
        "ridge",
        "lasso",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "lightgbm",
    }
    assert {"rmse", "mae", "r2"} <= set(comparison.columns)
    monkeypatch.setattr(settings, "MODEL_ARTIFACT_DIR", tmp_path / "retrain_artifacts")
    newly_trained = retrain_if_due(data_path, "ridge")
    assert newly_trained is not None
    assert newly_trained["metadata"]["model_name"] == "ridge"


def test_walk_forward_metrics_and_drift_hook(tmp_path, monkeypatch):
    """Walk-forward folds aggregate metrics and the drift hook is explicit."""
    data_path = tmp_path / "features.parquet"
    feature_file(data_path)
    monkeypatch.setattr(settings, "WALK_FORWARD_INITIAL_TRAIN_SIZE", 50)
    monkeypatch.setattr(settings, "WALK_FORWARD_TEST_SIZE", 10)
    monkeypatch.setattr(settings, "WALK_FORWARD_STEP_SIZE", 10)
    monkeypatch.setattr(
        settings, "MLFLOW_TRACKING_URI", f"file://{tmp_path / 'walk_mlruns'}"
    )
    result = walk_forward_validate(data_path, "ridge", params={"alpha": 0.1})
    assert len(result["fold_metrics"]) >= 3
    assert "rmse" in result["mean_metrics"]
    assert drift_check({"ic": -0.2, "rmse": 2.0}, {"ic": 0.1, "rmse": 1.0})[0]


def test_prediction_rejects_feature_order_mismatch(tmp_path, monkeypatch):
    """Incoming model columns must match the stored order exactly."""
    path = tmp_path / "features.parquet"
    frame = feature_file(path)
    monkeypatch.setattr(
        settings, "MLFLOW_TRACKING_URI", f"file://{tmp_path / 'tracking'}"
    )
    result = train_model(
        path, "ridge", params={"alpha": 0.1}, artifact_dir=tmp_path / "artifacts"
    )
    reversed_path = tmp_path / "bad.parquet"
    frame[["symbol", "date", "roc_5", "rsi_14", "future_return_1d"]].to_parquet(
        reversed_path, index=False
    )
    with pytest.raises(ValueError, match="mismatch"):
        predict(
            reversed_path,
            "ridge",
            output_dir=tmp_path / "out",
            artifact_dir=tmp_path / "artifacts",
        )


def test_prediction_skips_nonfinite_features_and_honors_date_range(
    tmp_path, monkeypatch
):
    """Prediction drops invalid rows and applies inclusive date bounds."""
    path = tmp_path / "features.parquet"
    frame = feature_file(path)
    monkeypatch.setattr(
        settings, "MLFLOW_TRACKING_URI", f"file://{tmp_path / 'date_tracking'}"
    )
    train_model(
        path, "ridge", params={"alpha": 0.1}, artifact_dir=tmp_path / "artifacts"
    )
    probe = frame.iloc[:4].copy()
    probe.loc[probe.index[2], "rsi_14"] = np.nan
    probe.loc[probe.index[1], "roc_5"] = np.inf
    probe_path = tmp_path / "probe.parquet"
    probe.to_parquet(probe_path, index=False)
    output = predict(
        probe_path,
        "ridge",
        start_date="2024-01-02",
        end_date="2024-01-04",
        output_dir=tmp_path / "out",
        artifact_dir=tmp_path / "artifacts",
    )
    assert len(output) == 1
    assert output["date"].iloc[0] == pd.Timestamp("2024-01-04")
    empty = frame.iloc[:0].copy()
    empty_path = tmp_path / "empty.parquet"
    empty.to_parquet(empty_path, index=False)
    no_rows = predict(
        empty_path,
        "ridge",
        output_dir=tmp_path / "empty_out",
        artifact_dir=tmp_path / "artifacts",
    )
    assert no_rows.empty
    assert no_rows.columns.tolist() == [
        "symbol",
        "date",
        "predicted_return",
        "model_name",
        "model_version",
    ]
