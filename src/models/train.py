"""Chronological model fitting, evaluation, tracking, and retraining hooks."""

from datetime import datetime, timedelta, timezone
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import settings
from src.models.dataset import (
    clean_feature_data,
    load_feature_data,
    split_features_and_target,
)
from src.models.evaluation import evaluate_predictions
from src.models.model_factory import REGISTRY, get_model
from src.models.registry import dataset_hash, save_model
from src.models.splitting import chronological_split, walk_forward_splits

logger = logging.getLogger(__name__)


def _gap_for_target(target_column: str) -> int:
    """Read the forward horizon from a target label to purge overlaps."""
    try:
        return int(target_column.removeprefix("future_return_").removesuffix("d"))
    except ValueError as exc:
        raise ValueError(
            f"Cannot determine purge gap from target {target_column!r}"
        ) from exc


def _log_mlflow(
    model_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    metadata: dict[str, Any],
    data_path: Path,
) -> None:
    """Track a completed run; connection and missing-library errors are explicit."""
    if settings.MLFLOW_TRACKING_URI.startswith("file:"):
        # Recent MLflow releases require an explicit opt-in to their supported
        # local file-store mode; this is enabled only for a configured file URI.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is required for training; install dependencies or configure a file: tracking URI"
        ) from exc
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    try:
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run():
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("hyperparameters", str(params))
            mlflow.log_param("dataset_path", str(data_path))
            mlflow.log_param("dataset_hash", metadata["dataset_hash"])
            mlflow.log_param("features", ",".join(metadata["feature_columns"]))
            mlflow.log_param("feature_policy", metadata["feature_policy"])
            mlflow.log_param("target_column", metadata["target_column"])
            mlflow.log_param("split_boundaries", str(metadata["split_boundaries"]))
            mlflow.log_param("model_version", metadata["version"])
            mlflow.log_metrics(
                {k: float(v) for k, v in metrics.items() if np.isfinite(v)}
            )
            mlflow.log_artifact(metadata["artifact_path"])
            mlflow.log_artifact(
                str(Path(metadata["artifact_path"]).with_suffix(".json"))
            )
    except Exception as exc:
        raise RuntimeError(
            f"MLflow tracking failed at {settings.MLFLOW_TRACKING_URI}: {exc}"
        ) from exc


def train_model(
    data_path: str | Path,
    model_name: str = "ridge",
    target_column: str | None = None,
    params: dict[str, Any] | None = None,
    feature_policy: str | None = None,
    artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Fit one model, assess chronological holdouts, save artifact and log run."""
    if not np.isclose(
        settings.TRAIN_RATIO + settings.VALIDATION_RATIO + settings.TEST_RATIO, 1.0
    ):
        raise ValueError("TRAIN_RATIO, VALIDATION_RATIO, and TEST_RATIO must sum to 1")
    path = Path(data_path)
    target = target_column or settings.DEFAULT_TARGET_COLUMN
    frame = clean_feature_data(load_feature_data(path), target, feature_policy)
    X, y, dates = split_features_and_target(frame, target, feature_policy)
    train_idx, val_idx, test_idx = chronological_split(
        dates, settings.TRAIN_RATIO, settings.VALIDATION_RATIO, _gap_for_target(target)
    )
    estimator = get_model(model_name, params)
    try:
        estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
        val_metrics = evaluate_predictions(
            y.iloc[val_idx], estimator.predict(X.iloc[val_idx])
        )
        test_metrics = evaluate_predictions(
            y.iloc[test_idx], estimator.predict(X.iloc[test_idx])
        )
        metrics = {f"val_{k}": v for k, v in val_metrics.items()} | {
            f"test_{k}": v for k, v in test_metrics.items()
        }
        digest = dataset_hash(path)
        metadata = save_model(
            estimator,
            model_name,
            target,
            X.columns.tolist(),
            (dates.iloc[train_idx].min(), dates.iloc[train_idx].max()),
            metrics,
            digest,
            artifact_dir,
            feature_policy or settings.ML_FEATURE_POLICY,
        )
        metadata["split_boundaries"] = {
            "train_end": str(dates.iloc[train_idx].max()),
            "validation_start": str(dates.iloc[val_idx].min()),
            "test_start": str(dates.iloc[test_idx].min()),
        }
        metadata["dataset_path"] = str(path)
        _log_mlflow(model_name, params or {}, metrics, metadata, path)
        return {"model": estimator, "metadata": metadata, "metrics": metrics}
    except Exception:
        logger.exception(
            "Training failed for %s; artifact is not registered until successful fit",
            model_name,
        )
        raise


def walk_forward_validate(
    data_path: str | Path,
    model_name: str = "ridge",
    target_column: str | None = None,
    params: dict[str, Any] | None = None,
    feature_policy: str | None = None,
) -> dict[str, Any]:
    """Fit and measure model over purged expanding or rolling date folds."""
    path = Path(data_path)
    target = target_column or settings.DEFAULT_TARGET_COLUMN
    frame = clean_feature_data(load_feature_data(path), target, feature_policy)
    X, y, dates = split_features_and_target(frame, target, feature_policy)
    records: list[dict[str, float]] = []
    folds = walk_forward_splits(
        dates,
        settings.WALK_FORWARD_INITIAL_TRAIN_SIZE,
        settings.WALK_FORWARD_TEST_SIZE,
        settings.WALK_FORWARD_STEP_SIZE,
        gap=_gap_for_target(target),
    )
    for fold, (train_idx, test_idx) in enumerate(folds):
        estimator = get_model(model_name, params)
        estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
        metrics = evaluate_predictions(
            y.iloc[test_idx], estimator.predict(X.iloc[test_idx])
        )
        records.append(metrics)
        logger.info("Walk-forward fold %d metrics: %s", fold, metrics)
    if not records:
        raise ValueError("Dataset is too short for configured walk-forward windows")
    summary = pd.DataFrame(records).mean(numeric_only=True).to_dict()
    try:
        if settings.MLFLOW_TRACKING_URI.startswith("file:"):
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run():
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("dataset_path", str(path))
            mlflow.log_param("dataset_hash", dataset_hash(path))
            mlflow.log_param("features", ",".join(X.columns))
            mlflow.log_param("target_column", target)
            mlflow.log_metrics(
                {f"mean_{k}": float(v) for k, v in summary.items() if np.isfinite(v)}
            )
            for i, row in enumerate(records):
                mlflow.log_metrics(
                    {
                        f"fold_{i}_{k}": float(v)
                        for k, v in row.items()
                        if np.isfinite(v)
                    }
                )
    except ImportError as exc:
        raise RuntimeError("MLflow is required for walk-forward tracking") from exc
    except Exception as exc:
        raise RuntimeError(f"MLflow tracking failed: {exc}") from exc
    return {"fold_metrics": records, "mean_metrics": summary}


def train_all_models(
    data_path: str | Path,
    target_column: str | None = None,
    feature_policy: str | None = None,
) -> pd.DataFrame:
    """Train all registered models and compare test metrics."""
    rows = []
    for name in REGISTRY:
        result = train_model(
            data_path, name, target_column, feature_policy=feature_policy
        )
        rows.append(
            {
                "model_name": name,
                **{
                    k.removeprefix("test_"): v
                    for k, v in result["metrics"].items()
                    if k.startswith("test_")
                },
            }
        )
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def should_retrain(
    latest_model_metadata: dict[str, Any] | None,
    now: datetime | None = None,
    interval_days: int | None = None,
) -> tuple[bool, str]:
    """Decide whether the saved model is absent or older than its schedule."""
    if not latest_model_metadata:
        return True, "no existing model"
    created = datetime.fromisoformat(latest_model_metadata["created_at"])
    current = now or datetime.now(timezone.utc)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    due = current - created >= timedelta(
        days=interval_days or settings.RETRAIN_INTERVAL_DAYS
    )
    return due, (
        "retraining interval elapsed" if due else "model is within retraining interval"
    )


def drift_check(
    recent_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    tolerance: float = 0.25,
) -> tuple[bool, str]:
    """Flag materially degraded recent IC or error as a retraining hook."""
    ic_bad = (
        np.isfinite(baseline_metrics.get("ic", np.nan))
        and recent_metrics.get("ic", -np.inf) < baseline_metrics["ic"] - tolerance
    )
    base_error = baseline_metrics.get("rmse", np.nan)
    err_bad = (
        np.isfinite(base_error)
        and base_error > 0
        and recent_metrics.get("rmse", np.inf) > base_error * (1 + tolerance)
    )
    return bool(ic_bad or err_bad), (
        "recent IC/error degraded" if ic_bad or err_bad else "no material metric drift"
    )


def retrain_if_due(
    data_path: str | Path, model_name: str, target_column: str | None = None
) -> dict[str, Any] | None:
    """Train and register a model when its persisted training schedule is due."""
    from src.models.registry import get_latest_model

    try:
        latest = get_latest_model(model_name)
    except FileNotFoundError:
        latest = None
    due, reason = should_retrain(latest)
    logger.info("Retraining decision for %s: %s (%s)", model_name, due, reason)
    return train_model(data_path, model_name, target_column) if due else None
