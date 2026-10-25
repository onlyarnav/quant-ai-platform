"""Point-in-time prediction generation from feature parquet data."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import settings
from src.models.dataset import IDENTIFIER_COLUMNS, TARGET_PREFIX
from src.models.registry import get_latest_model

logger = logging.getLogger(__name__)
PREDICTION_COLUMNS: tuple[str, ...] = (
    "symbol",
    "date",
    "predicted_return",
    "model_name",
    "model_version",
)


def predict(
    data_path: str | Path,
    model_name: str = "ridge",
    start_date: str | None = None,
    end_date: str | None = None,
    output_dir: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    latest_per_symbol: bool = False,
) -> pd.DataFrame:
    """Generate and persist predictions from a feature parquet file.

    Rows with missing features are omitted. Since only the requested row's
    point-in-time feature vector is used, appending future rows cannot alter
    an earlier prediction.
    """
    path = Path(data_path)
    frame = pd.read_parquet(path)
    model_info = get_latest_model(model_name, artifact_dir)
    feature_columns = model_info["feature_columns"]
    actual = [
        c
        for c in frame.columns
        if c not in IDENTIFIER_COLUMNS
        and not c.startswith(TARGET_PREFIX)
        and pd.api.types.is_numeric_dtype(frame[c])
    ]
    if model_info.get("feature_policy") == "scale_free":
        allowed = set(settings.SCALE_FREE_FEATURE_COLUMNS)
        actual = [column for column in actual if column in allowed]
    if actual != feature_columns:
        raise ValueError(
            f"Feature columns/order mismatch. Expected {feature_columns}, received {actual}"
        )
    missing = {"symbol", "date"} - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction data missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if start_date:
        frame = frame.loc[frame["date"] >= pd.Timestamp(start_date)]
    if end_date:
        frame = frame.loc[frame["date"] <= pd.Timestamp(end_date)]
    if latest_per_symbol and not frame.empty:
        latest_dates = frame.groupby("symbol")["date"].transform("max")
        frame = frame.loc[frame["date"].eq(latest_dates)]
    finite = np.isfinite(frame[feature_columns].to_numpy(dtype=float)).all(axis=1)
    dropped = int((~finite).sum())
    if dropped:
        logger.warning(
            "Skipping %d rows with missing or infinite model features", dropped
        )
    usable = frame.loc[finite].copy()
    predictions = (
        model_info["model"].predict(usable[feature_columns])
        if len(usable)
        else np.array([], dtype=float)
    )
    output = pd.DataFrame(
        {
            "symbol": usable["symbol"].astype(str).to_numpy(),
            "date": usable["date"].to_numpy(),
            "predicted_return": predictions,
            "model_name": model_name,
            "model_version": model_info["version"],
        },
        columns=PREDICTION_COLUMNS,
    )
    output = output.sort_values(["date", "symbol"]).reset_index(drop=True)
    destination = Path(output_dir or settings.PREDICTIONS_DIR)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / path.name
    output.to_parquet(output_path, index=False)
    logger.info("Wrote %d predictions to %s", len(output), output_path)
    return output


def predict_latest(
    model_name: str = "ridge",
    features_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> list[pd.DataFrame]:
    """Predict each feature parquet file using its latest row per symbol."""
    root = Path(features_dir or settings.FEATURES_DIR)
    results = []
    for path in sorted(root.glob("*.parquet")):
        frame = pd.read_parquet(path)
        if "date" not in frame:
            raise ValueError(f"Feature file {path} lacks date")
        results.append(
            predict(path, model_name, output_dir=output_dir, latest_per_symbol=True)
        )
    return results
