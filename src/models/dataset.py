import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_PREFIX: str = "future_return_"
DEFAULT_TARGET_COLUMN: str = "future_return_1d"
IDENTIFIER_COLUMNS: list[str] = [
    "symbol",
    "date",
    "asset_class",
    "exchange",
    "currency",
    "source",
]


def load_feature_data(path: str | Path) -> pd.DataFrame:

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Feature file not found: {file_path}")

    logger.info("Loading feature data from %s", file_path)
    try:
        df = pd.read_parquet(file_path)
    except Exception as exc:
        raise ValueError(f"Failed to read Parquet file {file_path}: {exc}") from exc

    logger.info("Loaded feature data with shape %s", df.shape)
    return df


def _get_feature_columns(df: pd.DataFrame) -> list[str]:

    candidates = [
        col
        for col in df.columns
        if col not in IDENTIFIER_COLUMNS and not col.startswith(TARGET_PREFIX)
    ]
    numeric = df[candidates].select_dtypes(include=[np.number]).columns.tolist()

    skipped = sorted(set(candidates) - set(numeric))
    if skipped:
        logger.warning("Skipping non-numeric columns: %s", skipped)
    return numeric


def clean_feature_data(
    df: pd.DataFrame, target_column: str = DEFAULT_TARGET_COLUMN
) -> pd.DataFrame:

    if "date" not in df.columns:
        raise ValueError("Feature DataFrame must contain a 'date' column.")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    logger.info("Cleaning feature data. Input rows: %d", len(df))

    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])

    feature_columns = _get_feature_columns(result)
    check_columns = feature_columns + [target_column]

    result[check_columns] = result[check_columns].replace([np.inf, -np.inf], np.nan)
    result = result.dropna(subset=check_columns)

    sort_keys = ["date", "symbol"] if "symbol" in result.columns else ["date"]
    result = result.sort_values(sort_keys).reset_index(drop=True)

    logger.info(
        "Cleaning complete. Output rows: %d (dropped %d)",
        len(result), len(df) - len(result),
    )
    return result


def split_features_and_target(
    df: pd.DataFrame, target_column: str = DEFAULT_TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:

    if "date" not in df.columns:
        raise ValueError("Feature DataFrame must contain a 'date' column.")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    feature_columns = _get_feature_columns(df)
    if not feature_columns:
        raise ValueError("No feature columns found after exclusions.")

    X = df[feature_columns].copy()
    y = df[target_column].copy()
    dates = df["date"].copy()

    logger.info(
        "Split complete. X shape: %s, target: %s, features: %d",
        X.shape, target_column, len(feature_columns),
    )
    return X, y, dates


def build_ml_dataset(
    path: str | Path, target_column: str = DEFAULT_TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:

    df = load_feature_data(path)
    df = clean_feature_data(df, target_column)
    return split_features_and_target(df, target_column)