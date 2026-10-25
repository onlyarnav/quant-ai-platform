import logging
from pathlib import Path

import numpy as np
import pandas as pd
from config.settings import settings

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
    """Load a feature parquet file, raising a clear error for invalid files."""
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


def _get_feature_columns(
    df: pd.DataFrame, feature_policy: str | None = None
) -> list[str]:
    """Select numeric model columns while excluding identifiers and labels.

    The default scale-free policy limits linear-model inputs and cross-symbol
    learning to ratios and oscillators. ``all_numeric`` is available for
    explicit experiments; all scaling remains inside the training pipeline.
    """
    candidates = [
        col
        for col in df.columns
        if col not in IDENTIFIER_COLUMNS and not col.startswith(TARGET_PREFIX)
    ]
    numeric = df[candidates].select_dtypes(include=[np.number]).columns.tolist()
    policy = feature_policy or settings.ML_FEATURE_POLICY
    if policy == "scale_free":
        numeric = [
            column
            for column in settings.SCALE_FREE_FEATURE_COLUMNS
            if column in numeric
        ]
    elif policy != "all_numeric":
        raise ValueError("feature_policy must be 'scale_free' or 'all_numeric'.")
    logger.info("Selected %d %s model features: %s", len(numeric), policy, numeric)

    skipped = sorted(set(candidates) - set(numeric))
    if skipped:
        logger.warning("Skipping non-numeric columns: %s", skipped)
    return numeric


def clean_feature_data(
    df: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
    feature_policy: str | None = None,
) -> pd.DataFrame:
    """Return cleaned feature rows without mutating the supplied frame."""
    if "date" not in df.columns:
        raise ValueError("Feature DataFrame must contain a 'date' column.")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    logger.info("Cleaning feature data. Input rows: %d", len(df))

    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])

    feature_columns = _get_feature_columns(result, feature_policy)
    if not feature_columns:
        raise ValueError("No model features found for the selected feature policy.")
    check_columns = feature_columns + [target_column]

    result[check_columns] = result[check_columns].replace([np.inf, -np.inf], np.nan)
    result = result.dropna(subset=check_columns)

    sort_keys = ["date", "symbol"] if "symbol" in result.columns else ["date"]
    result = result.sort_values(sort_keys).reset_index(drop=True)

    logger.info(
        "Cleaning complete. Output rows: %d (dropped %d)",
        len(result),
        len(df) - len(result),
    )
    return result


def split_features_and_target(
    df: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
    feature_policy: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate feature matrix, target series, and dates without label leakage."""
    if "date" not in df.columns:
        raise ValueError("Feature DataFrame must contain a 'date' column.")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    feature_columns = _get_feature_columns(df, feature_policy)
    if not feature_columns:
        raise ValueError("No feature columns found after exclusions.")

    X = df[feature_columns].copy()
    y = df[target_column].copy()
    dates = df["date"].copy()

    logger.info(
        "Split complete. X shape: %s, target: %s, features: %d",
        X.shape,
        target_column,
        len(feature_columns),
    )
    return X, y, dates


def build_ml_dataset(
    path: str | Path,
    target_column: str = DEFAULT_TARGET_COLUMN,
    feature_policy: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load, clean, and split a feature parquet into model inputs and labels."""
    df = load_feature_data(path)
    df = clean_feature_data(df, target_column, feature_policy)
    return split_features_and_target(df, target_column, feature_policy)
