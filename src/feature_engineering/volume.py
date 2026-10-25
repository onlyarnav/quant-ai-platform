"""Volume-derived feature calculations."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def add_volume_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add rolling volume average, relative volume, and OBV without mutating input.

    Args:
        df: Chronologically sorted frame with close and volume columns.
        window: Positive trailing window for the volume average.

    Returns:
        A copy containing ``volume_ma_20``, ``volume_ratio`` and ``obv``.
    """
    missing = {"close", "volume"} - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame must contain columns: {sorted(missing)}")
    if window < 1:
        raise ValueError("window must be positive")
    result = df.copy()
    result["volume_ma_20"] = result["volume"].rolling(window).mean()
    result["volume_ratio"] = result["volume"] / result["volume_ma_20"]
    direction = result["close"].diff().gt(0).astype(float) - result["close"].diff().lt(
        0
    ).astype(float)
    result["obv"] = (direction * result["volume"]).fillna(0).cumsum()
    logger.info("Added rolling volume features (window=%d)", window)
    return result
