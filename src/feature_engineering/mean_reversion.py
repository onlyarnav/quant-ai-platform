import logging

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_ZSCORE_WINDOW: int = 20
DEFAULT_SMA_DISTANCE_WINDOW: int = 20
DEFAULT_EMA_DISTANCE_WINDOW: int = 20


def _validate_close_column(df: pd.DataFrame) -> None:

    if "close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'close' column.")


def add_zscore(df: pd.DataFrame, window: int = DEFAULT_ZSCORE_WINDOW) -> pd.DataFrame:

    _validate_close_column(df)
    logger.info("Computing Z-Score feature for window: %d", window)

    result = df.copy()
    rolling_mean = result["close"].rolling(window=window).mean()
    rolling_std = result["close"].rolling(window=window).std()

    column_name = f"zscore_{window}"
    result[column_name] = (result["close"] - rolling_mean) / rolling_std

    logger.info("Finished computing Z-Score feature. Added column: %s", column_name)
    return result


def add_distance_from_sma(
    df: pd.DataFrame, window: int = DEFAULT_SMA_DISTANCE_WINDOW
) -> pd.DataFrame:

    _validate_close_column(df)
    logger.info("Computing Distance from SMA feature for window: %d", window)

    result = df.copy()
    sma = result["close"].rolling(window=window).mean()
    result["distance_from_sma"] = (result["close"] - sma) / sma

    logger.info("Finished computing Distance from SMA feature. Added column: 'distance_from_sma'")
    return result


def add_distance_from_ema(
    df: pd.DataFrame, window: int = DEFAULT_EMA_DISTANCE_WINDOW
) -> pd.DataFrame:

    _validate_close_column(df)
    logger.info("Computing Distance from EMA feature for window: %d", window)

    result = df.copy()
    ema = result["close"].ewm(span=window, adjust=False).mean()
    result["distance_from_ema"] = (result["close"] - ema) / ema

    logger.info("Finished computing Distance from EMA feature. Added column: 'distance_from_ema'")
    return result


def add_mean_reversion_features(df: pd.DataFrame) -> pd.DataFrame:
 
    logger.info("Adding all mean-reversion features (Z-Score, Distance from SMA/EMA).")
    result = add_zscore(df, DEFAULT_ZSCORE_WINDOW)
    result = add_distance_from_sma(result, DEFAULT_SMA_DISTANCE_WINDOW)
    result = add_distance_from_ema(result, DEFAULT_EMA_DISTANCE_WINDOW)
    logger.info("All mean-reversion features added successfully.")
    return result
