import logging

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SMA_WINDOWS: list[int] = [5, 10, 20, 50, 200]
DEFAULT_EMA_WINDOWS: list[int] = [10, 20, 50]


def add_sma(df: pd.DataFrame, windows: list[int] = DEFAULT_SMA_WINDOWS) -> pd.DataFrame:
    
    if "close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'close' column.")

    logger.info("Computing SMA features for windows: %s", windows)

    result = df.copy()
    for window in windows:
        column_name = f"sma_{window}"
        result[column_name] = result["close"].rolling(window=window).mean()

    logger.info("Finished computing SMA features. Added columns: %s",
                [f"sma_{w}" for w in windows])
    return result


def add_ema(df: pd.DataFrame, windows: list[int] = DEFAULT_EMA_WINDOWS) -> pd.DataFrame:
   
    if "close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'close' column.")

    logger.info("Computing EMA features for windows: %s", windows)

    result = df.copy()
    for window in windows:
        column_name = f"ema_{window}"
        result[column_name] = result["close"].ewm(span=window, adjust=False).mean()

    logger.info("Finished computing EMA features. Added columns: %s",
                [f"ema_{w}" for w in windows])
    return result


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all default trend-based features (SMA + EMA) to the DataFrame.

    This is the main entry point used by the feature engineering
    pipeline to attach the full set of trend indicators.

    Args:
        df: Input DataFrame. Must contain a `close` column, sorted
            chronologically in ascending order by date.

    Returns:
        A new DataFrame with all default SMA and EMA columns added.
    """
    logger.info("Adding all trend features (SMA + EMA).")
    result = add_sma(df, DEFAULT_SMA_WINDOWS)
    result = add_ema(result, DEFAULT_EMA_WINDOWS)
    logger.info("All trend features added successfully.")
    return result
