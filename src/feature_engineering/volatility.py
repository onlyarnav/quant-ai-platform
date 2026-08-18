import logging

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_VOLATILITY_WINDOWS: list[int] = [10, 20, 50]
DEFAULT_ATR_WINDOW: int = 14
DEFAULT_BOLLINGER_WINDOW: int = 20
DEFAULT_BOLLINGER_NUM_STD: float = 2.0


def _validate_returns_column(df: pd.DataFrame) -> None:
    if "returns" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'returns' column.")


def _validate_close_column(df: pd.DataFrame) -> None:
    if "close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'close' column.")


def _validate_hlc_columns(df: pd.DataFrame) -> None:
    required = {"high", "low", "close"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Input DataFrame must contain columns: {sorted(missing)}"
        )


def add_volatility(
    df: pd.DataFrame,
    windows: list[int] = DEFAULT_VOLATILITY_WINDOWS,
) -> pd.DataFrame:
    _validate_returns_column(df)

    logger.info(
        "Computing rolling volatility features for windows: %s",
        windows,
    )

    result = df.copy()

    for window in windows:
        column_name = f"volatility_{window}"
        result[column_name] = (
            result["returns"]
            .rolling(window=window)
            .std()
        )

    logger.info(
        "Finished computing volatility features. Added columns: %s",
        [f"volatility_{w}" for w in windows],
    )

    return result


def add_atr(
    df: pd.DataFrame,
    window: int = DEFAULT_ATR_WINDOW,
) -> pd.DataFrame:
    _validate_hlc_columns(df)

    logger.info("Computing ATR feature for window: %d", window)

    result = df.copy()

    prev_close = result["close"].shift(1)

    high_low = result["high"] - result["low"]
    high_prev_close = (result["high"] - prev_close).abs()
    low_prev_close = (result["low"] - prev_close).abs()

    true_range = pd.concat(
        [
            high_low,
            high_prev_close,
            low_prev_close,
        ],
        axis=1,
    ).max(axis=1)

    column_name = f"atr_{window}"

    result[column_name] = (
        true_range
        .rolling(window=window)
        .mean()
    )

    logger.info(
        "Finished computing ATR feature. Added column: %s",
        column_name,
    )

    return result


def add_bollinger_width(
    df: pd.DataFrame,
    window: int = DEFAULT_BOLLINGER_WINDOW,
    num_std: float = DEFAULT_BOLLINGER_NUM_STD,
) -> pd.DataFrame:
    _validate_close_column(df)

    logger.info(
        "Computing Bollinger Band Width feature with window=%d, num_std=%.2f",
        window,
        num_std,
    )

    result = df.copy()

    middle_band = (
        result["close"]
        .rolling(window=window)
        .mean()
    )

    std_dev = (
        result["close"]
        .rolling(window=window)
        .std()
    )

    upper_band = middle_band + (num_std * std_dev)
    lower_band = middle_band - (num_std * std_dev)

    result["bollinger_width"] = (
        (upper_band - lower_band) / middle_band
    )

    logger.info(
        "Finished computing Bollinger Band Width feature."
    )

    return result


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(
        "Adding all volatility features."
    )

    result = add_volatility(
        df,
        DEFAULT_VOLATILITY_WINDOWS,
    )

    result = add_atr(
        result,
        DEFAULT_ATR_WINDOW,
    )

    result = add_bollinger_width(
        result,
        DEFAULT_BOLLINGER_WINDOW,
        DEFAULT_BOLLINGER_NUM_STD,
    )

    logger.info(
        "All volatility features added successfully."
    )

    return result
