import logging

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_RSI_WINDOW: int = 14
DEFAULT_MACD_FAST: int = 12
DEFAULT_MACD_SLOW: int = 26
DEFAULT_MACD_SIGNAL: int = 9
DEFAULT_ROC_WINDOWS: list[int] = [5, 10]
DEFAULT_MOMENTUM_WINDOWS: list[int] = [5, 10]


def _validate_close_column(df: pd.DataFrame) -> None:

    if "close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'close' column.")


def add_rsi(df: pd.DataFrame, window: int = DEFAULT_RSI_WINDOW) -> pd.DataFrame:

    _validate_close_column(df)
    logger.info("Computing RSI feature for window: %d", window)

    result = df.copy()
    delta = result["close"].diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    column_name = f"rsi_{window}"
    result[column_name] = rsi

    logger.info("Finished computing RSI feature. Added column: %s", column_name)
    return result


def add_macd(
    df: pd.DataFrame,
    fast: int = DEFAULT_MACD_FAST,
    slow: int = DEFAULT_MACD_SLOW,
    signal: int = DEFAULT_MACD_SIGNAL,
) -> pd.DataFrame:

    _validate_close_column(df)
    logger.info(
        "Computing MACD features with fast=%d, slow=%d, signal=%d",
        fast, slow, signal,
    )

    result = df.copy()
    ema_fast = result["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = result["close"].ewm(span=slow, adjust=False).mean()

    result["macd"] = ema_fast - ema_slow
    result["macd_signal"] = result["macd"].ewm(span=signal, adjust=False).mean()

    logger.info("Finished computing MACD features. Added columns: ['macd', 'macd_signal']")
    return result


def add_roc(df: pd.DataFrame, windows: list[int] = DEFAULT_ROC_WINDOWS) -> pd.DataFrame:

    _validate_close_column(df)
    logger.info("Computing ROC features for windows: %s", windows)

    result = df.copy()
    for window in windows:
        column_name = f"roc_{window}"
        shifted = result["close"].shift(window)
        result[column_name] = (result["close"] - shifted) / shifted * 100

    logger.info("Finished computing ROC features. Added columns: %s",
                [f"roc_{w}" for w in windows])
    return result


def add_momentum(df: pd.DataFrame, windows: list[int] = DEFAULT_MOMENTUM_WINDOWS) -> pd.DataFrame:

    _validate_close_column(df)
    logger.info("Computing Momentum features for windows: %s", windows)

    result = df.copy()
    for window in windows:
        column_name = f"momentum_{window}"
        result[column_name] = result["close"] - result["close"].shift(window)

    logger.info("Finished computing Momentum features. Added columns: %s",
                [f"momentum_{w}" for w in windows])
    return result


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Adding all momentum features (RSI, MACD, ROC, Momentum).")
    result = add_rsi(df, DEFAULT_RSI_WINDOW)
    result = add_macd(result, DEFAULT_MACD_FAST, DEFAULT_MACD_SLOW, DEFAULT_MACD_SIGNAL)
    result = add_roc(result, DEFAULT_ROC_WINDOWS)
    result = add_momentum(result, DEFAULT_MOMENTUM_WINDOWS)
    logger.info("All momentum features added successfully.")
    return result
