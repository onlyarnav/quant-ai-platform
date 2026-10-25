"""Forward return targets kept separate from model input features."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)
DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 10)


def add_future_return_targets(
    df: pd.DataFrame, horizons: tuple[int, ...] = DEFAULT_HORIZONS
) -> pd.DataFrame:
    """Add forward returns within each symbol; these columns are labels only.

    Args:
        df: Frame containing close and, optionally, symbol.
        horizons: Positive forward observation counts.

    Returns:
        Copy with ``future_return_<n>d`` labels. Grouping prevents labels
        from crossing symbol boundaries in a multi-asset panel.
    """
    if "close" not in df:
        raise ValueError("Input DataFrame must contain a 'close' column.")
    if not horizons or any(not isinstance(h, int) or h < 1 for h in horizons):
        raise ValueError("horizons must contain positive integers")
    result = df.copy()
    grouped = (
        result.groupby("symbol", sort=False)["close"]
        if "symbol" in result
        else result["close"]
    )
    for horizon in horizons:
        future = grouped.shift(-horizon)
        result[f"future_return_{horizon}d"] = future / result["close"] - 1
    logger.info("Added future return targets for horizons %s", horizons)
    return result


# Stable public alias used by feature-building callers.
add_targets = add_future_return_targets
