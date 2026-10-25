"""Cross-sectional long/short ranking signals."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def rank_signals(predictions_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Long the strongest and short weakest forecasts for each date.

    If fewer than ``2 * top_n`` finite forecasts exist, long and short groups
    are clipped to disjoint sets (top half long, bottom half short); a middle
    symbol is held when the count is odd. Non-finite forecasts are held and
    receive a missing rank.
    """
    required = {"symbol", "date", "predicted_return"}
    missing = required - set(predictions_df.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    result = predictions_df.copy().reset_index(drop=True)
    values = pd.to_numeric(result["predicted_return"], errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
    result["rank"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["signal"] = 0
    # A stable symbol tie-break ensures ranking is invariant to input row order.
    for _, indexes in result.groupby("date").groups.items():
        valid_indexes = [index for index in indexes if finite[index]]
        ordered = (
            result.loc[valid_indexes]
            .assign(_numeric_return=values.iloc[valid_indexes])
            .sort_values(
                ["_numeric_return", "symbol"], ascending=[False, True], kind="stable"
            )
        )
        count = len(ordered)
        result.loc[ordered.index, "rank"] = range(1, count + 1)
        side = min(top_n, count // 2) if count < 2 * top_n else top_n
        if side:
            result.loc[ordered.index[:side], "signal"] = 1
            result.loc[ordered.index[-side:], "signal"] = -1
    result["signal"] = result["signal"].astype("int64")
    logger.info("Created rank signals for %d dates", result["date"].nunique())
    return result
