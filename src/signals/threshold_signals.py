"""Threshold-based return forecast conversion to trading signals."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def threshold_signals(predictions_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Assign long, short, or hold based on an absolute return threshold.

    Confidence is absolute forecast divided by threshold, clipped at one.
    """
    if threshold <= 0 or not np.isfinite(threshold):
        raise ValueError("threshold must be finite and positive")
    if "predicted_return" not in predictions_df:
        raise ValueError("predictions must contain 'predicted_return'")
    result = predictions_df.copy()
    values = pd.to_numeric(result["predicted_return"], errors="coerce")
    numeric = values.to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(numeric)
    signal = np.zeros(len(result), dtype="int64")
    signal[finite & (numeric > threshold)] = 1
    signal[finite & (numeric < -threshold)] = -1
    result["signal"] = signal
    result["confidence"] = (values.abs() / threshold).where(finite, 0.0).clip(upper=1.0)
    logger.info("Created threshold signals at threshold=%g", threshold)
    return result
