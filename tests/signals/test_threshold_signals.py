"""Threshold mapping edge cases based on the configured signal contract."""

import numpy as np
import pandas as pd
import pytest

from src.signals.threshold_signals import threshold_signals


def test_threshold_edges_buy_sell_hold_and_confidence():
    """Only forecasts strictly beyond threshold cross into long or short."""
    frame = pd.DataFrame({"predicted_return": [0.2, 0.1, 0.0, -0.1, -0.2]})
    result = threshold_signals(frame, 0.1)
    assert result["signal"].tolist() == [1, 0, 0, 0, -1]
    assert result["signal"].dtype.kind in "iu"
    assert result["confidence"].tolist() == [1.0, 1.0, 0.0, 1.0, 1.0]
    assert frame.columns.tolist() == ["predicted_return"]


@pytest.mark.parametrize("threshold", [0.0, -0.1, np.nan, np.inf, -np.inf])
def test_threshold_rejects_nonpositive_or_nonfinite_thresholds(threshold):
    """Threshold configuration must be finite and strictly positive."""
    with pytest.raises(ValueError, match="finite and positive"):
        threshold_signals(pd.DataFrame({"predicted_return": [0.1]}), threshold)


def test_threshold_handles_nan_infinity_empty_and_missing_column():
    """Non-finite forecasts safely hold; empty inputs retain output columns."""
    result = threshold_signals(
        pd.DataFrame({"predicted_return": [np.nan, np.inf, -np.inf]}), 0.05
    )
    assert result["signal"].tolist() == [0, 0, 0]
    empty = threshold_signals(pd.DataFrame({"predicted_return": []}), 0.05)
    assert empty.empty and {"signal", "confidence"} <= set(empty.columns)
    with pytest.raises(ValueError, match="predicted_return"):
        threshold_signals(pd.DataFrame({"close": [1.0]}), 0.05)
