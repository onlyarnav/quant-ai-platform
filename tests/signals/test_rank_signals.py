"""Cross-sectional ranking behavior, tie handling, and invalid forecasts."""

import numpy as np
import pandas as pd
import pytest

from src.signals.rank_signals import rank_signals


def test_rank_order_top_n_and_ties_are_symbol_deterministic():
    """Highest forecasts rank first; tied returns break ties by symbol."""
    frame = pd.DataFrame(
        {
            "symbol": ["D", "B", "C", "A"],
            "date": pd.Timestamp("2025-01-01"),
            "predicted_return": [0.1, 0.2, 0.2, -0.1],
        }
    )
    result = rank_signals(frame, 1).set_index("symbol")
    assert result.loc["B", "rank"] == 1
    assert result.loc["C", "rank"] == 2
    assert result.loc["B", "signal"] == 1
    assert result.loc["A", "signal"] == -1
    pd.testing.assert_frame_equal(
        rank_signals(frame, 1).sort_values("symbol").reset_index(drop=True),
        rank_signals(frame.sample(frac=1, random_state=3), 1)
        .sort_values("symbol")
        .reset_index(drop=True),
    )


def test_rank_clips_short_and_long_sides_for_small_universe():
    """With fewer than 2N assets, sides remain disjoint and an odd center holds."""
    frame = pd.DataFrame(
        {
            "symbol": list("ABCDE"),
            "date": pd.Timestamp("2025-01-01"),
            "predicted_return": [0.5, 0.4, 0.0, -0.2, -0.3],
        }
    )
    result = rank_signals(frame, 4)
    assert result["signal"].value_counts().to_dict() == {1: 2, -1: 2, 0: 1}
    assert result.loc[result["symbol"] == "C", "signal"].item() == 0


def test_rank_invalid_forecasts_hold_and_do_not_receive_a_rank():
    """NaN and infinite values must not be selected as extreme short or long."""
    frame = pd.DataFrame(
        {
            "symbol": list("ABCDE"),
            "date": pd.Timestamp("2025-01-01"),
            "predicted_return": [0.5, 0.2, 0.0, np.nan, -np.inf],
        }
    )
    result = rank_signals(frame, 1).set_index("symbol")
    assert result.loc["D", "signal"] == 0
    assert result.loc["E", "signal"] == 0
    assert pd.isna(result.loc["D", "rank"])
    assert pd.isna(result.loc["E", "rank"])


def test_rank_empty_and_invalid_inputs():
    """Empty panels are valid; missing identifiers and bad top-N are rejected."""
    empty = pd.DataFrame(columns=["symbol", "date", "predicted_return"])
    result = rank_signals(empty, 2)
    assert result.empty and {"rank", "signal"} <= set(result.columns)
    with pytest.raises(ValueError, match="missing columns"):
        rank_signals(pd.DataFrame({"date": [1], "predicted_return": [0.1]}), 1)
    for bad_n in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            rank_signals(empty, bad_n)
