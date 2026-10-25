"""Signal mapping tests over deterministic synthetic predictions."""

import pandas as pd
import pytest
from src.signals.pipeline import generate_signals, run_signal_pipeline
from src.signals.threshold_signals import threshold_signals
from src.signals.rank_signals import rank_signals


def predictions():
    """Return a balanced two-date forecast panel."""
    return pd.DataFrame(
        {
            "symbol": list("ABCDE") * 2,
            "date": list(pd.date_range("2024-01-01", periods=2).repeat(5)),
            "predicted_return": [0.2, 0.1, 0, -0.1, -0.2] * 2,
        }
    )


def test_threshold_boundaries_and_confidence():
    """Threshold boundaries hold and signals stay within contract values."""
    result = threshold_signals(predictions(), 0.1)
    assert result.signal.tolist()[:5] == [1, 0, 0, 0, -1]
    assert set(result.signal) <= {-1, 0, 1}
    with pytest.raises(ValueError):
        threshold_signals(predictions(), 0)


def test_rank_determinism_and_schema():
    """Rank output is invariant to row order and handles undersized universes."""
    frame = predictions()
    a = generate_signals(frame, "rank", top_n=2)
    b = generate_signals(frame.sample(frac=1, random_state=2), "rank", top_n=2)
    pd.testing.assert_frame_equal(a, b)
    assert a.columns.tolist() == [
        "symbol",
        "date",
        "predicted_return",
        "signal",
        "confidence",
        "rank",
    ]
    assert set(a.signal) == {-1, 0, 1}
    assert len(rank_signals(frame.iloc[:3], 2)) == 3


def test_signal_parquet_schema(tmp_path):
    """File runner writes the stable parquet contract in sorted order."""
    source = tmp_path / "predictions"
    source.mkdir()
    predictions().to_parquet(source / "equity_TEST_daily.parquet", index=False)
    output = run_signal_pipeline(
        source, tmp_path / "signals", "threshold", threshold=0.1
    )
    assert len(output) == 1
    stored = pd.read_parquet(output[0])
    assert stored.columns.tolist() == [
        "symbol",
        "date",
        "predicted_return",
        "signal",
        "confidence",
        "rank",
    ]
    assert stored[["date", "symbol"]].equals(
        stored[["date", "symbol"]]
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
