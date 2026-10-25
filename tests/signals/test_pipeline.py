"""Signal pipeline schema, deduplication, sorting, and point-in-time checks."""

import numpy as np
import pandas as pd
import pytest

from src.signals.pipeline import generate_signals, run_signal_pipeline


def prediction_panel() -> pd.DataFrame:
    """Return two daily cross-sections with explicit model versions."""
    return pd.DataFrame(
        {
            "symbol": ["C", "A", "B", "C", "A", "B"],
            "date": list(pd.date_range("2025-01-01", periods=2).repeat(3)),
            "predicted_return": [-0.03, 0.02, 0.01, -0.04, 0.05, 0.02],
            "model_name": "ridge",
            "model_version": "v1",
        }
    )


def test_pipeline_schema_types_sorting_and_shuffle_determinism():
    """Pipeline preserves identifiers and emits sorted typed contract columns."""
    frame = prediction_panel()
    result = generate_signals(frame, "threshold", threshold=0.01)
    shuffled = generate_signals(
        frame.sample(frac=1, random_state=8), "threshold", threshold=0.01
    )
    pd.testing.assert_frame_equal(result, shuffled)
    assert result.columns.tolist() == [
        "symbol",
        "date",
        "predicted_return",
        "signal",
        "confidence",
        "rank",
    ]
    assert result["signal"].dtype.kind in "iu"
    assert result["confidence"].dtype.kind == "f"
    assert result[["date", "symbol"]].equals(
        result[["date", "symbol"]]
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
    assert result["date"].tolist() == sorted(frame["date"].tolist())


def test_rank_signals_are_point_in_time_and_invalid_forecasts_hold():
    """Appending a later date cannot alter earlier ranks, confidence, or signals."""
    frame = prediction_panel()
    first = generate_signals(frame, "rank", top_n=1)
    later = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "date": pd.Timestamp("2025-01-10"),
            "predicted_return": [100.0, -100.0],
        }
    )
    extended = generate_signals(
        pd.concat([frame, later], ignore_index=True), "rank", top_n=1
    )
    pd.testing.assert_frame_equal(
        first,
        extended.loc[extended["date"] <= pd.Timestamp("2025-01-02")].reset_index(
            drop=True
        ),
    )
    invalid = frame.iloc[:2].copy()
    invalid["predicted_return"] = [np.nan, np.inf]
    result = generate_signals(invalid, "rank", top_n=1)
    assert result["signal"].tolist() == [0, 0]


def test_duplicate_predictions_are_order_independent_and_averaged():
    """Conflicting duplicate rows collapse deterministically to their mean."""
    frame = pd.DataFrame(
        {
            "symbol": ["A", "A", "B"],
            "date": pd.Timestamp("2025-01-01"),
            "predicted_return": [0.2, -0.2, 0.0],
        }
    )
    first = generate_signals(frame, "threshold", threshold=0.1)
    shuffled = generate_signals(
        frame.sample(frac=1, random_state=12), "threshold", threshold=0.1
    )
    pd.testing.assert_frame_equal(first, shuffled)
    assert first["symbol"].tolist() == ["A", "B"]
    assert first.loc[first["symbol"] == "A", "predicted_return"].item() == 0.0
    assert first.loc[first["symbol"] == "A", "signal"].item() == 0


def test_pipeline_empty_invalid_method_and_parquet_output(tmp_path):
    """Empty files round-trip; malformed columns and methods fail explicitly."""
    with pytest.raises(ValueError, match="method"):
        generate_signals(pd.DataFrame(), "unsupported")
    with pytest.raises(ValueError, match="missing columns"):
        generate_signals(pd.DataFrame({"predicted_return": [0.1]}))
    with pytest.raises(ValueError, match="positive integer"):
        generate_signals(prediction_panel(), "rank", top_n=1.5)
    empty = generate_signals(
        pd.DataFrame(columns=["symbol", "date", "predicted_return"]),
        "threshold",
        threshold=0.01,
    )
    assert empty.empty and empty.columns.tolist() == [
        "symbol",
        "date",
        "predicted_return",
        "signal",
        "confidence",
        "rank",
    ]
    source = tmp_path / "predictions"
    source.mkdir()
    prediction_panel().to_parquet(source / "equity_TEST_daily.parquet", index=False)
    outputs = run_signal_pipeline(source, tmp_path / "signals", "rank", top_n=1)
    output = pd.read_parquet(outputs[0])
    assert len(output) == 6
    pd.testing.assert_frame_equal(
        output,
        output.sort_values(["date", "symbol"]).reset_index(drop=True),
    )
