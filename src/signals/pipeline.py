"""File-level signal generation from prediction parquet datasets."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import settings
from src.signals.rank_signals import rank_signals
from src.signals.threshold_signals import threshold_signals

logger = logging.getLogger(__name__)
SIGNAL_COLUMNS: tuple[str, ...] = (
    "symbol",
    "date",
    "predicted_return",
    "signal",
    "confidence",
    "rank",
)


def generate_signals(
    predictions_df: pd.DataFrame, method: str = "threshold", **params: Any
) -> pd.DataFrame:
    """Transform forecasts while preserving the fixed signal output schema.

    Duplicate symbol/date forecasts are combined by taking the mean of their
    finite values. This makes deduplication independent of input row order;
    groups with no finite forecasts remain missing and produce HOLD signals.
    """
    if method not in {"threshold", "rank", "topn"}:
        raise ValueError("method must be 'threshold', 'rank', or 'topn'")
    required = {"symbol", "date", "predicted_return"}
    missing = required - set(predictions_df.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    normalized = predictions_df[["symbol", "date", "predicted_return"]].copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    numeric = pd.to_numeric(normalized["predicted_return"], errors="coerce")
    normalized["predicted_return"] = numeric.where(
        np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
    )
    # Averaging duplicates prevents whichever row happened to arrive last from winning.
    predictions = normalized.groupby(["symbol", "date"], as_index=False, dropna=False)[
        "predicted_return"
    ].mean()
    if method == "threshold":
        result = threshold_signals(
            predictions, float(params.get("threshold", settings.SIGNAL_THRESHOLD))
        )
    elif method in {"rank", "topn"}:
        result = rank_signals(predictions, params.get("top_n", settings.TOP_N))
        if "confidence" not in result:
            # Normalize within each date so later forecasts cannot affect earlier confidence.
            numeric = pd.to_numeric(result["predicted_return"], errors="coerce")
            finite = numeric.where(
                np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
            )
            absolute = finite.abs()
            scale = absolute.groupby(result["date"]).transform("max")
            result["confidence"] = absolute.div(scale.where(scale.ne(0))).fillna(0)
    if "confidence" not in result:
        result["confidence"] = float("nan")
    if "rank" not in result:
        result["rank"] = pd.Series([pd.NA] * len(result), dtype="Int64")
    result["date"] = pd.to_datetime(result["date"])
    result["signal"] = result["signal"].astype("int64")
    result = result.sort_values(["date", "symbol"]).reset_index(drop=True)
    # Signals dated t use close-t information and are intended for execution at t+1.
    return result[list(SIGNAL_COLUMNS)]


def run_signal_pipeline(
    predictions_dir: str | Path | None = None,
    signals_dir: str | Path | None = None,
    method: str = "threshold",
    **params: Any,
) -> list[Path]:
    """Read prediction parquet files and write schema-conformant signals."""
    source = Path(predictions_dir or settings.PREDICTIONS_DIR)
    destination = Path(signals_dir or settings.SIGNALS_DIR)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for path in sorted(source.glob("*.parquet")):
        frame = pd.read_parquet(path)
        signals = generate_signals(frame, method, **params)
        output = destination / path.name
        signals.to_parquet(output, index=False)
        outputs.append(output)
        logger.info("Wrote signals to %s", output)
    return outputs
