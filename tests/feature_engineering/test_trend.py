"""Synthetic correctness tests for point-in-time feature calculations."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.pipeline import build_feature_dataset
from src.feature_engineering.targets import add_future_return_targets
from src.feature_engineering.trend import add_trend_features
from src.feature_engineering.momentum import add_momentum_features
from src.feature_engineering.volatility import add_volatility_features
from src.feature_engineering.mean_reversion import add_mean_reversion_features
from src.feature_engineering.volume import add_volume_features


def prices(n: int = 240) -> pd.DataFrame:
    """Build deterministic prices with enough history for all rolling windows."""
    close = 100 + np.arange(n) * 0.1 + np.sin(np.arange(n) / 5)
    return pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.date_range("2024-01-01", periods=n),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "adj_close": close,
            "volume": 1000 + np.arange(n),
            "returns": pd.Series(close).pct_change().fillna(0),
            "log_returns": np.log(pd.Series(close) / pd.Series(close).shift()).fillna(
                0
            ),
        }
    )


def test_feature_output_is_copy_and_trailing():
    """Feature helpers preserve input and use trailing observations only."""
    raw = prices()
    original = raw.copy()
    output = add_trend_features(raw)
    assert raw.equals(original)
    changed = raw.copy()
    changed.loc[220:, "close"] *= 3
    assert (
        add_trend_features(changed)
        .loc[:219, "sma_20"]
        .equals(output.loc[:219, "sma_20"])
    )


def test_feature_family_outputs():
    """Feature transforms expose expected contract columns and handle warmups."""
    raw = prices()
    assert "rsi_14" in add_momentum_features(raw)
    assert "volatility_20" in add_volatility_features(raw)
    assert "distance_from_sma" in add_mean_reversion_features(raw)
    volume = add_volume_features(raw)
    assert {"volume_ma_20", "volume_ratio", "obv"} <= set(volume.columns)
    assert volume["volume_ratio"].notna().any()


def test_targets_are_grouped_and_forward():
    """Forward labels never flow across symbols and terminal rows are missing."""
    frame = pd.concat(
        [prices(15), prices(15).assign(symbol="BBB", close=lambda x: x.close + 10)],
        ignore_index=True,
    )
    targets = add_future_return_targets(frame, (1, 5))
    assert targets.loc[14, "future_return_1d"] != pytest.approx(
        (frame.loc[15, "close"] / frame.loc[14, "close"]) - 1
    )
    assert pd.isna(targets.loc[14, "future_return_1d"])
    assert pd.isna(targets.loc[29, "future_return_5d"])


def test_pipeline_multi_symbol_sorted_and_schema():
    """Panel feature generation is deterministic and retains targets."""
    raw = pd.concat(
        [prices(), prices().assign(symbol="BBB")], ignore_index=True
    ).sample(frac=1, random_state=3)
    features = build_feature_dataset(raw)
    assert features[["date", "symbol"]].equals(
        features[["date", "symbol"]]
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
    assert {
        "sma_5",
        "rsi_14",
        "future_return_1d",
        "future_return_5d",
        "future_return_10d",
    } <= set(features.columns)
