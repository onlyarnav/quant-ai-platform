"""Model input feature selection excludes leakage columns."""

import pandas as pd
import pytest
from src.models.dataset import clean_feature_data, split_features_and_target


def test_scale_free_features_default_and_all_numeric_override():
    """Targets and identifiers are never features; optional policy adds levels."""
    frame = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "date": pd.date_range("2024-01-01", periods=2),
            "rsi_14": [40.0, 50.0],
            "close": [10.0, 11.0],
            "future_return_1d": [0.1, 0.2],
        }
    )
    clean = clean_feature_data(frame)
    X, _, _ = split_features_and_target(clean)
    assert X.columns.tolist() == ["rsi_14"]
    X_all, _, _ = split_features_and_target(clean, feature_policy="all_numeric")
    assert "close" in X_all and not any(c.startswith("future_return") for c in X_all)
    with pytest.raises(ValueError):
        clean_feature_data(frame, feature_policy="bad")
