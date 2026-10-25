"""Dataset preparation tests for cleaning, schemas, and leakage prevention."""

import numpy as np
import pandas as pd
import pytest

from src.models.dataset import (
    build_ml_dataset,
    clean_feature_data,
    load_feature_data,
    split_features_and_target,
)


@pytest.fixture
def features() -> pd.DataFrame:
    """Return a small panel with scale-free, level, and multiple label columns."""
    return pd.DataFrame(
        {
            "symbol": ["A", "A", "B", "B"],
            "date": pd.date_range("2025-01-01", periods=4),
            "rsi_14": [30.0, 40.0, np.inf, 60.0],
            "roc_5": [1.0, np.nan, 3.0, 4.0],
            "close": [100.0, 101.0, 200.0, 202.0],
            "future_return_1d": [0.01, 0.02, 0.03, np.inf],
            "future_return_5d": [0.1, 0.2, 0.3, 0.4],
            "future_return_10d": [0.5, 0.6, 0.7, 0.8],
            "exchange": ["NSE"] * 4,
        }
    )


def test_cleaning_drops_nonfinite_rows_without_mutation(features):
    """NaN and infinities in selected inputs or target are dropped on a copy."""
    before = features.copy(deep=True)
    clean = clean_feature_data(features)
    assert clean["symbol"].tolist() == ["A"]
    assert features.equals(before)
    assert np.isfinite(clean[["rsi_14", "roc_5", "future_return_1d"]]).all().all()


def test_feature_target_separation_excludes_every_forward_label(features):
    """All future-return columns stay out of X, regardless of target chosen."""
    clean = clean_feature_data(features, feature_policy="all_numeric")
    X, y, dates = split_features_and_target(
        clean, "future_return_5d", feature_policy="all_numeric"
    )
    assert X.columns.tolist() == ["rsi_14", "roc_5", "close"]
    assert not any(column.startswith("future_return_") for column in X.columns)
    assert y.equals(clean["future_return_5d"])
    assert dates.equals(clean["date"])


def test_dataset_parquet_round_trip_and_bad_paths(features, tmp_path):
    """The file API reads valid parquet and reports missing or corrupt files."""
    path = tmp_path / "features.parquet"
    features.to_parquet(path, index=False)
    X, y, dates = build_ml_dataset(path)
    assert list(X.columns) == ["rsi_14", "roc_5"]
    assert len(X) == len(y) == len(dates) == 1
    assert load_feature_data(path).shape == features.shape
    with pytest.raises(FileNotFoundError):
        load_feature_data(tmp_path / "missing.parquet")
    corrupt = tmp_path / "corrupt.parquet"
    corrupt.write_text("not parquet")
    with pytest.raises(ValueError, match="Failed to read Parquet"):
        load_feature_data(corrupt)


def test_dataset_rejects_missing_contract_columns_and_unknown_policy(features):
    """Missing date/target/features and unknown feature policies fail clearly."""
    with pytest.raises(ValueError, match="date"):
        clean_feature_data(features.drop(columns="date"))
    with pytest.raises(ValueError, match="Target column"):
        clean_feature_data(features, "missing_target")
    with pytest.raises(ValueError, match="feature_policy"):
        clean_feature_data(features, feature_policy="unknown")
    no_scale_free = features.drop(columns=["rsi_14", "roc_5"])
    with pytest.raises(ValueError, match="No model features"):
        clean_feature_data(no_scale_free)
