"""Chronological and walk-forward split invariants."""

import numpy as np
import pandas as pd
import pytest
from src.models.splitting import chronological_split, walk_forward_splits


def test_panel_splits_are_chronological_and_keep_dates():
    """No date is split across sets, and a purge gap is respected."""
    dates = pd.Series(list(pd.date_range("2024-01-01", periods=30).repeat(3)))
    train, val, test = chronological_split(dates, 0.6, 0.2, gap=2)
    assert dates.iloc[train].max() < dates.iloc[val].min()
    assert (dates.iloc[val].min() - dates.iloc[train].max()).days >= 3
    assert set(dates.iloc[train]).isdisjoint(set(dates.iloc[val]))
    assert dates.iloc[val].max() < dates.iloc[test].min()
    unique_dates = pd.Index(dates.unique()).sort_values()
    train_dates = unique_dates.get_indexer(pd.Index(dates.iloc[train].unique()))
    val_dates = unique_dates.get_indexer(pd.Index(dates.iloc[val].unique()))
    assert train_dates.max() + 2 < val_dates.min()


def test_walk_forward_has_gap_and_rejects_invalid_sizes():
    """Each walk-forward train period precedes its test period."""
    dates = pd.Series(pd.date_range("2024-01-01", periods=30).repeat(2))
    folds = list(walk_forward_splits(dates, 10, 4, 4, gap=2))
    assert folds
    for train, test in folds:
        assert dates.iloc[train].max() < dates.iloc[test].min()
        assert (dates.iloc[test].min() - dates.iloc[train].max()).days >= 3
    with pytest.raises(ValueError):
        list(walk_forward_splits(dates, 0, 4, 4))
    with pytest.raises(ValueError, match="integers"):
        list(walk_forward_splits(dates, 10, 4, 4, gap=1.5))
    rolling = list(walk_forward_splits(dates, 10, 4, 4, expanding=False))
    assert rolling
    assert len(rolling[-1][0]) <= len(rolling[0][0])


def test_chronological_split_rejects_empty_invalid_and_too_small_inputs():
    """Invalid ratios, missing dates, and excessive purge gaps are rejected."""
    dates = pd.date_range("2025-01-01", periods=10)
    for ratios in ((0, 0.5), (0.5, 0.5), (0.7, 0.4), (np.nan, 0.1)):
        with pytest.raises(ValueError):
            chronological_split(dates, *ratios)
    with pytest.raises(ValueError, match="missing or invalid"):
        chronological_split([dates[0], pd.NaT, dates[2]])
    with pytest.raises(ValueError, match="at least two"):
        chronological_split([dates[0], dates[0]])
    with pytest.raises(ValueError, match="gap is too large"):
        chronological_split(dates, 0.6, 0.2, gap=10)
