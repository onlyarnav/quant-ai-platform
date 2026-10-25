"""Leakage-aware chronological split utilities."""

from collections.abc import Iterator, Sequence
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _date_groups(dates: Sequence[object] | pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted unique dates and a row-to-date group index."""
    values = pd.to_datetime(pd.Series(dates), errors="coerce")
    if values.isna().any():
        raise ValueError("dates must not contain missing or invalid values")
    unique = pd.Index(values.unique()).sort_values()
    if len(unique) < 2:
        raise ValueError("at least two unique dates are required")
    return unique.to_numpy(), np.asarray(unique.get_indexer(values), dtype=int)


def chronological_split(
    dates: Sequence[object] | pd.Series,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    gap: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split panel row indices by date into train, validation and test sets.

    ``gap`` removes whole unique date periods before validation and test to
    purge labels whose forward horizon could overlap the next split.
    """
    if (
        not 0 < train_ratio < 1
        or not 0 <= val_ratio < 1
        or train_ratio + val_ratio >= 1
    ):
        raise ValueError(
            "train_ratio and val_ratio must be valid and sum to less than 1"
        )
    if not isinstance(gap, int) or gap < 0:
        raise ValueError("gap must be a non-negative integer")
    unique, groups = _date_groups(dates)
    n = len(unique)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    if train_end < 1 or val_end <= train_end or val_end >= n:
        raise ValueError("ratios leave an empty train, validation, or test split")
    train_dates = np.arange(max(0, train_end - gap))
    val_dates = np.arange(train_end + gap, max(train_end + gap, val_end - gap))
    test_dates = np.arange(val_end + gap, n)
    if not len(train_dates) or not len(val_dates) or not len(test_dates):
        raise ValueError("gap is too large for the requested split ratios")
    logger.info(
        "Chronological split: train=%d val=%d test=%d dates",
        len(train_dates),
        len(val_dates),
        len(test_dates),
    )
    return tuple(np.flatnonzero(np.isin(groups, part)) for part in (train_dates, val_dates, test_dates))  # type: ignore[return-value]


def walk_forward_splits(
    dates: Sequence[object] | pd.Series,
    initial_train_size: int,
    test_size: int,
    step_size: int,
    expanding: bool = True,
    gap: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield train/test row indices over date periods, never shuffled."""
    sizes = (initial_train_size, test_size, step_size, gap)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in sizes):
        raise ValueError("window sizes and gap must be integers")
    if min(initial_train_size, test_size, step_size) < 1 or gap < 0:
        raise ValueError("window sizes must be positive and gap non-negative")
    unique, groups = _date_groups(dates)
    start = initial_train_size
    while start + gap < len(unique):
        test_start = start + gap
        test_end = min(test_start + test_size, len(unique))
        train_start = 0 if expanding else max(0, start - initial_train_size)
        train_dates = np.arange(train_start, start)
        test_dates = np.arange(test_start, test_end)
        if not len(test_dates):
            break
        yield np.flatnonzero(np.isin(groups, train_dates)), np.flatnonzero(
            np.isin(groups, test_dates)
        )
        start += step_size
