"""Regression and cross-sectional prediction metrics."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _correlation(left: np.ndarray, right: np.ndarray, method: str = "pearson") -> float:
    """Compute a finite-value correlation, returning NaN for degenerate input."""
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 2 or np.std(left[mask]) == 0 or np.std(right[mask]) == 0:
        logger.warning("Correlation undefined for constant or insufficient arrays")
        return float("nan")
    if method == "spearman":
        return float(
            pd.Series(left[mask]).corr(pd.Series(right[mask]), method="spearman")
        )
    return float(np.corrcoef(left[mask], right[mask])[0, 1])


def _arrays(y_true: object, y_pred: object) -> tuple[np.ndarray, np.ndarray]:
    """Coerce paired numeric values and remove non-finite pairs."""
    left, right = (
        np.asarray(y_true, dtype=float).ravel(),
        np.asarray(y_pred, dtype=float).ravel(),
    )
    if left.shape != right.shape:
        raise ValueError("y_true and y_pred must have the same length")
    mask = np.isfinite(left) & np.isfinite(right)
    return left[mask], right[mask]


def mse(y_true: object, y_pred: object) -> float:
    """Mean squared error."""
    a, b = _arrays(y_true, y_pred)
    return float(np.mean((a - b) ** 2)) if len(a) else float("nan")


def rmse(y_true: object, y_pred: object) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: object, y_pred: object) -> float:
    """Mean absolute error."""
    a, b = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(a - b))) if len(a) else float("nan")


def r2(y_true: object, y_pred: object) -> float:
    """Coefficient of determination, undefined for constant targets."""
    a, b = _arrays(y_true, y_pred)
    if len(a) < 2 or np.var(a) == 0:
        logger.warning("R2 undefined for constant or insufficient targets")
        return float("nan")
    return float(1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2))


def directional_accuracy(y_true: object, y_pred: object) -> float:
    """Fraction of predictions with matching return sign."""
    a, b = _arrays(y_true, y_pred)
    return float(np.mean(np.sign(a) == np.sign(b))) if len(a) else float("nan")


def information_coefficient(y_true: object, y_pred: object) -> float:
    """Pearson correlation between predictions and outcomes."""
    a, b = _arrays(y_true, y_pred)
    return _correlation(a, b)


def rank_ic(y_true: object, y_pred: object) -> float:
    """Spearman correlation between predictions and outcomes."""
    a, b = _arrays(y_true, y_pred)
    return _correlation(a, b, "spearman")


def evaluate_predictions(y_true: object, y_pred: object) -> dict[str, float]:
    """Return all standard regression and investment ranking metrics."""
    return {
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "ic": information_coefficient(y_true, y_pred),
        "rank_ic": rank_ic(y_true, y_pred),
    }


def cross_sectional_ic(
    df: pd.DataFrame, date_col: str, pred_col: str, target_col: str
) -> dict[str, float]:
    """Calculate per-date Pearson IC and aggregate mean, standard deviation, IR."""
    missing = {date_col, pred_col, target_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    values = [
        information_coefficient(group[target_col], group[pred_col])
        for _, group in df.groupby(date_col)
    ]
    finite = np.asarray(values)[np.isfinite(values)]
    mean = float(finite.mean()) if len(finite) else float("nan")
    std = float(finite.std(ddof=1)) if len(finite) > 1 else float("nan")
    return {
        "mean_ic": mean,
        "ic_std": std,
        "ic_ir": mean / std if np.isfinite(std) and std != 0 else float("nan"),
    }
