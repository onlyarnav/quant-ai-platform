"""Scaled linear regression model builders."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def linear_regression(params: dict[str, Any] | None = None) -> Any:
    """Build a linear model with train-fitted standardization."""
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("Install scikit-learn to train linear models") from exc
    logger.info("Creating scaled LinearRegression")
    return Pipeline(
        [("scaler", StandardScaler()), ("model", LinearRegression(**(params or {})))]
    )


def ridge(params: dict[str, Any] | None = None) -> Any:
    """Build a scaled Ridge model."""
    try:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("Install scikit-learn to train Ridge") from exc
    return Pipeline([("scaler", StandardScaler()), ("model", Ridge(**(params or {})))])


def lasso(params: dict[str, Any] | None = None) -> Any:
    """Build a scaled Lasso model."""
    try:
        from sklearn.linear_model import Lasso
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("Install scikit-learn to train Lasso") from exc
    return Pipeline([("scaler", StandardScaler()), ("model", Lasso(**(params or {})))])
