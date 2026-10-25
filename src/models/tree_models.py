"""Tree model builders."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def random_forest(params: dict[str, Any] | None = None) -> Any:
    """Build a deterministically seeded random forest regressor."""
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as exc:
        raise ImportError("Install scikit-learn to train tree models") from exc
    return RandomForestRegressor(**(params or {}))


def gradient_boosting(params: dict[str, Any] | None = None) -> Any:
    """Build a gradient boosting regressor."""
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError as exc:
        raise ImportError("Install scikit-learn to train tree models") from exc
    return GradientBoostingRegressor(**(params or {}))
