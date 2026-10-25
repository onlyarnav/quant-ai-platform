"""Central registry for supported regression estimators."""

import logging
from typing import Any, Callable

from config.settings import settings
from src.models.baseline import lasso, linear_regression, ridge
from src.models.boosting_models import lightgbm_regressor, xgboost_regressor
from src.models.tree_models import gradient_boosting, random_forest

logger = logging.getLogger(__name__)


def _defaults() -> dict[str, dict[str, Any]]:
    """Return parameter dictionaries from the settings object."""
    return {
        "linear": settings.LINEAR_REGRESSION_PARAMS,
        "ridge": settings.RIDGE_PARAMS,
        "lasso": settings.LASSO_PARAMS,
        "random_forest": settings.RANDOM_FOREST_PARAMS,
        "gradient_boosting": settings.GRADIENT_BOOSTING_PARAMS,
        "xgboost": settings.XGBOOST_PARAMS,
        "lightgbm": settings.LIGHTGBM_PARAMS,
    }


REGISTRY: dict[str, Callable[[dict[str, Any] | None], Any]] = {
    "linear": linear_regression,
    "ridge": ridge,
    "lasso": lasso,
    "random_forest": random_forest,
    "gradient_boosting": gradient_boosting,
    "xgboost": xgboost_regressor,
    "lightgbm": lightgbm_regressor,
}


def get_model(name: str, params: dict[str, Any] | None = None) -> Any:
    """Construct a registered model using configured defaults and seed."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Valid models: {', '.join(REGISTRY)}")
    values = dict(_defaults()[name])
    if params:
        values.update(params)
    if name in {"random_forest", "gradient_boosting", "xgboost", "lightgbm"}:
        values.setdefault("random_state", settings.RANDOM_SEED)
    logger.info("Building %s model with params %s", name, values)
    return REGISTRY[name](values)
