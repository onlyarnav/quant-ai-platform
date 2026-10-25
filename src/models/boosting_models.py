"""Optional XGBoost and LightGBM builders."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def xgboost_regressor(params: dict[str, Any] | None = None) -> Any:
    """Build a seeded XGBoost regressor."""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("Install xgboost to use the xgboost model") from exc
    return XGBRegressor(**(params or {}))


def lightgbm_regressor(params: dict[str, Any] | None = None) -> Any:
    """Build a seeded LightGBM regressor."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError("Install lightgbm to use the lightgbm model") from exc
    return LGBMRegressor(**(params or {}))
