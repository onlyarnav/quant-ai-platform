"""Optional booster integration tests and actionable missing-library errors."""

import builtins

import numpy as np
import pytest

from src.models.boosting_models import lightgbm_regressor, xgboost_regressor


@pytest.fixture(params=["xgboost", "lightgbm"])
def booster(request):
    """Import optional native libraries, skipping only when unavailable here."""
    try:
        if request.param == "xgboost":
            import xgboost  # noqa: F401
    except Exception as exc:
        pytest.skip(f"optional xgboost dependency unavailable: {exc}")
    if request.param == "xgboost":
        return xgboost_regressor({"n_estimators": 3, "n_jobs": 1})
    try:
        import lightgbm  # noqa: F401
    except Exception as exc:
        pytest.skip(f"optional lightgbm dependency unavailable: {exc}")
    return lightgbm_regressor({"n_estimators": 3, "verbosity": -1})


def test_booster_fit_predict_shape_and_finiteness(booster):
    """Each available booster fits deterministic synthetic regression data."""
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.sin(np.arange(20, dtype=float))
    booster.fit(X, y)
    prediction = booster.predict(X[:5])
    assert prediction.shape == (5,)
    assert np.isfinite(prediction).all()


@pytest.mark.parametrize(
    "module_name,builder",
    [("xgboost", xgboost_regressor), ("lightgbm", lightgbm_regressor)],
)
def test_missing_optional_dependency_has_clear_error(monkeypatch, module_name, builder):
    """An unavailable optional package raises an actionable ImportError."""
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == module_name:
            raise ImportError(f"blocked {module_name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match=module_name):
        builder()
