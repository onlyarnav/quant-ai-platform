"""Synthetic training and determinism tests for tree regressors."""

import numpy as np
import pytest

from src.models.model_factory import get_model
from src.models.tree_models import gradient_boosting, random_forest


@pytest.mark.parametrize("name", ["random_forest", "gradient_boosting"])
def test_tree_models_train_and_predict_reproducibly(name):
    """Factory-seeded tree models fit and repeat predictions on same data."""
    X = np.arange(30, dtype=float).reshape(15, 2)
    y = np.linspace(-0.1, 0.1, len(X))
    first, second = get_model(name, {"n_estimators": 8}), get_model(
        name, {"n_estimators": 8}
    )
    first.fit(X, y)
    second.fit(X, y)
    assert first.predict(X[:4]).shape == (4,)
    np.testing.assert_allclose(first.predict(X), second.predict(X))


def test_tree_wrappers_accept_explicit_params_and_reject_invalid_configs():
    """Direct builders preserve estimator APIs and defer invalid-param errors."""
    assert random_forest({"n_estimators": 3}).n_estimators == 3
    assert gradient_boosting({"n_estimators": 4}).n_estimators == 4
    with pytest.raises(ValueError):
        random_forest({"n_estimators": 0}).fit([[0], [1]], [0, 1])
