"""Synthetic fit and scaling tests for supported baseline regressors."""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.models.baseline import lasso, linear_regression, ridge


@pytest.mark.parametrize(
    "builder,params",
    [(linear_regression, {}), (ridge, {"alpha": 0.1}), (lasso, {"alpha": 0.01})],
)
def test_baseline_models_fit_predict_and_scale_train_only(builder, params):
    """Baselines produce finite predictions and fit scaling from training rows."""
    model = builder(params)
    assert isinstance(model, Pipeline)
    train_x = np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]])
    model.fit(train_x, np.array([0.0, 1.0, 2.0]))
    validation_x = np.array([[1000.0, -1000.0]])
    predictions = model.predict(validation_x)
    assert predictions.shape == (1,)
    assert np.isfinite(predictions).all()
    np.testing.assert_allclose(model.named_steps["scaler"].mean_, [2.0, 4.0])


def test_baseline_invalid_data_and_parameters_raise():
    """Estimator validation rejects bad parameters and non-finite training rows."""
    with pytest.raises(TypeError):
        ridge({"not_a_parameter": 1})
    with pytest.raises(ValueError):
        linear_regression().fit([[np.nan], [1.0]], [0.0, 1.0])
