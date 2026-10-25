"""Metric behavior on valid and degenerate synthetic returns."""

import numpy as np
import pandas as pd
import pytest
from src.models.evaluation import evaluate_predictions, cross_sectional_ic


def test_metrics_and_degenerate_inputs():
    """All metric outputs are defined, including constant inputs."""
    metrics = evaluate_predictions([1, 2, 3], [1, 2, 2])
    assert metrics["mse"] == pytest.approx(1 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert metrics["mae"] == pytest.approx(1 / 3)
    assert metrics["r2"] == pytest.approx(0.5)
    assert metrics["directional_accuracy"] == 1.0
    assert set(metrics) == {
        "mse",
        "rmse",
        "mae",
        "r2",
        "directional_accuracy",
        "ic",
        "rank_ic",
    }
    assert np.isnan(evaluate_predictions([1, 1], [1, 2])["ic"])
    assert np.isnan(evaluate_predictions([np.nan], [1])["mae"])
    with pytest.raises(ValueError, match="same length"):
        evaluate_predictions([1, 2], [1])


def test_cross_sectional_ic_summary():
    """Cross-sectional IC returns a mean, standard deviation, and ratio."""
    frame = pd.DataFrame(
        {"date": [1, 1, 1, 2, 2, 2], "p": [1, 2, 3, 1, 2, 3], "y": [2, 4, 6, 1, 3, 5]}
    )
    assert cross_sectional_ic(frame, "date", "p", "y")["mean_ic"] == 1.0
    with pytest.raises(ValueError, match="Missing columns"):
        cross_sectional_ic(frame, "date", "missing", "y")
