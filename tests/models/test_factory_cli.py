"""Factory validation and side-effect-free CLI argument parsing."""

import pytest
from src.models.model_factory import REGISTRY, get_model
from scripts.run_training import parse_args as training_args
from scripts.run_prediction import parse_args as prediction_args
from scripts.run_signals import parse_args as signals_args


def test_factory_names_and_cli_parsing():
    """All public model names are registered and command parsers are importable."""
    assert {
        "ridge",
        "lasso",
        "linear",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "lightgbm",
    } == set(REGISTRY)
    with pytest.raises(ValueError, match="Valid models"):
        get_model("mystery")
    assert training_args(["--symbol", "AAA", "--walk-forward"]).symbol == "AAA"
    assert prediction_args(["--latest"]).latest
    assert signals_args(["--method", "rank", "--top-n", "2"]).top_n == 2


def test_factory_applies_model_parameter_overrides():
    """Explicit factory params override the configured baseline defaults."""
    assert get_model("ridge", {"alpha": 0.1}).get_params()["model__alpha"] == 0.1
