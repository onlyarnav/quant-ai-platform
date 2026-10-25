"""End-to-end synthetic features-to-signals workflow verification."""

import numpy as np
import pandas as pd
from config.settings import settings
from src.feature_engineering.pipeline import build_feature_dataset
from src.models.train import train_model
from src.models.predict import predict
from src.signals.pipeline import run_signal_pipeline


def test_features_training_prediction_signals_roundtrip(tmp_path, monkeypatch):
    """All pipeline stages write parquet with the promised output schemas."""
    n = 320
    rng = np.random.default_rng(77)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    dates = pd.date_range("2023-01-01", periods=n)
    raw = pd.DataFrame(
        {
            "symbol": "TEST",
            "asset_class": "equity",
            "exchange": "NSE",
            "currency": "INR",
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": rng.integers(100, 1000, n),
            "returns": pd.Series(close).pct_change().fillna(0),
            "log_returns": np.log(pd.Series(close) / pd.Series(close).shift()).fillna(
                0
            ),
        }
    )
    feature_path = tmp_path / "equity_TEST_daily.parquet"
    build_feature_dataset(raw).to_parquet(feature_path, index=False)
    monkeypatch.setattr(
        settings, "MLFLOW_TRACKING_URI", f"file://{tmp_path / 'mlflow'}"
    )
    monkeypatch.setattr(settings, "MODEL_ARTIFACT_DIR", tmp_path / "artifacts")
    trained = train_model(feature_path, "ridge", artifact_dir=tmp_path / "artifacts")
    assert trained["metadata"]["feature_columns"]
    predictions = predict(
        feature_path,
        "ridge",
        output_dir=tmp_path / "predictions",
        artifact_dir=tmp_path / "artifacts",
    )
    assert predictions.columns.tolist() == [
        "symbol",
        "date",
        "predicted_return",
        "model_name",
        "model_version",
    ]
    prediction_dir = tmp_path / "predictions"
    signals_path = run_signal_pipeline(
        prediction_dir, tmp_path / "signals", "threshold", threshold=0.001
    )[0]
    signals = pd.read_parquet(signals_path)
    assert signals.columns.tolist() == [
        "symbol",
        "date",
        "predicted_return",
        "signal",
        "confidence",
        "rank",
    ]
    assert set(signals["signal"]) <= {-1, 0, 1}
