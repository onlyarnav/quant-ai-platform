"""Feature dataset assembly from processed parquet inputs."""

import logging
from pathlib import Path

import pandas as pd

from config.settings import settings
from src.feature_engineering.mean_reversion import add_mean_reversion_features
from src.feature_engineering.momentum import add_momentum_features
from src.feature_engineering.targets import add_future_return_targets
from src.feature_engineering.trend import add_trend_features
from src.feature_engineering.volatility import add_volatility_features
from src.feature_engineering.volume import add_volume_features

logger = logging.getLogger(__name__)


def build_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Create trailing, point-in-time features and forward-return labels.

    Args:
        df: Processed prices, with one symbol or a multi-symbol panel.

    Returns:
        Feature frame sorted by date and symbol. Calculations are grouped by
        symbol so rolling windows and forward labels never cross assets.
    """
    required = {"symbol", "date", "close", "high", "low", "volume", "returns"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Processed data is missing columns: {sorted(missing)}")
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values(["symbol", "date"]).reset_index(drop=True)
    pieces: list[pd.DataFrame] = []
    for _, group in result.groupby("symbol", sort=False):
        feature = add_trend_features(group)
        feature = add_momentum_features(feature)
        feature = add_volatility_features(feature)
        feature = add_mean_reversion_features(feature)
        feature = add_volume_features(feature)
        pieces.append(add_future_return_targets(feature))
    output = (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
    logger.info("Built feature dataset with shape %s", output.shape)
    return output


def run_feature_pipeline(
    processed_dir: str | Path | None = None, output_dir: str | Path | None = None
) -> list[Path]:
    """Build feature parquet files from processed parquet files."""
    source = Path(processed_dir or settings.DATA_PROCESSED_PATH)
    destination = Path(output_dir or settings.FEATURES_DIR)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for path in sorted(source.glob("**/*.parquet")):
        frame = pd.read_parquet(path)
        featured = build_feature_dataset(frame)
        output = destination / path.name
        featured.to_parquet(output, index=False)
        outputs.append(output)
        logger.info("Wrote %s", output)
    return outputs
