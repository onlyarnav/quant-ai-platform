"""
Master configuration loader for the Quant AI Research Platform.

Loads all settings from environment variables via .env file.
All engineers should import settings from here — never use os.getenv() directly.

Usage:
    from config.settings import settings
    print(settings.DATABASE_URL)
"""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """Central settings object for the platform."""

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────
    DATABASE_URL: str

    # ── External API Keys (optional integrations) ─────────────
    ALPHA_VANTAGE_KEY: str = Field(default="")
    FRED_API_KEY: str = Field(default="")
    NEWS_API_KEY: str = Field(default="")

    # ── Data Settings ─────────────────────────────────────────
    DATA_START_DATE: str = Field(default="2015-01-01")
    DATA_UPDATE_MODE: str = Field(default="incremental")
    DATA_REFRESH_INTERVAL_HOURS: int = Field(default=24)

    # ── Data Lake Paths ───────────────────────────────────────
    DATA_RAW_PATH: Path = _PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED_PATH: Path = _PROJECT_ROOT / "data" / "processed"
    DATA_FEATURES_PATH: Path = _PROJECT_ROOT / "data" / "features"
    DATA_SIGNALS_PATH: Path = _PROJECT_ROOT / "data" / "signals"
    DATA_PREDICTIONS_PATH: Path = _PROJECT_ROOT / "data" / "predictions"

    # ── Market Universe ───────────────────────────────────────
    MARKET: str = Field(default="NSE")
    STOCK_UNIVERSE: str = Field(default="NIFTY500")

    # ── ML Settings ───────────────────────────────────────────
    MODEL_LOOKBACK_WINDOW: int = Field(default=252)
    MODEL_PREDICTION_HORIZON: int = Field(default=1)
    MODEL_RETRAIN_INTERVAL_DAYS: int = Field(default=7)

    # ── Backtesting ───────────────────────────────────────────
    TRANSACTION_COST: float = Field(default=0.001)
    SLIPPAGE: float = Field(default=0.0005)

    # ── Portfolio ─────────────────────────────────────────────
    PORTFOLIO_RISK_FREE_RATE: float = Field(default=0.07)
    MAX_POSITION_SIZE: float = Field(default=0.05)

    # ── Pipeline Automation ───────────────────────────────────
    PIPELINE_RUN_TIME: str = Field(default="18:30")
    PIPELINE_TIMEZONE: str = Field(default="Asia/Kolkata")

    # ── MLflow ────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = Field(default="http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = Field(default="quant-ai-platform")
    FEATURES_DIR: Path = _PROJECT_ROOT / "data" / "features"
    PREDICTIONS_DIR: Path = _PROJECT_ROOT / "data" / "predictions"
    SIGNALS_DIR: Path = _PROJECT_ROOT / "data" / "signals"
    MODEL_ARTIFACT_DIR: Path = _PROJECT_ROOT / "models" / "artifacts"
    DEFAULT_TARGET_COLUMN: str = Field(default="future_return_1d")
    ML_FEATURE_POLICY: str = Field(default="scale_free")
    SCALE_FREE_FEATURE_COLUMNS: list[str] = Field(default_factory=lambda: [
        "rsi_14", "roc_5", "roc_10", "volatility_10", "volatility_20",
        "volatility_50", "bollinger_width", "zscore_20", "distance_from_sma",
        "distance_from_ema", "volume_ratio",
    ])
    TRAIN_RATIO: float = Field(default=0.70)
    VALIDATION_RATIO: float = Field(default=0.15)
    TEST_RATIO: float = Field(default=0.15)
    WALK_FORWARD_INITIAL_TRAIN_SIZE: int = Field(default=252)
    WALK_FORWARD_TEST_SIZE: int = Field(default=21)
    WALK_FORWARD_STEP_SIZE: int = Field(default=21)
    RANDOM_SEED: int = Field(default=42)
    RETRAIN_INTERVAL_DAYS: int = Field(default=7)
    SIGNAL_THRESHOLD: float = Field(default=0.01)
    TOP_N: int = Field(default=10)
    LINEAR_REGRESSION_PARAMS: dict = Field(default_factory=dict)
    RIDGE_PARAMS: dict = Field(default_factory=lambda: {"alpha": 1.0})
    LASSO_PARAMS: dict = Field(default_factory=lambda: {"alpha": 0.001, "max_iter": 10000})
    RANDOM_FOREST_PARAMS: dict = Field(default_factory=lambda: {"n_estimators": 200, "min_samples_leaf": 2})
    GRADIENT_BOOSTING_PARAMS: dict = Field(default_factory=lambda: {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 2})
    XGBOOST_PARAMS: dict = Field(default_factory=lambda: {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05})
    LIGHTGBM_PARAMS: dict = Field(default_factory=lambda: {"n_estimators": 200, "num_leaves": 15, "learning_rate": 0.05})

    # ── Validators ────────────────────────────────────────────
    @field_validator("DATA_UPDATE_MODE")
    @classmethod
    def validate_update_mode(cls, v: str) -> str:
        allowed = {"full", "incremental"}
        if v not in allowed:
            raise ValueError(f"DATA_UPDATE_MODE must be one of {allowed}, got '{v}'")
        return v


settings = Settings()
