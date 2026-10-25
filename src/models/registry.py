"""Local, versioned model artifact persistence."""

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)


def dataset_hash(path: str | Path) -> str:
    """Hash a dataset file in bounded-memory chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_model(
    model: Any,
    model_name: str,
    target_column: str,
    feature_columns: list[str],
    train_date_range: tuple[object, object],
    metrics: dict[str, float],
    dataset_hash_value: str,
    artifact_dir: str | Path | None = None,
    feature_policy: str = "scale_free",
) -> dict[str, Any]:
    """Save a model and JSON metadata under a timestamp version."""
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("Install joblib to save model artifacts") from exc
    root = Path(artifact_dir or settings.MODEL_ARTIFACT_DIR) / model_name
    root.mkdir(parents=True, exist_ok=True)
    version = "v" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    artifact = root / f"{version}.joblib"
    metadata = {
        "model_name": model_name,
        "version": version,
        "target_column": target_column,
        "feature_columns": list(feature_columns),
        "feature_policy": feature_policy,
        "train_date_range": [str(x) for x in train_date_range],
        "metrics": metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_hash": dataset_hash_value,
        "artifact_path": str(artifact),
    }
    try:
        joblib.dump(model, artifact)
        (root / f"{version}.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=True), encoding="utf-8"
        )
    except Exception:
        artifact.unlink(missing_ok=True)
        raise
    logger.info("Saved model %s version %s", model_name, version)
    return metadata


def get_latest_model(
    model_name: str, artifact_dir: str | Path | None = None
) -> dict[str, Any]:
    """Load the newest model artifact and its ordered feature metadata."""
    root = Path(artifact_dir or settings.MODEL_ARTIFACT_DIR) / model_name
    metadata_files = sorted(root.glob("v*.json"))
    if not metadata_files:
        raise FileNotFoundError(f"No saved model found for {model_name!r} in {root}")
    metadata_path = metadata_files[-1]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("Install joblib to load model artifacts") from exc
    metadata["model"] = joblib.load(metadata["artifact_path"])
    return metadata
