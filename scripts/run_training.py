"""Train and compare configured ML models from feature parquet data."""

import argparse
import logging
from pathlib import Path

from config.settings import settings
from src.models.train import train_all_models, train_model, walk_forward_validate

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse training CLI options."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbol")
    group.add_argument("--all", action="store_true", dest="all_symbols")
    parser.add_argument("--model", default="ridge")
    parser.add_argument("--target", default=settings.DEFAULT_TARGET_COLUMN)
    parser.add_argument("--walk-forward", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run training for one or all feature parquet files."""
    args = parse_args(argv)
    paths = sorted(Path(settings.FEATURES_DIR).glob("*.parquet"))
    if args.symbol:
        paths = [path for path in paths if args.symbol.lower() in path.stem.lower()]
    if not paths:
        raise FileNotFoundError(
            f"No matching feature parquet files in {settings.FEATURES_DIR}"
        )
    for path in paths:
        if args.walk_forward:
            logger.info("%s", walk_forward_validate(path, args.model, args.target))
        elif args.all_symbols:
            logger.info(
                "%s", train_all_models(path, args.target).to_string(index=False)
            )
        else:
            logger.info("%s", train_model(path, args.model, args.target)["metrics"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
