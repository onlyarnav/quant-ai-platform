"""Generate predictions from stored feature parquet data."""

import argparse
import logging

from config.settings import settings
from src.models.predict import predict, predict_latest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse prediction options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ridge")
    parser.add_argument("--symbol")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--latest", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Predict for the requested symbol or all configured files."""
    args = parse_args(argv)
    if args.symbol:
        from pathlib import Path

        matches = [
            p
            for p in Path(settings.FEATURES_DIR).glob("*.parquet")
            if args.symbol.lower() in p.stem.lower()
        ]
        if not matches:
            raise FileNotFoundError(f"No feature parquet matches symbol {args.symbol}")
        for path in matches:
            predict(path, args.model, args.start_date, args.end_date)
    elif args.latest:
        predict_latest(args.model)
    else:
        from pathlib import Path

        for path in sorted(Path(settings.FEATURES_DIR).glob("*.parquet")):
            predict(path, args.model, args.start_date, args.end_date)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
