"""Generate threshold or cross-sectional rank signals."""

import argparse
import logging

from config.settings import settings
from src.signals.pipeline import run_signal_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse signal options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=("threshold", "rank", "topn"), default="threshold"
    )
    parser.add_argument("--threshold", type=float, default=settings.SIGNAL_THRESHOLD)
    parser.add_argument("--top-n", type=int, default=settings.TOP_N)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run configured signal generation."""
    args = parse_args(argv)
    params = (
        {"threshold": args.threshold}
        if args.method == "threshold"
        else {"top_n": args.top_n}
    )
    run_signal_pipeline(method=args.method, **params)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
