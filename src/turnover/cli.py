"""Command line entry point:  turnover run --data data/MFG10YearTerminationData.csv"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from turnover.data import DataError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="turnover", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run the analysis and write the report")
    run.add_argument("--data", type=Path, default=Path("data/MFG10YearTerminationData.csv"))
    run.add_argument("--out", type=Path, default=Path("reports"))
    run.add_argument("--first-test-year", type=int, default=2014)
    run.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    from turnover.pipeline import run as run_pipeline

    try:
        run_pipeline(args.data, args.out, args.first_test_year, args.seed)
    except DataError as error:
        logging.getLogger("turnover").error("%s", error)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
