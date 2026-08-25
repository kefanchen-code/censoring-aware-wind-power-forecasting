"""Command-line entry point for the PAP benchmark v2."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from censored_wind_power.benchmark.core import BenchmarkError, list_models
from censored_wind_power.benchmark.protocol import load_config
from censored_wind_power.benchmark.runner import aggregate_existing, run_benchmark


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Reproducible benchmark for potential available power forecasting"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "pap_benchmark_v2.json",
        help="benchmark JSON configuration",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="fit, predict, save, and compare")
    run.add_argument("--models", nargs="+", default=None)
    run.add_argument("--scenarios", nargs="+", default=None)
    run.add_argument("--seeds", nargs="+", type=int, default=None)
    run.add_argument("--force", action="store_true", help="replace compatible artifacts")
    run.add_argument(
        "--smoke",
        action="store_true",
        help="one scenario, one seed, one epoch, and a separate smoke output",
    )

    compare = subparsers.add_parser(
        "compare", help="aggregate existing compatible forecast artifacts only"
    )
    compare.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write a provisional summary even when configured artifacts are missing",
    )
    subparsers.add_parser("list-models", help="list currently registered adapters")
    return parser


def main() -> int:
    """Execute one benchmark command."""

    arguments = build_parser().parse_args()
    config_path = arguments.config.resolve()
    try:
        if arguments.command == "run":
            result = run_benchmark(
                PROJECT_DIR,
                config_path,
                models_override=arguments.models,
                scenarios_override=arguments.scenarios,
                seeds_override=arguments.seeds,
                smoke=arguments.smoke,
                force=arguments.force,
            )
        elif arguments.command == "compare":
            config = load_config(config_path)
            result = aggregate_existing(
                PROJECT_DIR,
                config,
                require_complete=not arguments.allow_incomplete,
            )
        else:
            # Loading the runner above registers built-in models.
            result = {"registered_models": list_models()}
    except BenchmarkError as error:
        print("benchmark error: %s" % error, file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
