#!/usr/bin/env python3
from __future__ import annotations

import argparse

from src.baseline_runner import run_baseline_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run candidate-selection baselines.")
    parser.add_argument("--config", default="configs/depmap_baselines.yaml")
    parser.add_argument("--raw-data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results, output_path = run_baseline_pipeline(
        config_path=args.config,
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
    )

    print(results.to_string(index=False))
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
