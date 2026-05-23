#!/usr/bin/env python3
from __future__ import annotations

import argparse

from src.rl_runner import run_dqn_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the SEACTS DQN agent.")
    parser.add_argument("--config", default="configs/depmap_baselines.yaml")
    parser.add_argument("--raw-data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results, output_path = run_dqn_training_pipeline(
        config_path=args.config,
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
    )
    for row in results.to_dict(orient="records"):
        print(row)
    print({"output_path": str(output_path)})


if __name__ == "__main__":
    main()
