#!/usr/bin/env python3
from __future__ import annotations

import argparse

from src.visualization import generate_baseline_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot data and RL environment baseline results.")
    parser.add_argument(
        "--data-metrics",
        default="outputs/depmap_baselines/data_baseline_metrics.csv",
    )
    parser.add_argument(
        "--environment-metrics",
        default="outputs/depmap_baselines/environment_baseline_metrics.csv",
    )
    parser.add_argument("--dqn-metrics", default="outputs/depmap_baselines/dqn_eval_metrics.csv")
    parser.add_argument("--output-dir", default="outputs/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures = generate_baseline_figures(
        data_metrics_path=args.data_metrics,
        environment_metrics_path=args.environment_metrics,
        dqn_metrics_path=args.dqn_metrics,
        output_dir=args.output_dir,
    )
    for figure in figures:
        print(figure)


if __name__ == "__main__":
    main()
