#!/usr/bin/env python3
from __future__ import annotations

import argparse

from src.behavior_analysis import write_behavior_analysis_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize saved DQN behavior logs.")
    parser.add_argument(
        "--episodes",
        default="outputs/depmap_baselines/dqn_episode_summary.csv",
        help="Path to dqn_episode_summary.csv.",
    )
    parser.add_argument(
        "--steps",
        default="outputs/depmap_baselines/dqn_step_log.csv",
        help="Path to dqn_step_log.csv.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional Model.csv path for cancer-context summaries.",
    )
    parser.add_argument(
        "--context-column",
        default=None,
        help="Optional metadata column for context grouping, such as OncotreeLineage.",
    )
    parser.add_argument("--output-dir", default="outputs/depmap_baselines/behavior_analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = write_behavior_analysis_tables(
        episode_summary_path=args.episodes,
        step_log_path=args.steps,
        metadata_path=args.metadata,
        context_column=args.context_column,
        output_dir=args.output_dir,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
