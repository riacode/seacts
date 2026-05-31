#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.baselines import AverageModalityPolicy, ModalityScorePolicy, OraclePolicy, RandomPolicy
from src.baselines import evaluate_policy
from src.config import load_baseline_config
from src.data import load_project_data
from src.episodes import EpisodeBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run candidate-selection baselines.")
    parser.add_argument("--config", default="configs/depmap_baselines.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_baseline_config(args.config)
    data = load_project_data(
        dependency_path=config.data.dependency_path,
        modality_paths=config.data.modalities,
        metadata_path=config.data.metadata_path,
    )

    builder = EpisodeBuilder(
        dependency=data.dependency,
        dependency_threshold=config.episodes.dependency_threshold,
        candidates_per_episode=config.episodes.candidates_per_episode,
        positives_per_episode=config.episodes.positives_per_episode,
        min_candidates_per_cell_line=config.episodes.min_candidates_per_cell_line,
        seed=config.seed,
    )
    episodes = builder.build(config.episodes.n_episodes)

    policies = [
        RandomPolicy(seed=config.seed),
        OraclePolicy(),
    ]
    policies.extend(
        ModalityScorePolicy(name, modality, query_cost=1.0)
        for name, modality in data.modalities.items()
    )
    if data.modalities:
        policies.append(
            AverageModalityPolicy(
                modalities=data.modalities,
                query_cost=config.evaluation.full_query_cost,
            )
        )

    rows = [evaluate_policy(policy, episodes, top_k=config.evaluation.top_k) for policy in policies]
    results = pd.DataFrame(rows).sort_values("selected_dependency")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline_metrics.csv"
    results.to_csv(output_path, index=False)

    print(results.to_string(index=False))
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
