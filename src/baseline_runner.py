from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.baselines import AverageModalityPolicy, ModalityScorePolicy, OraclePolicy, RandomPolicy
from src.baselines import evaluate_policy
from src.config import BaselineConfig, load_baseline_config
from src.data import load_project_data
from src.episodes import EpisodeBuilder
from src.tracking import log_baseline_results, wandb_baseline_run


def run_baseline_pipeline(
    config_path: str | Path,
    raw_data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    config = load_baseline_config(config_path)
    dependency_path = _resolve_data_path(config.data.dependency_path, raw_data_dir)
    metadata_path = _resolve_optional_data_path(config.data.metadata_path, raw_data_dir)
    modality_paths = {
        name: _resolve_data_path(path, raw_data_dir)
        for name, path in config.data.modalities.items()
    }

    data = load_project_data(
        dependency_path=dependency_path,
        modality_paths=modality_paths,
        metadata_path=metadata_path,
    )
    episodes = _build_episodes(config, data.dependency)
    policies = _build_policies(config, data.modalities)

    resolved_output_dir = Path(output_dir) if output_dir is not None else Path(config.output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = resolved_output_dir / "baseline_metrics.csv"

    with wandb_baseline_run(config, config_path) as wandb_run:
        rows = [evaluate_policy(policy, episodes, top_k=config.evaluation.top_k) for policy in policies]
        results = pd.DataFrame(rows).sort_values("selected_dependency")
        results.to_csv(output_path, index=False)
        log_baseline_results(wandb_run, results, output_path)

    return results, output_path


def _resolve_optional_data_path(path: Path | None, raw_data_dir: str | Path | None) -> Path | None:
    if path is None:
        return None
    return _resolve_data_path(path, raw_data_dir)


def _resolve_data_path(path: Path, raw_data_dir: str | Path | None) -> Path:
    if raw_data_dir is None:
        return path
    return Path(raw_data_dir) / path.name


def _build_episodes(config: BaselineConfig, dependency: pd.DataFrame):
    builder = EpisodeBuilder(
        dependency=dependency,
        dependency_threshold=config.episodes.dependency_threshold,
        candidates_per_episode=config.episodes.candidates_per_episode,
        positives_per_episode=config.episodes.positives_per_episode,
        min_candidates_per_cell_line=config.episodes.min_candidates_per_cell_line,
        seed=config.seed,
    )
    return builder.build(config.episodes.n_episodes)


def _build_policies(config: BaselineConfig, modalities: dict[str, pd.DataFrame]):
    policies = [
        RandomPolicy(seed=config.seed),
        OraclePolicy(),
    ]
    policies.extend(
        ModalityScorePolicy(name, modality, query_cost=1.0)
        for name, modality in modalities.items()
    )
    if modalities:
        policies.append(
            AverageModalityPolicy(
                modalities=modalities,
                query_cost=config.evaluation.full_query_cost,
            )
        )
    return policies
