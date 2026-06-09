from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_baseline_runner import _build_episodes, _resolve_data_path, _resolve_optional_data_path
from src.config import load_baseline_config
from src.data import load_project_data
from src.environment import EvidenceAcquisitionEnv
from src.environment_baselines import build_environment_policies, evaluate_environment_policy
from src.modality_scores import build_supervised_modality_scores
from src.splits import load_cell_line_split_config, maybe_split_dependency_by_cell_line
from src.tracking import log_baseline_results, wandb_baseline_run


def run_environment_baseline_pipeline(
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
    split_config = load_cell_line_split_config(config_path)
    train_dependency, _, eval_dependency = maybe_split_dependency_by_cell_line(
        config,
        data.dependency,
        split_config,
    )
    episodes = _build_episodes(config, eval_dependency)
    modalities = (
        build_supervised_modality_scores(
            data.dependency,
            data.modalities,
            train_cell_lines=set(train_dependency.index.astype(str)),
        )
        if config.environment.use_supervised_modality_scores
        else data.modalities
    )
    env = EvidenceAcquisitionEnv(
        modalities,
        query_costs=config.environment.query_costs,
        repeated_query_penalty=config.environment.repeated_query_penalty,
        selection_reward_scale=config.environment.selection_reward_scale,
    )
    policies = build_environment_policies(env.modality_names, seed=config.seed)

    resolved_output_dir = Path(output_dir) if output_dir is not None else Path(config.output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = resolved_output_dir / "environment_baseline_metrics.csv"

    with wandb_baseline_run(config, config_path, run_name="environment-baselines") as wandb_run:
        rows = [
            evaluate_environment_policy(policy, env, episodes, top_k=config.evaluation.top_k)
            for policy in policies
        ]
        results = pd.DataFrame(rows).sort_values("total_reward", ascending=False)
        results.to_csv(output_path, index=False)
        log_baseline_results(wandb_run, results, output_path)

    return results, output_path
