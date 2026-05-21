from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_baseline_runner import _build_episodes, _resolve_data_path, _resolve_optional_data_path
from src.config import load_baseline_config
from src.data import load_project_data
from src.environment import EvidenceAcquisitionEnv
from src.environment_baselines import build_environment_policies, evaluate_environment_policy
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
    episodes = _build_episodes(config, data.dependency)
    env = EvidenceAcquisitionEnv(data.modalities)
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
