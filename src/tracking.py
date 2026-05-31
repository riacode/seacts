from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import BaselineConfig


@contextmanager
def wandb_baseline_run(
    config: BaselineConfig,
    config_path: str | Path,
    run_name: str = "data-baselines",
) -> Iterator[Any | None]:
    if not config.tracking.wandb.enabled:
        yield None
        return

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B tracking is enabled, but wandb is not installed. "
            "Install project dependencies or disable tracking.wandb.enabled."
        ) from error

    with wandb.init(
        entity=config.tracking.wandb.entity,
        project=config.tracking.wandb.project,
        name=run_name,
        job_type="baseline",
        config=_wandb_config(config, config_path),
    ) as run:
        yield run


def log_baseline_results(run: Any | None, results: pd.DataFrame, output_path: Path) -> None:
    if run is None:
        return

    metrics: dict[str, float | str] = {"output_path": str(output_path)}
    for row in results.to_dict(orient="records"):
        policy = row["policy"]
        for key, value in row.items():
            if key == "policy":
                continue
            metrics[f"{policy}/{key}"] = value

    run.log(metrics)
    run.log({"baseline_metrics": _wandb_table(results)})


def _wandb_config(config: BaselineConfig, config_path: str | Path) -> dict[str, Any]:
    return {
        "config_path": str(config_path),
        "seed": config.seed,
        "data": {
            "dependency_path": str(config.data.dependency_path),
            "metadata_path": str(config.data.metadata_path) if config.data.metadata_path else None,
            "modalities": {
                name: str(path) for name, path in config.data.modalities.items()
            },
        },
        "episodes": {
            "n_episodes": config.episodes.n_episodes,
            "candidates_per_episode": config.episodes.candidates_per_episode,
            "positives_per_episode": config.episodes.positives_per_episode,
            "dependency_threshold": config.episodes.dependency_threshold,
            "min_candidates_per_cell_line": config.episodes.min_candidates_per_cell_line,
        },
        "evaluation": {
            "top_k": config.evaluation.top_k,
            "full_query_cost": config.evaluation.full_query_cost,
        },
        "environment": {
            "query_costs": config.environment.query_costs,
            "repeated_query_penalty": config.environment.repeated_query_penalty,
        },
        "output_dir": str(config.output_dir),
    }


def _wandb_table(results: pd.DataFrame) -> Any:
    import wandb

    return wandb.Table(dataframe=results)


def log_dqn_behavior_results(
    run: Any | None,
    episodes_df: pd.DataFrame,
    steps_df: pd.DataFrame,
) -> None:
    if run is None:
        return

    import wandb

    run.log(
        {
            "dqn_episode_summary": wandb.Table(dataframe=episodes_df),
            "dqn_step_log": wandb.Table(dataframe=steps_df),
            "behavior/hit_rate": float(episodes_df["hit_at_k"].mean()),
            "behavior/mean_n_queries": float(episodes_df["n_queries"].mean()),
            "behavior/mean_dependency_regret": float(episodes_df["dependency_regret"].mean()),
            "behavior/query_count_histogram": wandb.Histogram(episodes_df["n_queries"].tolist()),
            "behavior/dependency_regret_histogram": wandb.Histogram(
                episodes_df["dependency_regret"].tolist()
            ),
        }
    )

    modality_columns = [column for column in episodes_df.columns if column.startswith("n_query_")]
    if modality_columns:
        modality_rows = [
            [column.removeprefix("n_query_"), float(episodes_df[column].mean())]
            for column in modality_columns
        ]
        modality_table = wandb.Table(data=modality_rows, columns=["modality", "mean_queries"])
        run.log(
            {
                "behavior/modality_usage": wandb.plot.bar(
                    modality_table,
                    "modality",
                    "mean_queries",
                    title="Mean queries per episode by modality",
                )
            }
        )

    query_steps = steps_df[steps_df["action_type"] == "query"]
    if not query_steps.empty and "gene_true_rank" in query_steps.columns:
        run.log(
            {
                "behavior/queried_gene_rank_histogram": wandb.Histogram(
                    query_steps["gene_true_rank"].tolist()
                ),
                "behavior/queries_by_step": wandb.plot.histogram(
                    wandb.Table(dataframe=query_steps[["step", "gene_true_rank"]]),
                    "step",
                    title="Query actions by step",
                ),
            }
        )

    run.summary["behavior/hit_rate"] = float(episodes_df["hit_at_k"].mean())
    run.summary["behavior/mean_n_queries"] = float(episodes_df["n_queries"].mean())
    run.summary["behavior/mean_dependency_regret"] = float(
        episodes_df["dependency_regret"].mean()
    )
