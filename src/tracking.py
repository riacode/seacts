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
        name="depmap-baselines",
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
        "output_dir": str(config.output_dir),
    }


def _wandb_table(results: pd.DataFrame) -> Any:
    import wandb

    return wandb.Table(dataframe=results)
