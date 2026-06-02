from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _wandb_api() -> Any:
    import wandb

    return wandb.Api()


def find_wandb_run(
    entity: str,
    project: str,
    name_substrings: tuple[str, ...],
) -> Any | None:
    """Return the newest run whose name contains any of the given substrings."""
    api = _wandb_api()
    for run in api.runs(f"{entity}/{project}", order="-created_at"):
        label = run.name or ""
        if any(substring in label for substring in name_substrings):
            return run
    return None


def eval_row_from_wandb_summary(run: Any, policy: str) -> dict[str, float | int | str] | None:
    """Build a dqn_eval_metrics row from run.summary eval/* keys."""
    row: dict[str, float | int | str] = {"policy": policy}
    for key, value in dict(run.summary).items():
        if not str(key).startswith("eval/"):
            continue
        metric_key = str(key).removeprefix("eval/")
        if hasattr(value, "item"):
            value = value.item()
        row[metric_key] = value
    if "total_reward" not in row:
        return None
    return row


def load_dqn_eval_metrics_from_wandb(
    entity: str,
    project: str,
    run_name_substrings: tuple[str, ...],
    policy: str = "rl_env_dqn",
) -> pd.DataFrame | None:
    run = find_wandb_run(entity, project, run_name_substrings)
    if run is None:
        return None
    row = eval_row_from_wandb_summary(run, policy)
    if row is None:
        return None
    return pd.DataFrame([row])


def cache_eval_metrics(frame: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
