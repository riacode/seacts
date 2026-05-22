from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


METRIC_COLUMNS = ("hit_at_k", "ndcg_at_k", "mrr_at_k")


def generate_baseline_figures(
    data_metrics_path: str | Path | None,
    environment_metrics_path: str | Path | None,
    output_dir: str | Path,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data_results = _read_metrics(data_metrics_path)
    environment_results = _read_metrics(environment_metrics_path)

    figures: list[Path] = []
    if environment_results is not None:
        figures.append(_plot_cost_vs_target_reward(environment_results, output_path))
        figures.append(_plot_total_reward(environment_results, output_path))
        figures.append(_plot_query_count_vs_hit_rate(environment_results, output_path))

    combined = _combine_results(data_results, environment_results)
    if combined is not None:
        figures.append(_plot_ranking_metrics(combined, output_path))

    if not figures:
        raise FileNotFoundError("No baseline metric CSV files were found to visualize.")
    return figures


def _read_metrics(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    metrics_path = Path(path)
    if not metrics_path.exists():
        return None
    frame = pd.read_csv(metrics_path)
    if frame.empty:
        return None
    return frame


def _combine_results(
    data_results: pd.DataFrame | None,
    environment_results: pd.DataFrame | None,
) -> pd.DataFrame | None:
    frames = []
    if data_results is not None:
        data = data_results.copy()
        data["baseline_family"] = "data"
        frames.append(data)
    if environment_results is not None:
        environment = environment_results.copy()
        environment["baseline_family"] = "rl_env"
        frames.append(environment)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _plot_cost_vs_target_reward(results: pd.DataFrame, output_dir: Path) -> Path:
    plt = _pyplot()
    frame = results.copy()
    frame["target_reward"] = -frame["selected_dependency"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(frame["query_cost"], frame["target_reward"], s=80)
    for row in frame.itertuples(index=False):
        ax.annotate(
            _short_policy_name(str(row.policy)),
            (float(row.query_cost), float(row.target_reward)),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Query cost")
    ax.set_ylabel("Target reward (-selected dependency)")
    ax.set_title("RL Environment Baselines: Cost vs Target Quality")
    ax.grid(alpha=0.25)
    return _save(fig, output_dir / "environment_cost_vs_target_reward.png")


def _plot_total_reward(results: pd.DataFrame, output_dir: Path) -> Path:
    plt = _pyplot()
    frame = results.sort_values("total_reward", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh([_short_policy_name(policy) for policy in frame["policy"]], frame["total_reward"])
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Total episode reward")
    ax.set_title("RL Environment Baselines: Reward After Query Costs")
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, output_dir / "environment_total_reward.png")


def _plot_query_count_vs_hit_rate(results: pd.DataFrame, output_dir: Path) -> Path:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(results["n_queries"], results["hit_at_k"], s=80)
    for row in results.itertuples(index=False):
        ax.annotate(
            _short_policy_name(str(row.policy)),
            (float(row.n_queries), float(row.hit_at_k)),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Number of evidence queries")
    ax.set_ylabel("Hit@k")
    ax.set_title("RL Environment Baselines: Query Count vs Hit Rate")
    ax.grid(alpha=0.25)
    return _save(fig, output_dir / "environment_queries_vs_hit_rate.png")


def _plot_ranking_metrics(results: pd.DataFrame, output_dir: Path) -> Path:
    plt = _pyplot()
    frame = results.copy()
    policies = [_short_policy_name(policy) for policy in frame["policy"]]
    x_positions = list(range(len(frame)))
    width = 0.24

    fig_width = max(10.0, len(frame) * 0.65)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    for offset, metric in enumerate(METRIC_COLUMNS):
        xs = [x + (offset - 1) * width for x in x_positions]
        ax.bar(xs, frame[metric], width=width, label=metric)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(policies, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Metric value")
    ax.set_title("Baseline Ranking Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, output_dir / "baseline_ranking_metrics.png")


def _save(fig: Any, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    _pyplot().close(fig)
    return path


def _short_policy_name(policy: str) -> str:
    return (
        policy.removeprefix("rl_env_")
        .removeprefix("data_")
        .replace("_then_select", "")
        .replace("_", "\n")
    )


def _pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Baseline visualization requires matplotlib. Install project dependencies first."
        ) from error
    return plt
