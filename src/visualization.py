from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Any

import pandas as pd

from src.behavior_analysis import summarize_query_efficiency


METRIC_COLUMNS = ("hit_at_k", "ndcg_at_k", "mrr_at_k")
DQN_POLICY = "rl_env_dqn"

# Poster subset (7): bounds, budget baselines, strong fixed policy, learned, oracle.
POSTER_POLICY_ORDER = (
    "rl_env_random_select",
    "rl_env_query_expression_budget_8_then_select",
    "rl_env_query_expression_budget_12_then_select",
    "rl_env_query_expression_then_select",
    "rl_env_query_all_average_then_select",
    DQN_POLICY,
    "rl_env_oracle_select",
)

POLICY_LABELS = {
    "rl_env_oracle_select": "Oracle",
    "rl_env_random_select": "Random",
    "rl_env_dqn": "DQN",
    "rl_env_query_all_average_then_select": "All avg",
    "rl_env_query_expression_then_select": "Expr full",
    "rl_env_query_cna_then_select": "CNA full",
    "rl_env_query_damaging_mutation_then_select": "Damage full",
    "rl_env_query_hotspot_mutation_then_select": "Hotspot full",
    "rl_env_query_expression_budget_4_then_select": "Expr budget 4",
    "rl_env_query_expression_budget_8_then_select": "Expr budget 8",
    "rl_env_query_expression_budget_12_then_select": "Expr budget 12",
    "data_oracle_dependency": "Data oracle",
    "data_random_select": "Data random",
    "data_average_all_modalities": "Data all avg",
}

DQN_COLOR = "#C0392B"
DQN_EDGE_WIDTH = 2.5

ABLATION_VARIANT_ORDER = (
    "scale1_long_mlp_1step",
    "scale1_long_mlp_3step",
    "scale1_long_structured_1step",
    "scale1_long_structured_dueling_1step",
    "scale1_long_structured_3step",
    "scale1_long_structured_dueling_3step",
)

ABLATION_LABELS = {
    "scale1_long_mlp_1step": "MLP\n1-step",
    "scale1_long_mlp_3step": "MLP\n3-step",
    "scale1_long_structured_1step": "Structured\n1-step",
    "scale1_long_structured_dueling_1step": "Structured dueling\n1-step",
    "scale1_long_structured_3step": "Structured\n3-step",
    "scale1_long_structured_dueling_3step": "Structured dueling\n3-step",
}


def generate_baseline_figures(
    data_metrics_path: str | Path | None,
    environment_metrics_path: str | Path | None,
    output_dir: str | Path,
    dqn_metrics_path: str | Path | None = None,
    dqn_trajectory_path: str | Path | None = None,
    poster_subset: bool = True,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data_results = _read_metrics(data_metrics_path)
    environment_results = _read_metrics(environment_metrics_path)
    dqn_results = _read_metrics(dqn_metrics_path)
    dqn_trajectories = _read_metrics(dqn_trajectory_path)

    environment_comparison = _combine_environment_results(environment_results, dqn_results)
    if environment_comparison is not None and poster_subset:
        environment_comparison = _prepare_poster_frame(environment_comparison)

    figures: list[Path] = []
    if environment_comparison is not None:
        figures.append(_plot_cost_vs_target_reward(environment_comparison, output_path))
        figures.append(_plot_total_reward(environment_comparison, output_path))
        figures.append(_plot_query_count_vs_hit_rate(environment_comparison, output_path))

    combined = _combine_results(data_results, environment_results, dqn_results)
    if combined is not None:
        if poster_subset:
            combined = _prepare_poster_frame(combined)
        figures.append(_plot_ranking_metrics(combined, output_path))
    if dqn_trajectories is not None:
        figures.append(_plot_dqn_query_count_distribution(dqn_trajectories, output_path))
        figures.append(_plot_dqn_modality_usage(dqn_trajectories, output_path))

    if not figures:
        raise FileNotFoundError("No baseline metric CSV files were found to visualize.")
    return figures


def generate_ablation_figures(
    ablation_summary_path: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ablation_summary_path)
    frame = _prepare_ablation_frame(frame)
    return [
        _plot_ablation_total_reward(frame, output_path),
        _plot_ablation_quality_vs_queries(frame, output_path),
        _plot_ablation_modality_usage(frame, output_path),
    ]


def generate_behavior_figures(
    episode_summary_path: str | Path,
    step_log_path: str | Path,
    output_dir: str | Path,
    context_usage_path: str | Path | None = None,
    max_trajectory_episodes: int = 12,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    episodes = pd.read_csv(episode_summary_path)
    steps = pd.read_csv(step_log_path)

    figures = [
        _plot_regret_vs_queries(episodes, output_path),
        _plot_query_efficiency(summarize_query_efficiency(steps), output_path),
        _plot_trajectory_strip(episodes, steps, output_path, max_trajectory_episodes),
    ]
    context_usage = _read_metrics(context_usage_path)
    if context_usage is not None:
        figures.append(_plot_modality_usage_by_context(context_usage, output_path))
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


def _prepare_poster_frame(results: pd.DataFrame) -> pd.DataFrame:
    frame = results[results["policy"].isin(POSTER_POLICY_ORDER)].copy()
    order = {policy: index for index, policy in enumerate(POSTER_POLICY_ORDER)}
    frame["_sort"] = frame["policy"].map(order)
    return frame.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)


def _prepare_ablation_frame(results: pd.DataFrame) -> pd.DataFrame:
    if "variant" not in results.columns:
        raise ValueError("Ablation summary must include a 'variant' column.")
    frame = results[results["variant"].isin(ABLATION_VARIANT_ORDER)].copy()
    if frame.empty:
        raise ValueError("No expected DQN ablation variants were found.")
    order = {variant: index for index, variant in enumerate(ABLATION_VARIANT_ORDER)}
    frame["_sort"] = frame["variant"].map(order)
    return frame.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)


def _combine_results(
    data_results: pd.DataFrame | None,
    environment_results: pd.DataFrame | None,
    dqn_results: pd.DataFrame | None,
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
    if dqn_results is not None:
        dqn = dqn_results.copy()
        dqn["baseline_family"] = "rl"
        frames.append(dqn)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _combine_environment_results(
    environment_results: pd.DataFrame | None,
    dqn_results: pd.DataFrame | None,
) -> pd.DataFrame | None:
    frames = [frame.copy() for frame in (environment_results, dqn_results) if frame is not None]
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _plot_cost_vs_target_reward(results: pd.DataFrame, output_dir: Path) -> Path:
    plt = _pyplot()
    frame = results.copy()
    frame["target_reward"] = -frame["selected_dependency"]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for index, row in enumerate(frame.itertuples(index=False)):
        is_dqn = str(row.policy) == DQN_POLICY
        ax.scatter(
            row.query_cost,
            row.target_reward,
            s=220 if is_dqn else 90,
            c=DQN_COLOR if is_dqn else "#4C72B0",
            edgecolors="black",
            linewidths=DQN_EDGE_WIDTH if is_dqn else 0.8,
            zorder=3 if is_dqn else 2,
        )
        y_offset = 7 if index % 2 == 0 else -12
        ax.annotate(
            _short_policy_name(str(row.policy)),
            (float(row.query_cost), float(row.target_reward)),
            xytext=(7, y_offset),
            textcoords="offset points",
            fontsize=8 if is_dqn else 7,
            fontweight="bold" if is_dqn else "normal",
        )
    ax.set_xlabel("Query cost")
    ax.set_ylabel("Target reward (-selected dependency)")
    ax.set_title("Cost vs Target Quality")
    ax.grid(alpha=0.25)
    return _save(fig, output_dir / "environment_cost_vs_target_reward.png")


def _plot_total_reward(results: pd.DataFrame, output_dir: Path) -> Path:
    plt = _pyplot()
    frame = results.iloc[::-1].reset_index(drop=True)

    fig_height = max(5.5, len(frame) * 0.55)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    labels = [_short_policy_name(policy, max_width=14) for policy in frame["policy"]]
    colors = [DQN_COLOR if policy == DQN_POLICY else "#4C72B0" for policy in frame["policy"]]
    edge_widths = [DQN_EDGE_WIDTH if policy == DQN_POLICY else 0.8 for policy in frame["policy"]]
    bars = ax.barh(labels, frame["total_reward"], color=colors, edgecolor="black")
    for bar, width in zip(bars, edge_widths, strict=True):
        bar.set_linewidth(width)
    for index, policy in enumerate(frame["policy"]):
        if policy == DQN_POLICY:
            ax.get_yticklabels()[index].set_fontweight("bold")
            ax.get_yticklabels()[index].set_color(DQN_COLOR)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Total episode reward (target quality − query cost)")
    ax.set_title("Total Reward After Query Costs")
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, output_dir / "environment_total_reward.png")


def _plot_query_count_vs_hit_rate(results: pd.DataFrame, output_dir: Path) -> Path:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for index, row in enumerate(results.itertuples(index=False)):
        is_dqn = str(row.policy) == DQN_POLICY
        ax.scatter(
            row.n_queries,
            row.hit_at_k,
            s=220 if is_dqn else 90,
            c=DQN_COLOR if is_dqn else "#4C72B0",
            edgecolors="black",
            linewidths=DQN_EDGE_WIDTH if is_dqn else 0.8,
            zorder=3 if is_dqn else 2,
        )
        y_offset = 7 if index % 2 == 0 else -12
        ax.annotate(
            _short_policy_name(str(row.policy)),
            (float(row.n_queries), float(row.hit_at_k)),
            xytext=(7, y_offset),
            textcoords="offset points",
            fontsize=8 if is_dqn else 7,
            fontweight="bold" if is_dqn else "normal",
        )
    ax.set_xlabel("Number of evidence queries")
    ax.set_ylabel("Hit@k")
    ax.set_title("Query Count vs Hit Rate")
    ax.grid(alpha=0.25)
    return _save(fig, output_dir / "environment_queries_vs_hit_rate.png")


def _plot_ablation_total_reward(results: pd.DataFrame, output_dir: Path) -> Path:
    plt = _pyplot()
    frame = results.sort_values("total_reward", ascending=True).reset_index(drop=True)
    labels = [ABLATION_LABELS.get(variant, variant) for variant in frame["variant"]]
    colors = ["#4C72B0" if "mlp" in variant else DQN_COLOR for variant in frame["variant"]]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.barh(labels, frame["total_reward"], color=colors, edgecolor="black", linewidth=0.8)
    best_index = int(frame["total_reward"].idxmax())
    bars[best_index].set_linewidth(2.4)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Total episode reward")
    ax.set_title("DQN Architecture Ablation")
    ax.grid(axis="x", alpha=0.25)
    for index, value in enumerate(frame["total_reward"]):
        ax.text(float(value) + 0.01, index, f"{float(value):.3f}", va="center", fontsize=8)
    return _save(fig, output_dir / "dqn_ablation_total_reward.png")


def _plot_ablation_quality_vs_queries(results: pd.DataFrame, output_dir: Path) -> Path:
    plt = _pyplot()
    frame = results.copy()
    frame["target_reward"] = -frame["selected_dependency"]

    fig, ax = plt.subplots(figsize=(9, 6))
    for row in frame.itertuples(index=False):
        variant = str(row.variant)
        is_mlp = "mlp" in variant
        ax.scatter(
            row.n_queries,
            row.target_reward,
            s=120 if is_mlp else 220,
            c="#4C72B0" if is_mlp else DQN_COLOR,
            edgecolors="black",
            linewidths=0.9,
        )
        ax.annotate(
            ABLATION_LABELS.get(variant, variant).replace("\n", " "),
            (float(row.n_queries), float(row.target_reward)),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Mean evidence queries")
    ax.set_ylabel("Target quality (-selected dependency)")
    ax.set_title("Ablation Query-Quality Tradeoff")
    ax.grid(alpha=0.25)
    return _save(fig, output_dir / "dqn_ablation_quality_vs_queries.png")


def _plot_ablation_modality_usage(results: pd.DataFrame, output_dir: Path) -> Path:
    plt = _pyplot()
    modality_columns = [column for column in results.columns if column.startswith("n_query_")]
    frame = results.set_index("variant")[modality_columns].copy()
    frame.index = [ABLATION_LABELS.get(variant, variant).replace("\n", " ") for variant in frame.index]
    frame.columns = [column.removeprefix("n_query_").replace("_", " ") for column in frame.columns]

    fig, ax = plt.subplots(figsize=(11, 6))
    frame.plot(kind="bar", stacked=True, ax=ax, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("")
    ax.set_ylabel("Mean queries per episode")
    ax.set_title("Ablation Modality Usage")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Modality", loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, output_dir / "dqn_ablation_modality_usage.png")


def _plot_ranking_metrics(results: pd.DataFrame, output_dir: Path) -> Path:
    plt = _pyplot()
    frame = results.copy()
    metric_colors = ("#4C72B0", "#DD8452", "#55A868")
    policies = [_short_policy_name(policy, max_width=12) for policy in frame["policy"]]
    x_positions = list(range(len(frame)))
    width = 0.24

    fig_width = max(10.0, len(frame) * 1.1)
    fig, ax = plt.subplots(figsize=(fig_width, 7.0))
    for offset, metric in enumerate(METRIC_COLUMNS):
        xs = [x + (offset - 1) * width for x in x_positions]
        bar_colors = []
        edge_colors = []
        edge_widths = []
        for policy in frame["policy"]:
            if policy == DQN_POLICY:
                bar_colors.append(metric_colors[offset])
                edge_colors.append("black")
                edge_widths.append(DQN_EDGE_WIDTH)
            else:
                bar_colors.append(metric_colors[offset])
                edge_colors.append("black")
                edge_widths.append(0.6)
        bars = ax.bar(xs, frame[metric], width=width, label=metric, color=bar_colors, edgecolor=edge_colors)
        for bar, edge_width in zip(bars, edge_widths, strict=True):
            bar.set_linewidth(edge_width)

    ax.set_xticks(x_positions)
    tick_labels = ax.set_xticklabels(policies, rotation=35, ha="right", fontsize=8)
    for index, policy in enumerate(frame["policy"]):
        if policy == DQN_POLICY:
            tick_labels[index].set_fontweight("bold")
            tick_labels[index].set_color(DQN_COLOR)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Metric value")
    ax.set_title("Ranking Metrics")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, output_dir / "baseline_ranking_metrics.png")


def _plot_dqn_query_count_distribution(results: pd.DataFrame, output_dir: Path) -> Path:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    max_queries = int(results["n_queries"].max()) if not results.empty else 0
    bins = range(0, max_queries + 2)
    ax.hist(
        results["n_queries"],
        bins=bins,
        align="left",
        edgecolor="black",
        color=DQN_COLOR,
        linewidth=1.2,
    )
    ax.set_xlabel("Number of evidence queries")
    ax.set_ylabel("Episodes")
    ax.set_title("DQN Query Count Distribution")
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, output_dir / "dqn_query_count_distribution.png")


def _plot_dqn_modality_usage(results: pd.DataFrame, output_dir: Path) -> Path:
    plt = _pyplot()
    modality_columns = [column for column in results.columns if column.startswith("n_query_")]
    usage = results[modality_columns].mean().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = [column.removeprefix("n_query_").replace("_", "\n") for column in usage.index]
    ax.barh(labels, usage.values, color=DQN_COLOR, edgecolor="black", linewidth=1.0)
    ax.set_xlabel("Mean queries per episode")
    ax.set_title("DQN Modality Usage")
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, output_dir / "dqn_modality_usage.png")


def _plot_regret_vs_queries(episodes: pd.DataFrame, output_dir: Path) -> Path:
    plt = _pyplot()
    frame = episodes.copy()
    hit = frame["hit_at_k"].astype(float) >= 1.0 if "hit_at_k" in frame else pd.Series(True, index=frame.index)

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.scatter(
        frame.loc[hit, "n_queries"],
        frame.loc[hit, "dependency_regret"],
        s=55,
        c="#4C72B0",
        edgecolors="black",
        linewidths=0.5,
        alpha=0.75,
        label="Hit@k",
    )
    ax.scatter(
        frame.loc[~hit, "n_queries"],
        frame.loc[~hit, "dependency_regret"],
        s=75,
        c=DQN_COLOR,
        edgecolors="black",
        linewidths=0.8,
        alpha=0.9,
        label="Miss",
    )
    ax.set_xlabel("Number of evidence queries")
    ax.set_ylabel("Dependency regret")
    ax.set_title("DQN Regret vs Query Count")
    ax.legend()
    ax.grid(alpha=0.25)
    return _save(fig, output_dir / "dqn_regret_vs_queries.png")


def _plot_query_efficiency(query_efficiency: pd.DataFrame, output_dir: Path) -> Path:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(
        query_efficiency["gene_true_rank"].astype(str),
        query_efficiency["query_fraction"],
        color=DQN_COLOR,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.axhline(1.0 / max(len(query_efficiency), 1), color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("True dependency rank of queried gene")
    ax.set_ylabel("Fraction of DQN query actions")
    ax.set_title("DQN Query Efficiency by True Dependency Rank")
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, output_dir / "dqn_query_efficiency_by_true_rank.png")


def _plot_modality_usage_by_context(results: pd.DataFrame, output_dir: Path) -> Path:
    plt = _pyplot()
    modality_columns = [column for column in results.columns if column.startswith("n_query_")]
    context_columns = [
        column
        for column in results.columns
        if column not in {*modality_columns, "n_episodes"}
    ]
    if not modality_columns or not context_columns:
        raise ValueError("Context modality usage table must include context and n_query_* columns.")

    context_column = context_columns[0]
    frame = results.sort_values("n_episodes", ascending=False).head(15).copy()
    values = frame[modality_columns].astype(float)
    modality_labels = [column.removeprefix("n_query_").replace("_", "\n") for column in modality_columns]
    counts = frame["n_episodes"] if "n_episodes" in frame else pd.Series(0, index=frame.index)
    context_labels = [
        f"{context} (n={int(n)})"
        for context, n in zip(frame[context_column], counts, strict=False)
    ]

    fig_height = max(5.0, len(frame) * 0.42)
    fig, ax = plt.subplots(figsize=(9.5, fig_height))
    image = ax.imshow(values.to_numpy(), aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(modality_labels)))
    ax.set_xticklabels(modality_labels, fontsize=8)
    ax.set_yticks(range(len(context_labels)))
    ax.set_yticklabels(context_labels, fontsize=8)
    ax.set_xlabel("Modality")
    ax.set_title("DQN Modality Usage by Cancer Context")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Mean queries per episode")

    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            ax.text(
                col_index,
                row_index,
                f"{values.iat[row_index, col_index]:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )
    return _save(fig, output_dir / "dqn_modality_usage_by_context.png")


def _plot_trajectory_strip(
    episodes: pd.DataFrame,
    steps: pd.DataFrame,
    output_dir: Path,
    max_episodes: int,
) -> Path:
    plt = _pyplot()
    candidates = _trajectory_episode_ids(episodes, max_episodes)
    frame = steps[steps["episode_id"].isin(candidates)].copy()
    if frame.empty:
        raise ValueError("No matching step rows were found for trajectory plotting.")

    modalities = sorted(value for value in frame["modality"].dropna().unique())
    color_map = {
        modality: color
        for modality, color in zip(
            modalities,
            ["#4C72B0", "#55A868", "#DD8452", "#8172B3", "#937860", "#64B5CD"],
            strict=False,
        )
    }
    color_map["select"] = DQN_COLOR

    fig_height = max(4.5, len(candidates) * 0.45)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    label_by_episode = _trajectory_labels(episodes)
    y_positions = {episode_id: index for index, episode_id in enumerate(candidates)}
    for row in frame.itertuples(index=False):
        action_type = str(row.action_type)
        color_key = "select" if action_type == "select" else str(row.modality)
        ax.broken_barh(
            [(float(row.step), 0.85)],
            (y_positions[int(row.episode_id)] - 0.35, 0.7),
            facecolors=color_map.get(color_key, "#999999"),
            edgecolors="black",
            linewidth=0.4,
        )

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([label_by_episode.get(episode_id, str(episode_id)) for episode_id in candidates], fontsize=8)
    ax.set_xlabel("DQN step")
    ax.set_title("Example DQN Evidence-Acquisition Trajectories")
    handles = [
        plt.Line2D([0], [0], color=color, linewidth=8, label=label)
        for label, color in color_map.items()
    ]
    ax.legend(handles=handles, loc="upper right", ncols=min(3, len(handles)))
    ax.grid(axis="x", alpha=0.2)
    return _save(fig, output_dir / "dqn_example_trajectories.png")


def _trajectory_episode_ids(episodes: pd.DataFrame, max_episodes: int) -> list[int]:
    frame = episodes.copy()
    if "dependency_regret" not in frame:
        return [int(value) for value in frame["episode_id"].head(max_episodes)]

    successes = frame[frame["dependency_regret"] == 0.0].sort_values("n_queries").head(max_episodes // 2)
    failures = (
        frame[~frame["episode_id"].isin(successes["episode_id"])]
        .sort_values("dependency_regret", ascending=False)
        .head(max_episodes - len(successes))
    )
    selected = pd.concat([successes, failures], ignore_index=True)
    return [int(value) for value in selected["episode_id"]]


def _trajectory_labels(episodes: pd.DataFrame) -> dict[int, str]:
    labels = {}
    for row in episodes.itertuples(index=False):
        regret = getattr(row, "dependency_regret", None)
        selected_gene = getattr(row, "selected_gene", "")
        labels[int(row.episode_id)] = (
            f"{int(row.episode_id)} {selected_gene} r={float(regret):.2f}"
            if regret is not None
            else f"{int(row.episode_id)} {selected_gene}"
        )
    return labels


def _save(fig: Any, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    _pyplot().close(fig)
    return path


def _short_policy_name(policy: str, max_width: int = 16) -> str:
    if policy in POLICY_LABELS:
        return "\n".join(textwrap.wrap(POLICY_LABELS[policy], width=max_width))
    label = (
        policy.removeprefix("rl_env_")
        .removeprefix("data_")
        .replace("_then_select", "")
        .replace("_", " ")
    )
    return "\n".join(textwrap.wrap(label, width=max_width))


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
