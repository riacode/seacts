from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Any

import pandas as pd

from src.behavior_analysis import summarize_query_efficiency


METRIC_COLUMNS = ("hit_at_k", "ndcg_at_k", "mrr_at_k")
DQN_POLICY = "rl_env_dqn"
CANCER_CONTEXT_DQN_POLICIES = {
    "default": "rl_env_dqn_context_128",
    "larger": "rl_env_dqn_context_256",
}
CONTEXT_DQN_POSTER_POLICY = CANCER_CONTEXT_DQN_POLICIES["larger"]
POSTER_DQN_POLICIES = frozenset({DQN_POLICY, CONTEXT_DQN_POSTER_POLICY})

# Poster/report subset: match the sequential comparison table in the README.
POSTER_POLICY_ORDER = (
    "rl_env_random_select",
    "rl_env_query_cna_budget_12_then_select",
    "rl_env_query_expression_then_select",
    "rl_env_query_cna_then_select",
    DQN_POLICY,
    CONTEXT_DQN_POSTER_POLICY,
    "rl_env_oracle_select",
)

POLICY_LABELS = {
    "rl_env_oracle_select": "Oracle",
    "rl_env_random_select": "Random",
    "rl_env_dqn": "Structured DQN",
    CONTEXT_DQN_POSTER_POLICY: "Context DQN",
    "rl_env_query_all_average_then_select": "All avg",
    "rl_env_query_expression_then_select": "Expr full",
    "rl_env_query_cna_then_select": "CNA full",
    "rl_env_query_damaging_mutation_then_select": "Damage full",
    "rl_env_query_hotspot_mutation_then_select": "Hotspot full",
    "rl_env_query_expression_budget_4_then_select": "Expr budget 4",
    "rl_env_query_expression_budget_8_then_select": "Expr budget 8",
    "rl_env_query_expression_budget_12_then_select": "Expr budget 12",
    "rl_env_query_cna_budget_4_then_select": "CNA budget 4",
    "rl_env_query_cna_budget_8_then_select": "CNA budget 8",
    "rl_env_query_cna_budget_12_then_select": "CNA budget 12",
    "data_oracle_dependency": "Data oracle",
    "data_random_select": "Data random",
    "data_average_all_modalities": "Data all avg",
}

DQN_COLOR = "#6A4C93"
BASELINE_COLOR = "#B7A6D6"
MLP_COLOR = "#C8C1D9"
SELECT_COLOR = "#3B2E58"
DQN_EDGE_WIDTH = 2.5
# Match poster_outputs/reward_graph.png at dpi=180: 5056 x 2656 px.
POSTER_BAR_FIGSIZE = (5056 / 180, 2656 / 180)
POSTER_BAR_DPI = 180
POSTER_TITLE_SIZE = 42
POSTER_SUPTITLE_SIZE = 46
POSTER_LABEL_SIZE = 34
POSTER_TICK_SIZE = 30
POSTER_ANNOTATION_SIZE = 28
POSTER_LEGEND_SIZE = 28

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

CANCER_CONTEXT_VARIANT_ORDER = ("larger",)

CANCER_CONTEXT_LABELS = {
    "larger": "Context DQN",
}

# Skip old expert/fast runs when summarizing context sweeps.
CONTEXT_SWEEP_EXCLUDE = frozenset(
    {
        "ctx_larger_expert1k",
        "ctx_larger_init_structured_expert",
        "struct_lineage_env_scores",  # structured control, not a context DQN variant
    }
)

CONTEXT_SWEEP_SHORT_LABELS = {
    "ctx_fusion_lineage_init_structured": "fusion+init",
    "ctx_fusion_lineage_init_frozen": "fusion+frozen",
    "ctx_fusion_lineage_shaping_only": "fusion+shape",
    "ctx_baseline_larger": "context from start\n1-step",
    "ctx_larger_ctx32": "context from start\nctx32",
    "ctx_larger_ctx64": "context from start\nctx64",
    "ctx_larger_dueling": "context from start\ndueling",
    "ctx_larger_lower_lr": "lower_lr",
    "ctx_larger_min_queries_6": "min_q6",
    "ctx_larger_init_structured": "context from start\n+ init",
    "ctx_select_only_init_structured": "SELECT only\n+ init",
    "ctx_select_only_init_frozen": "SELECT only\n+ frozen queries",
    "ctx_init_structured_30k": "init+30k",
    "ctx_init_structured_ft_lr2e5": "init+lr2e5",
    "struct_lineage_env_scores": "struct+lineage env",
    "ctx_lineage_env_init_structured": "lineage scores\n+ init",
    "ctx_larger_init_frozen": "init_frozen",
    "ctx_larger_init_structured_ctx32": "init+ctx32",
    "ctx_larger_init_structured_min_queries_6": "init+min_q6",
    "ctx_larger_init_structured_ctx32_min_queries_6": "init+ctx32+q6",
}

CONTEXT_SWEEP_POSTER_VARIANTS = (
    "ctx_select_only_init_frozen",
    "ctx_larger_init_structured",
    "ctx_larger_ctx32",
    "ctx_larger_ctx64",
    "ctx_baseline_larger",
    "ctx_larger_dueling",
    "ctx_lineage_env_init_structured",
    "ctx_fusion_lineage_init_structured",
)

MODALITY_COLORS = {
    "cna": "#4C72B0",
    "damaging_mutation": "#55A868",
    "expression": "#DD8452",
    "hotspot_mutation": "#C44E52",
    "select": SELECT_COLOR,
}

MODALITY_QUERY_COLUMNS = (
    "n_query_expression",
    "n_query_cna",
    "n_query_damaging_mutation",
    "n_query_hotspot_mutation",
)

# Poster structured baseline — sweep winner first, then ablation fallback.
# Partner/git layout omits the dqn_sweeps/ prefix (see outputs/depmap_baselines/).
STRUCTURED_DQN_RUN_DIRS: tuple[tuple[str, ...], ...] = (
    ("dqn_sweeps", "best_structured_1step_larger"),
    ("best_structured_1step_larger",),
    ("dqn_sweeps", "best_structured_1step_default"),
    ("dqn_ablation", "scale1_long_structured_1step"),
)


def load_structured_dqn_poster_metrics(
    results_root: str | Path,
) -> tuple[pd.DataFrame, Path | None]:
    """Load structured DQN eval metrics and optional trajectory path for poster plots."""
    root = Path(results_root)
    for parts in STRUCTURED_DQN_RUN_DIRS:
        run_dir = root.joinpath(*parts)
        eval_path = run_dir / "dqn_eval_metrics.csv"
        if not eval_path.exists():
            continue
        frame = pd.read_csv(eval_path).copy()
        frame["policy"] = DQN_POLICY
        trajectory_path = run_dir / "dqn_trajectory_metrics.csv"
        trajectory = trajectory_path if trajectory_path.exists() else None
        return frame, trajectory

    candidates = [str(root.joinpath(*parts) / "dqn_eval_metrics.csv") for parts in STRUCTURED_DQN_RUN_DIRS]
    raise FileNotFoundError(
        "Structured DQN eval metrics not found on the results volume. "
        f"Checked: {', '.join(candidates)}. "
        "Run modal_sweep_dqn.py (best_structured_1step_larger) or ensure ablation metrics exist."
    )


def append_cancer_context_dqn_metrics(
    dqn_results: pd.DataFrame,
    cancer_context_dir: str | Path,
    variants: tuple[str, ...] = ("larger",),
) -> pd.DataFrame:
    """Add cancer-context eval rows alongside structured DQN metrics (poster: larger only)."""
    extra_rows: list[dict[str, float | int | str]] = []
    root = Path(cancer_context_dir)
    for variant in variants:
        policy = CANCER_CONTEXT_DQN_POLICIES.get(variant)
        if policy is None:
            continue
        eval_path = root / variant / "dqn_eval_metrics.csv"
        if not eval_path.exists():
            nested = root / "dqn_cancer_context" / variant / "dqn_eval_metrics.csv"
            eval_path = nested if nested.exists() else eval_path
        if not eval_path.exists():
            continue
        row = pd.read_csv(eval_path).iloc[0].to_dict()
        row["policy"] = policy
        extra_rows.append(row)
    if not extra_rows:
        return dqn_results
    return pd.concat([dqn_results, pd.DataFrame(extra_rows)], ignore_index=True)


def _resolve_context_sweep_eval_path(sweep_root: Path, variant: str) -> Path | None:
    """Find eval CSV for a sweep variant (handles Modal pull double-nesting)."""
    candidates = (
        sweep_root / variant / "dqn_eval_metrics.csv",
        sweep_root / "dqn_context_sweeps" / variant / "dqn_eval_metrics.csv",
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def _resolve_context_sweep_behavior_dir(sweep_root: Path, variant: str) -> Path | None:
    """Find behavior_figures for a context sweep variant (handles nested pulls)."""
    candidates = (
        sweep_root / variant / "behavior_figures",
        sweep_root / "dqn_context_sweeps" / variant / "behavior_figures",
    )
    for path in candidates:
        if path.is_dir():
            return path
    return None


def resolve_best_context_sweep_variant(
    sweep_root: str | Path,
    *,
    variant: str | None = None,
    exclude: frozenset[str] = CONTEXT_SWEEP_EXCLUDE,
) -> str | None:
    """Return the highest eval-reward context sweep variant on disk."""
    if variant:
        return variant
    table = load_context_sweep_eval_table(sweep_root, exclude=exclude)
    if table.empty:
        return None
    return str(table.iloc[0]["variant"])


def resolve_context_sweep_variant_with_behavior(
    sweep_root: str | Path,
    *,
    preferred_variant: str | None = None,
    exclude: frozenset[str] = CONTEXT_SWEEP_EXCLUDE,
) -> tuple[str | None, Path | None]:
    """Best eval variant that has behavior figures, walking down the ranked list."""
    root = Path(sweep_root)
    table = load_context_sweep_eval_table(root, exclude=exclude)
    if table.empty:
        return None, None

    variants_to_try: list[str] = []
    if preferred_variant:
        variants_to_try.append(preferred_variant)
    for variant_name in table["variant"].astype(str):
        if variant_name not in variants_to_try:
            variants_to_try.append(variant_name)

    for variant_name in variants_to_try:
        behavior_dir = _resolve_context_sweep_behavior_dir(root, variant_name)
        if behavior_dir is not None:
            return variant_name, behavior_dir

    best_eval = str(table.iloc[0]["variant"])
    return preferred_variant or best_eval, None


def _iter_context_sweep_eval_paths(sweep_root: Path) -> list[tuple[str, Path]]:
    """List (variant_name, eval_path) for all sweep runs under sweep_root."""
    found: dict[str, Path] = {}
    patterns = (
        sweep_root.glob("*/dqn_eval_metrics.csv"),
        sweep_root.glob("dqn_context_sweeps/*/dqn_eval_metrics.csv"),
    )
    for pattern in patterns:
        for eval_path in pattern:
            variant_name = eval_path.parent.name
            if variant_name == "dqn_context_sweeps":
                continue
            found[variant_name] = eval_path
    return list(found.items())


def append_context_sweep_metrics(
    dqn_results: pd.DataFrame,
    sweep_root: str | Path,
    *,
    variant: str | None = None,
    policy: str = CONTEXT_DQN_POSTER_POLICY,
) -> tuple[pd.DataFrame, str | None]:
    """Add one Context DQN row from ``dqn_context_sweeps`` (best or named variant).

    Returns updated metrics and the sweep variant name used (if any).
    """
    root = Path(sweep_root)
    if not root.exists():
        return dqn_results, None

    chosen_variant = resolve_best_context_sweep_variant(root, variant=variant)
    if chosen_variant is None:
        return dqn_results, None

    eval_path = _resolve_context_sweep_eval_path(root, chosen_variant)
    if eval_path is None:
        return dqn_results, None

    row = pd.read_csv(eval_path).iloc[0].to_dict()
    row["policy"] = policy
    without_context = dqn_results[dqn_results["policy"] != policy]
    updated = pd.concat([without_context, pd.DataFrame([row])], ignore_index=True)
    return updated, chosen_variant


def load_context_sweep_eval_table(
    sweep_root: str | Path,
    *,
    exclude: frozenset[str] = CONTEXT_SWEEP_EXCLUDE,
) -> pd.DataFrame:
    """Load eval metrics for all pulled context sweep variants."""
    rows: list[dict[str, float | int | str]] = []
    for variant_name, eval_path in _iter_context_sweep_eval_paths(Path(sweep_root)):
        if variant_name in exclude:
            continue
        row = pd.read_csv(eval_path).iloc[0].to_dict()
        row["variant"] = variant_name
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.sort_values("total_reward", ascending=False).reset_index(drop=True)


def generate_context_sweep_figures(
    sweep_root: str | Path,
    output_dir: str | Path,
    *,
    structured_total_reward: float = 1.035,
) -> list[Path]:
    """Plot total reward and modality usage for all context sweep runs."""
    frame = load_context_sweep_eval_table(sweep_root)
    if frame.empty:
        return []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return [
        _plot_context_sweep_reward_and_modality(
            frame,
            output_path,
            structured_total_reward=structured_total_reward,
        )
    ]


def _modality_usage_series(metrics: pd.DataFrame) -> pd.Series:
    """Mean per-modality query counts from eval or trajectory metrics."""
    columns = [column for column in MODALITY_QUERY_COLUMNS if column in metrics.columns]
    if not columns:
        columns = sorted(column for column in metrics.columns if column.startswith("n_query_"))
    if metrics.empty or not columns:
        return pd.Series(dtype=float)
    if len(metrics) == 1:
        return metrics.iloc[0][columns].astype(float)
    return metrics[columns].astype(float).mean()


def _modality_display_labels(columns: list[str]) -> list[str]:
    return [column.removeprefix("n_query_").replace("_", " ") for column in columns]


def _modality_bar_colors(columns: list[str]) -> list[str]:
    return [MODALITY_COLORS.get(column.removeprefix("n_query_"), "#888888") for column in columns]


def generate_context_dqn_modality_figure(
    context_metrics: pd.DataFrame,
    output_dir: str | Path,
    *,
    variant: str = "",
    structured_metrics: pd.DataFrame | None = None,
) -> Path | None:
    """Plot mean modality usage for the best Context DQN (optionally vs Structured)."""
    context_usage = _modality_usage_series(context_metrics)
    if context_usage.empty:
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    structured_usage = (
        _modality_usage_series(structured_metrics)
        if structured_metrics is not None and not structured_metrics.empty
        else None
    )
    return _plot_context_dqn_modality_usage(
        context_usage,
        output_path,
        variant=variant,
        structured_usage=structured_usage,
        context_reward=_metric_value(context_metrics, "total_reward"),
        structured_reward=_metric_value(structured_metrics, "total_reward") if structured_metrics is not None else None,
    )


def _metric_value(metrics: pd.DataFrame | None, column: str) -> float | None:
    if metrics is None or metrics.empty or column not in metrics.columns:
        return None
    value = metrics.iloc[0][column]
    return float(value) if pd.notna(value) else None


def _plot_context_dqn_modality_usage(
    context_usage: pd.Series,
    output_dir: Path,
    *,
    variant: str = "",
    structured_usage: pd.Series | None = None,
    context_reward: float | None = None,
    structured_reward: float | None = None,
) -> Path:
    plt = _pyplot()
    columns = [str(index) for index in context_usage.index]
    labels = _modality_display_labels(columns)
    colors = _modality_bar_colors(columns)

    if structured_usage is not None and not structured_usage.empty:
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        y_positions = list(range(len(labels)))
        bar_height = 0.34
        structured_values = [float(structured_usage.get(column, 0.0)) for column in columns]
        context_values = [float(context_usage.get(column, 0.0)) for column in columns]
        structured_positions = [y + bar_height / 2 for y in y_positions]
        context_positions = [y - bar_height / 2 for y in y_positions]
        ax.barh(
            structured_positions,
            structured_values,
            height=bar_height,
            color=colors,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.45,
            label="Structured DQN",
        )
        bars = ax.barh(
            context_positions,
            context_values,
            height=bar_height,
            color=colors,
            edgecolor="black",
            linewidth=1.0,
            label="Context DQN",
        )
        for bar, value in zip(bars, context_values, strict=True):
            if value <= 0:
                continue
            ax.text(
                value + 0.08,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}",
                va="center",
                ha="left",
                fontsize=8,
            )
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        title_variant = _context_sweep_short_label(variant) if variant else "Context DQN"
        reward_bits = []
        if context_reward is not None:
            reward_bits.append(f"Context={context_reward:.3f}")
        if structured_reward is not None:
            reward_bits.append(f"Structured={structured_reward:.3f}")
        reward_suffix = f" ({', '.join(reward_bits)})" if reward_bits else ""
        ax.set_title(f"Modality usage: Structured vs {title_variant}{reward_suffix}")
        ax.legend(loc="lower right", fontsize=8)
    else:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        values = context_usage.sort_values(ascending=True)
        sorted_columns = [str(index) for index in values.index]
        sorted_labels = _modality_display_labels(sorted_columns)
        sorted_colors = _modality_bar_colors(sorted_columns)
        bars = ax.barh(
            sorted_labels,
            values.values,
            color=sorted_colors,
            edgecolor="black",
            linewidth=1.0,
        )
        for bar, value in zip(bars, values.values, strict=True):
            if value <= 0:
                continue
            ax.text(
                value + 0.08,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}",
                va="center",
                ha="left",
                fontsize=9,
            )
        title_variant = _context_sweep_short_label(variant) if variant else "Context DQN"
        reward_suffix = f" (total reward={context_reward:.3f})" if context_reward is not None else ""
        ax.set_title(f"{title_variant}: mean modality usage{reward_suffix}")

    ax.set_xlabel("Mean queries per episode")
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, output_dir / "context_dqn_modality_usage.png")


def _context_sweep_short_label(variant: str) -> str:
    return CONTEXT_SWEEP_SHORT_LABELS.get(variant, variant.removeprefix("ctx_larger_"))


def _plot_context_sweep_reward_and_modality(
    results: pd.DataFrame,
    output_dir: Path,
    *,
    structured_total_reward: float = 1.035,
) -> Path:
    plt = _pyplot()
    available = {str(row.variant): row for row in results.itertuples(index=False)}
    ordered_rows = [
        available[variant]._asdict()
        for variant in CONTEXT_SWEEP_POSTER_VARIANTS
        if variant in available
    ]
    if ordered_rows:
        frame = pd.DataFrame(ordered_rows)
    else:
        frame = results.sort_values("total_reward", ascending=False).head(8).reset_index(drop=True)
    labels = [_context_sweep_short_label(str(v)) for v in frame["variant"]]
    modality_columns = [
        column
        for column in (
            "n_query_expression",
            "n_query_cna",
            "n_query_damaging_mutation",
            "n_query_hotspot_mutation",
        )
        if column in frame.columns
    ]
    modality_labels = [column.removeprefix("n_query_").replace("_", " ") for column in modality_columns]
    colors = [
        MODALITY_COLORS.get(column.removeprefix("n_query_"), "#999999")
        for column in modality_columns
    ]

    fig, (ax_reward, ax_mod) = plt.subplots(
        1,
        2,
        figsize=POSTER_BAR_FIGSIZE,
        dpi=POSTER_BAR_DPI,
        gridspec_kw={"width_ratios": [1.0, 1.35], "wspace": 0.62},
    )

    rewards = frame["total_reward"].astype(float)
    reward_colors = [
        DQN_COLOR if str(variant) == "ctx_select_only_init_frozen" else BASELINE_COLOR
        for variant in frame["variant"]
    ]
    y_positions = list(range(len(frame)))
    bars = ax_reward.barh(y_positions, rewards, color=reward_colors, edgecolor="black", linewidth=0.8)
    ax_reward.axvline(structured_total_reward, color="#333333", linestyle="--", linewidth=1.2, label="Structured")
    for bar, reward in zip(bars, rewards, strict=True):
        ax_reward.text(
            float(reward) + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{float(reward):.3f}",
            va="center",
            ha="left",
            fontsize=POSTER_ANNOTATION_SIZE,
        )
    ax_reward.set_yticks(y_positions)
    ax_reward.set_yticklabels(labels)
    ax_reward.invert_yaxis()
    ax_reward.set_xlabel("Eval total reward")
    ax_reward.set_title("Where should cancer context enter?", fontsize=POSTER_TITLE_SIZE, pad=18)
    ax_reward.tick_params(axis="both", labelsize=POSTER_TICK_SIZE)
    ax_reward.xaxis.label.set_size(POSTER_LABEL_SIZE)
    ax_reward.legend(loc="lower right", fontsize=POSTER_LEGEND_SIZE)
    ax_reward.grid(axis="x", alpha=0.25)

    left = [0.0] * len(frame)
    for column, label, color in zip(modality_columns, modality_labels, colors, strict=True):
        values = frame[column].astype(float).tolist()
        ax_mod.barh(y_positions, values, left=left, height=0.75, label=label, color=color, edgecolor="white")
        left = [previous + current for previous, current in zip(left, values, strict=True)]

    ax_mod.set_yticks(y_positions)
    ax_mod.set_yticklabels(labels)
    ax_mod.invert_yaxis()
    ax_mod.set_xlabel("Mean queries per episode")
    ax_mod.set_title("Acquisition behavior", fontsize=POSTER_TITLE_SIZE, pad=18)
    ax_mod.tick_params(axis="both", labelsize=POSTER_TICK_SIZE)
    ax_mod.xaxis.label.set_size(POSTER_LABEL_SIZE)
    ax_mod.legend(loc="lower right", fontsize=POSTER_LEGEND_SIZE, ncol=2)
    ax_mod.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "Cancer-context ablations: context throughout vs SELECT-only",
        fontsize=POSTER_SUPTITLE_SIZE,
        y=0.97,
    )
    fig.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.14, wspace=0.62)
    fig.tight_layout()
    return _save(fig, output_dir / "context_sweep_reward_and_modality.png")


def generate_baseline_figures(
    data_metrics_path: str | Path | None,
    environment_metrics_path: str | Path | None,
    output_dir: str | Path,
    dqn_metrics_path: str | Path | None = None,
    dqn_results: pd.DataFrame | None = None,
    dqn_trajectory_path: str | Path | None = None,
    poster_subset: bool = True,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data_results = _read_metrics(data_metrics_path)
    environment_results = _read_metrics(environment_metrics_path)
    if dqn_results is None:
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
    ablation_dir: str | Path | None = None,
    variant_order: tuple[str, ...] = ABLATION_VARIANT_ORDER,
    ablation_labels: dict[str, str] | None = None,
    title_prefix: str = "Architecture Ablation",
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = Path(ablation_summary_path)
    frame: pd.DataFrame | None
    if summary_path.exists():
        frame = pd.read_csv(summary_path)
    else:
        root = Path(ablation_dir) if ablation_dir is not None else summary_path.parent
        rows: list[dict[str, float | int | str]] = []
        for variant in variant_order:
            eval_path = root / variant / "dqn_eval_metrics.csv"
            if not eval_path.exists():
                continue
            row = pd.read_csv(eval_path).iloc[0].to_dict()
            row["variant"] = variant
            rows.append(row)
        frame = pd.DataFrame(rows) if rows else None
    if frame is None or frame.empty:
        return []
    labels = ablation_labels or ABLATION_LABELS
    frame = _prepare_ablation_frame(frame, variant_order=variant_order)
    return [
        _plot_ablation_total_reward(frame, output_path, labels=labels, title_prefix=title_prefix),
        _plot_ablation_quality_vs_queries(frame, output_path, labels=labels, title_prefix=title_prefix),
        _plot_ablation_modality_usage(frame, output_path, labels=labels, title_prefix=title_prefix),
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


def _prepare_ablation_frame(
    results: pd.DataFrame,
    variant_order: tuple[str, ...] = ABLATION_VARIANT_ORDER,
) -> pd.DataFrame:
    if "variant" not in results.columns:
        raise ValueError("Ablation summary must include a 'variant' column.")
    allowed = set(variant_order) | {f"cancer-context_{name}" for name in variant_order}
    frame = results[results["variant"].isin(allowed)].copy()
    if frame.empty:
        raise ValueError("No expected DQN ablation variants were found.")
    normalized = frame["variant"].astype(str).str.removeprefix("cancer-context_")
    order = {variant: index for index, variant in enumerate(variant_order)}
    frame["_sort"] = normalized.map(order)
    return frame.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)


def _ablation_label(variant: str, labels: dict[str, str]) -> str:
    key = str(variant).removeprefix("cancer-context_")
    return labels.get(key, labels.get(str(variant), str(variant)))


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
        is_dqn = str(row.policy) in POSTER_DQN_POLICIES
        ax.scatter(
            row.query_cost,
            row.target_reward,
            s=220 if is_dqn else 90,
            c=DQN_COLOR if is_dqn else BASELINE_COLOR,
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

    fig, ax = plt.subplots(figsize=POSTER_BAR_FIGSIZE, dpi=POSTER_BAR_DPI)
    labels = [_short_policy_name(policy, max_width=14) for policy in frame["policy"]]
    colors = [DQN_COLOR if policy in POSTER_DQN_POLICIES else BASELINE_COLOR for policy in frame["policy"]]
    edge_widths = [DQN_EDGE_WIDTH if policy in POSTER_DQN_POLICIES else 0.8 for policy in frame["policy"]]
    bars = ax.barh(labels, frame["total_reward"], color=colors, edgecolor="black")
    for bar, width in zip(bars, edge_widths, strict=True):
        bar.set_linewidth(width)
    for bar, reward in zip(bars, frame["total_reward"], strict=True):
        x = float(reward)
        y = bar.get_y() + bar.get_height() / 2
        pad = 0.01 if x >= 0 else -0.01
        ax.text(
            x + pad,
            y,
            f"{x:.3f}",
            va="center",
            ha="left" if x >= 0 else "right",
            fontsize=POSTER_ANNOTATION_SIZE,
            color="#111111",
        )
    for index, policy in enumerate(frame["policy"]):
        if policy in POSTER_DQN_POLICIES:
            ax.get_yticklabels()[index].set_fontweight("bold")
            ax.get_yticklabels()[index].set_color(DQN_COLOR)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Total episode reward (target quality − query cost)")
    ax.set_title("Total Reward After Query Costs", fontsize=POSTER_TITLE_SIZE, pad=18)
    ax.tick_params(axis="both", labelsize=POSTER_TICK_SIZE)
    ax.xaxis.label.set_size(POSTER_LABEL_SIZE)
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, output_dir / "environment_total_reward.png")


def _plot_query_count_vs_hit_rate(results: pd.DataFrame, output_dir: Path) -> Path:
    plt = _pyplot()
    frame = results[results["policy"] != "rl_env_query_all_average_then_select"].copy()
    fig, ax = plt.subplots(figsize=(11, 6.5))
    label_offsets = {
        DQN_POLICY: (9, 10),
        CONTEXT_DQN_POSTER_POLICY: (10, 8),
        "rl_env_query_expression_then_select": (10, -15),
        "rl_env_query_expression_budget_12_then_select": (10, 6),
        "rl_env_query_cna_budget_12_then_select": (10, -16),
        "rl_env_query_expression_budget_8_then_select": (10, -10),
        "rl_env_oracle_select": (8, 8),
        "rl_env_random_select": (8, 8),
    }
    for index, row in enumerate(frame.itertuples(index=False)):
        is_dqn = str(row.policy) in POSTER_DQN_POLICIES
        ax.scatter(
            row.n_queries,
            row.hit_at_k,
            s=220 if is_dqn else 90,
            c=DQN_COLOR if is_dqn else BASELINE_COLOR,
            edgecolors="black",
            linewidths=DQN_EDGE_WIDTH if is_dqn else 0.8,
            zorder=3 if is_dqn else 2,
        )
        offset = label_offsets.get(str(row.policy), (7, 7 if index % 2 == 0 else -12))
        ax.annotate(
            _short_policy_name(str(row.policy)),
            (float(row.n_queries), float(row.hit_at_k)),
            xytext=offset,
            textcoords="offset points",
            fontsize=8 if is_dqn else 7,
            fontweight="bold" if is_dqn else "normal",
        )
    ax.set_xlabel("Number of evidence queries")
    ax.set_ylabel("Hit@k")
    ax.set_title("Query Count vs Hit Rate")
    ax.grid(alpha=0.25)
    return _save(fig, output_dir / "environment_queries_vs_hit_rate.png")


def _plot_ablation_total_reward(
    results: pd.DataFrame,
    output_dir: Path,
    labels: dict[str, str] | None = None,
    title_prefix: str = "Architecture Ablation",
) -> Path:
    plt = _pyplot()
    label_map = labels or ABLATION_LABELS
    frame = results.sort_values("total_reward", ascending=True).reset_index(drop=True)
    bar_labels = [_ablation_label(variant, label_map) for variant in frame["variant"]]
    colors = [MLP_COLOR if "mlp" in str(variant) else DQN_COLOR for variant in frame["variant"]]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.barh(bar_labels, frame["total_reward"], color=colors, edgecolor="black", linewidth=0.8)
    best_index = int(frame["total_reward"].idxmax())
    bars[best_index].set_linewidth(2.4)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Total episode reward")
    ax.set_title(f"{title_prefix} (Total Reward)")
    ax.grid(axis="x", alpha=0.25)
    for index, value in enumerate(frame["total_reward"]):
        ax.text(float(value) + 0.01, index, f"{float(value):.3f}", va="center", fontsize=8)
    return _save(fig, output_dir / "dqn_ablation_total_reward.png")


def _plot_ablation_quality_vs_queries(
    results: pd.DataFrame,
    output_dir: Path,
    labels: dict[str, str] | None = None,
    title_prefix: str = "Architecture Ablation",
) -> Path:
    plt = _pyplot()
    label_map = labels or ABLATION_LABELS
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
            c=MLP_COLOR if is_mlp else DQN_COLOR,
            edgecolors="black",
            linewidths=0.9,
        )
        ax.annotate(
            _ablation_label(variant, label_map).replace("\n", " "),
            (float(row.n_queries), float(row.target_reward)),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Mean evidence queries")
    ax.set_ylabel("Target quality (-selected dependency)")
    ax.set_title(f"{title_prefix}: Query-Quality Tradeoff")
    ax.grid(alpha=0.25)
    return _save(fig, output_dir / "dqn_ablation_quality_vs_queries.png")


def _plot_ablation_modality_usage(
    results: pd.DataFrame,
    output_dir: Path,
    labels: dict[str, str] | None = None,
    title_prefix: str = "Architecture Ablation",
) -> Path:
    plt = _pyplot()
    label_map = labels or ABLATION_LABELS
    modality_columns = [column for column in results.columns if column.startswith("n_query_")]
    frame = results.set_index("variant")[modality_columns].copy()
    frame.index = [_ablation_label(variant, label_map).replace("\n", " ") for variant in frame.index]
    frame.columns = [column.removeprefix("n_query_").replace("_", " ") for column in frame.columns]

    fig, ax = plt.subplots(figsize=(11, 6))
    frame.plot(kind="bar", stacked=True, ax=ax, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("")
    ax.set_ylabel("Mean queries per episode")
    ax.set_title(f"{title_prefix}: Modality Usage")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Modality", loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, output_dir / "dqn_ablation_modality_usage.png")


def _plot_ranking_metrics(results: pd.DataFrame, output_dir: Path) -> Path:
    plt = _pyplot()
    frame = results.copy()
    metric_colors = ("#6A4C93", "#9C89B8", "#C8C1D9")
    policies = [_short_policy_name(policy, max_width=12) for policy in frame["policy"]]
    x_positions = list(range(len(frame)))
    width = 0.24

    fig, ax = plt.subplots(figsize=POSTER_BAR_FIGSIZE, dpi=POSTER_BAR_DPI)
    for offset, metric in enumerate(METRIC_COLUMNS):
        xs = [x + (offset - 1) * width for x in x_positions]
        bar_colors = []
        edge_colors = []
        edge_widths = []
        for policy in frame["policy"]:
            if policy in POSTER_DQN_POLICIES:
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
        for bar, value in zip(bars, frame[metric], strict=True):
            height = float(value)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=POSTER_ANNOTATION_SIZE,
                color="#111111",
            )

    ax.set_xticks(x_positions)
    tick_labels = ax.set_xticklabels(policies, rotation=35, ha="right", fontsize=POSTER_TICK_SIZE)
    for index, policy in enumerate(frame["policy"]):
        if policy in POSTER_DQN_POLICIES:
            tick_labels[index].set_fontweight("bold")
            tick_labels[index].set_color(DQN_COLOR)
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("Metric value")
    ax.set_title("Ranking Metrics", fontsize=POSTER_TITLE_SIZE, pad=18)
    ax.tick_params(axis="y", labelsize=POSTER_TICK_SIZE)
    ax.yaxis.label.set_size(POSTER_LABEL_SIZE)
    ax.legend(loc="upper left", fontsize=POSTER_LEGEND_SIZE)
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
    ax.set_title("Tuned Structured DQN: Query Count Distribution")
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, output_dir / "dqn_query_count_distribution.png")


def _plot_dqn_modality_usage(results: pd.DataFrame, output_dir: Path) -> Path:
    plt = _pyplot()
    modality_columns = [column for column in results.columns if column.startswith("n_query_")]
    usage = results[modality_columns].mean().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = [column.removeprefix("n_query_").replace("_", "\n") for column in usage.index]
    colors = _modality_bar_colors([str(column) for column in usage.index])
    ax.barh(labels, usage.values, color=colors, edgecolor="black", linewidth=1.0)
    ax.set_xlabel("Mean queries per episode")
    ax.set_title("Tuned Structured DQN: Mean Modality Usage")
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
        c=BASELINE_COLOR,
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
    fig, ax = plt.subplots(figsize=POSTER_BAR_FIGSIZE, dpi=POSTER_BAR_DPI)
    ax.bar(
        query_efficiency["gene_true_rank"].astype(str),
        query_efficiency["query_fraction"],
        color=DQN_COLOR,
        edgecolor="black",
        linewidth=1.8,
    )
    ax.axhline(1.0 / max(len(query_efficiency), 1), color="#3B2E58", linestyle="--", linewidth=2.4)
    ax.set_xlabel("True dependency rank of queried gene")
    ax.set_ylabel("Fraction of DQN query actions")
    ax.set_title("DQN Query Efficiency by True Dependency Rank", fontsize=POSTER_TITLE_SIZE, pad=18)
    ax.tick_params(axis="both", labelsize=POSTER_TICK_SIZE)
    ax.xaxis.label.set_size(POSTER_LABEL_SIZE)
    ax.yaxis.label.set_size(POSTER_LABEL_SIZE)
    ax.grid(axis="y", alpha=0.25)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.16)
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
    image = ax.imshow(values.to_numpy(), aspect="auto", cmap="Purples")
    ax.set_xticks(range(len(modality_labels)))
    ax.set_xticklabels(modality_labels, fontsize=8)
    ax.set_yticks(range(len(context_labels)))
    ax.set_yticklabels(context_labels, fontsize=8)
    ax.set_xlabel("Modality")
    ax.set_title("Modality Usage by Cancer Lineage")
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

    color_map = dict(MODALITY_COLORS)
    for modality in frame["modality"].dropna().unique():
        key = str(modality)
        if key not in color_map:
            color_map[key] = "#999999"

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
