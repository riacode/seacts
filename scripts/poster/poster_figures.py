from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "outputs" / "depmap_baselines"
POSTER = ROOT / "poster_outputs"

MODEL_COLOR = "#6A4C93"
BASE_COLOR = "#B7A6D6"
SELECT_COLOR = "#3B2E58"
BAR_FIGSIZE = (5056 / 180, 2656 / 180)
BAR_DPI = 180
TITLE_SIZE = 42
SUPTITLE_SIZE = 46
LABEL_SIZE = 34
TICK_SIZE = 30
ANNOTATION_SIZE = 28
LEGEND_SIZE = 28
PURPLE_HEATMAP = LinearSegmentedColormap.from_list(
    "poster_purple",
    ["#E4D9F2", "#8D6BB8", "#3B0F70"],
)

MODALITIES = (
    ("n_query_expression", "expression", "#DD8452"),
    ("n_query_cna", "cna", "#4C72B0"),
    ("n_query_damaging_mutation", "damaging mutation", "#55A868"),
    ("n_query_hotspot_mutation", "hotspot mutation", "#C44E52"),
)
MODALITY_COLOR_BY_KEY = {
    label.replace(" ", "_"): color
    for _column, label, color in MODALITIES
}
TRAJECTORY_COLORS = {**MODALITY_COLOR_BY_KEY, "select": "#333333"}


def generate_validation_reward_graph() -> None:
    metrics_path = RESULTS / "dqn_context_sweeps" / "ctx_select_only_init_frozen" / "dqn_training_metrics.csv"
    output_path = POSTER / "reward_graph.png"
    frame = pd.read_csv(metrics_path)
    validation = frame.dropna(subset=["validation_total_reward"]).copy()
    validation["step"] = validation["episode"] + 1

    fig, ax = plt.subplots(figsize=BAR_FIGSIZE, dpi=BAR_DPI)
    ax.plot(
        validation["step"],
        validation["validation_total_reward"],
        color=MODEL_COLOR,
        linewidth=4.2,
        marker="o",
        markersize=6.5,
        markerfacecolor="white",
        markeredgewidth=2.0,
        markeredgecolor=MODEL_COLOR,
    )

    ax.set_title("Validation Reward During Context DQN Training", fontsize=TITLE_SIZE, pad=24)
    ax.set_xlabel("Training episode", fontsize=LABEL_SIZE)
    ax.set_ylabel("Mean validation reward", fontsize=LABEL_SIZE)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value / 1000:.0f}k" if value else "0"))
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(axis="y", color="#D7D7E2", linewidth=1.6, alpha=0.85)
    ax.grid(axis="x", color="#EEEEF3", linewidth=1.0, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#7E7A89")
    ax.spines["bottom"].set_color("#7E7A89")
    ax.set_xlim(0, 20000)
    ax.set_ylim(0.62, 1.07)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.86, bottom=0.18)
    fig.savefig(output_path, dpi=BAR_DPI, facecolor="white")
    plt.close(fig)
    print(f"Wrote {output_path}")


def generate_ablation_storyline_figures() -> None:
    output_dir = POSTER / "02_architecture_ablations"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = (
        ("MLP\n1-step", RESULTS / "dqn_ablation" / "scale1_long_mlp_1step" / "dqn_eval_metrics.csv", BASE_COLOR),
        ("MLP\n3-step", RESULTS / "dqn_ablation" / "scale1_long_mlp_3step" / "dqn_eval_metrics.csv", BASE_COLOR),
        ("Structured\n1-step", RESULTS / "dqn_ablation" / "scale1_long_structured_1step" / "dqn_eval_metrics.csv", MODEL_COLOR),
        ("Structured dueling\n1-step", RESULTS / "dqn_ablation" / "scale1_long_structured_dueling_1step" / "dqn_eval_metrics.csv", MODEL_COLOR),
        ("Structured\n3-step", RESULTS / "dqn_ablation" / "scale1_long_structured_3step" / "dqn_eval_metrics.csv", MODEL_COLOR),
        ("Structured dueling\n3-step", RESULTS / "dqn_ablation" / "scale1_long_structured_dueling_3step" / "dqn_eval_metrics.csv", MODEL_COLOR),
    )
    frame = _load_run_frame(runs)
    frame = frame.sort_values("total_reward", ascending=True).reset_index(drop=True)
    _plot_reward_bars(
        frame,
        output_dir / "dqn_ablation_total_reward.png",
        title="Ablation Reward Comparison",
        xlim=(0.65, 1.08),
    )
    _plot_modality_bars(
        frame,
        output_dir / "dqn_ablation_modality_usage.png",
        title="Ablation Modality Usage",
    )


def generate_context_storyline_figure() -> None:
    output_dir = POSTER / "03_lineage_and_context"
    runs = (
        ("Context from start", RESULTS / "dqn_context_sweeps" / "ctx_larger_init_structured" / "dqn_eval_metrics.csv", BASE_COLOR),
        ("Smaller context\nembedding", RESULTS / "dqn_context_sweeps" / "ctx_larger_ctx32" / "dqn_eval_metrics.csv", BASE_COLOR),
        ("Dueling context\nhead", RESULTS / "dqn_context_sweeps" / "ctx_larger_dueling" / "dqn_eval_metrics.csv", BASE_COLOR),
        ("Lineage-specific\nquery scores", RESULTS / "dqn_context_sweeps" / "ctx_lineage_env_init_structured" / "dqn_eval_metrics.csv", BASE_COLOR),
        ("Context fusion", RESULTS / "dqn_context_sweeps" / "ctx_fusion_lineage_init_structured" / "dqn_eval_metrics.csv", BASE_COLOR),
        ("SELECT-only\nContext", RESULTS / "dqn_context_sweeps" / "ctx_select_only_init_frozen" / "dqn_eval_metrics.csv", MODEL_COLOR),
    )
    frame = _load_run_frame(runs)
    frame = frame.sort_values("total_reward", ascending=True).reset_index(drop=True)

    structured = pd.read_csv(RESULTS / "best_structured_1step_larger" / "dqn_eval_metrics.csv")
    structured_reward = float(structured.iloc[0]["total_reward"])
    _plot_context_reward_bars(
        frame,
        output_dir / "context_storyline_total_reward.png",
        title="Cancer Context Reward Comparison",
        xlim=(0.88, 1.08),
        structured_reward=structured_reward,
    )
    _plot_modality_bars(
        frame,
        output_dir / "context_storyline_modality_usage.png",
        title="Cancer Context Modality Usage",
    )


def generate_representative_context_trajectories() -> None:
    run_dir = RESULTS / "dqn_context_sweeps" / "ctx_select_only_init_frozen"
    output_path = POSTER / "03_lineage_and_context" / "context_dqn_representative_example_trajectories.png"
    steps = pd.read_csv(run_dir / "dqn_step_log.csv")
    episodes = pd.read_csv(run_dir / "dqn_episode_summary.csv")
    counts = _episode_modality_counts(steps)

    expression_only = counts[counts["non_expression"] == 0].sort_values("expression", ascending=True).index.astype(int).tolist()
    light_damage = counts[(counts["damaging_mutation"].between(1, 3)) & (counts["hotspot_mutation"] == 0)].sort_values(["damaging_mutation", "expression"]).index.astype(int).tolist()
    hotspot = counts[counts["hotspot_mutation"] > 0].sort_values(["hotspot_mutation", "damaging_mutation"]).index.astype(int).tolist()
    moderate_damage = counts[(counts["damaging_mutation"].between(4, 7)) & (counts["hotspot_mutation"] == 0)].sort_values(["damaging_mutation", "expression"]).index.astype(int).tolist()

    selected: list[int] = []
    for group, limit in ((expression_only, 8), (light_damage, 2), (hotspot, 1), (moderate_damage, 1)):
        added = 0
        for episode_id in group:
            if episode_id not in selected:
                selected.append(episode_id)
                added += 1
            if added >= limit:
                break
    selected = selected[:12]
    _plot_trajectories(
        steps=steps,
        episodes=episodes,
        selected=selected,
        output_path=output_path,
        title="Representative Final Context DQN Trajectories",
    )


def regenerate_lineage_heatmaps() -> None:
    output_dir = POSTER / "03_lineage_and_context"
    sources = (
        (
            RESULTS / "best_structured_1step_larger" / "best_structured_1step_larger" / "behavior_analysis" / "dqn_modality_usage_by_context.csv",
            output_dir / "structured_modality_by_oncotree_lineage.png",
            "Structured DQN: Modality Usage by Cancer Lineage",
            PURPLE_HEATMAP,
        ),
        (
            RESULTS / "dqn_context_sweeps" / "ctx_select_only_init_frozen" / "behavior_analysis" / "dqn_modality_usage_by_context.csv",
            output_dir / "context_dqn_modality_by_oncotree_lineage.png",
            "Final Context DQN: Modality Usage by Cancer Lineage",
            PURPLE_HEATMAP,
        ),
    )
    for source, dest, title, cmap in sources:
        _plot_lineage_heatmap(source, dest, title, cmap=cmap)


def _load_run_frame(runs: tuple[tuple[str, Path, str], ...]) -> pd.DataFrame:
    rows = []
    for label, path, color in runs:
        row = pd.read_csv(path).iloc[0].to_dict()
        row.update({"label": label, "color": color})
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_reward_bars(frame: pd.DataFrame, output_path: Path, *, title: str, xlim: tuple[float, float]) -> None:
    fig, ax = plt.subplots(figsize=BAR_FIGSIZE, dpi=BAR_DPI)
    _draw_reward_bars(ax, frame, xlim=xlim)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=18)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.88, bottom=0.14)
    fig.savefig(output_path, dpi=BAR_DPI, facecolor="white")
    plt.close(fig)
    print(f"Wrote {output_path}")


def _plot_context_reward_bars(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    xlim: tuple[float, float],
    structured_reward: float,
) -> None:
    fig, ax = plt.subplots(figsize=BAR_FIGSIZE, dpi=BAR_DPI)
    _draw_reward_bars(ax, frame, xlim=xlim)
    ax.axvline(structured_reward, color="#333333", linestyle="--", linewidth=2.4, label="Structured DQN")
    ax.legend(loc="lower right", fontsize=LEGEND_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=18)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.88, bottom=0.14)
    fig.savefig(output_path, dpi=BAR_DPI, facecolor="white")
    plt.close(fig)
    print(f"Wrote {output_path}")


def _draw_reward_bars(ax, frame: pd.DataFrame, *, xlim: tuple[float, float]) -> None:
    y_positions = list(range(len(frame)))
    bars = ax.barh(
        y_positions,
        frame["total_reward"].astype(float),
        color=frame["color"],
        edgecolor="black",
        linewidth=1.8,
    )
    for bar, value in zip(bars, frame["total_reward"], strict=True):
        ax.text(
            float(value) + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{float(value):.3f}",
            va="center",
            fontsize=ANNOTATION_SIZE,
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(frame["label"])
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_xlabel("Eval total reward")
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.xaxis.label.set_size(LABEL_SIZE)
    ax.grid(axis="x", alpha=0.25)


def _plot_modality_bars(frame: pd.DataFrame, output_path: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=BAR_FIGSIZE, dpi=BAR_DPI)
    _draw_modality_bars(ax, frame)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=18)
    ax.legend(loc="lower right", fontsize=LEGEND_SIZE, ncol=2)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.88, bottom=0.14)
    fig.savefig(output_path, dpi=BAR_DPI, facecolor="white")
    plt.close(fig)
    print(f"Wrote {output_path}")


def _draw_modality_bars(ax, frame: pd.DataFrame) -> None:
    y_positions = list(range(len(frame)))
    left = [0.0] * len(frame)
    for column, label, color in MODALITIES:
        values = frame[column].astype(float).tolist()
        ax.barh(y_positions, values, left=left, label=label, color=color, edgecolor="white", height=0.75)
        left = [previous + current for previous, current in zip(left, values, strict=True)]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(frame["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean queries per episode")
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.xaxis.label.set_size(LABEL_SIZE)
    ax.grid(axis="x", alpha=0.25)


def _episode_modality_counts(steps: pd.DataFrame) -> pd.DataFrame:
    query_steps = steps[steps["action_type"] == "query"].copy()
    counts = query_steps.pivot_table(index="episode_id", columns="modality", values="step", aggfunc="count", fill_value=0)
    for column in ("expression", "cna", "damaging_mutation", "hotspot_mutation"):
        if column not in counts:
            counts[column] = 0
    counts["non_expression"] = counts[["cna", "damaging_mutation", "hotspot_mutation"]].sum(axis=1)
    return counts


def _plot_trajectories(steps: pd.DataFrame, episodes: pd.DataFrame, selected: list[int], output_path: Path, title: str) -> None:
    frame = steps[steps["episode_id"].isin(selected)].copy()
    selected_episodes = episodes[episodes["episode_id"].isin(selected)].copy()
    labels = {
        int(row.episode_id): f"{int(row.episode_id)} {row.selected_gene} r={float(row.dependency_regret):.2f}"
        for row in selected_episodes.itertuples(index=False)
    }
    # legend room
    trajectory_figsize = (5056 / BAR_DPI, 3000 / BAR_DPI)
    fig, ax = plt.subplots(figsize=trajectory_figsize, dpi=BAR_DPI)
    y_positions = {episode_id: index for index, episode_id in enumerate(selected)}
    for row in frame.itertuples(index=False):
        action_type = str(row.action_type)
        color_key = "select" if action_type == "select" else str(row.modality)
        ax.broken_barh(
            [(float(row.step), 0.85)],
            (y_positions[int(row.episode_id)] - 0.35, 0.7),
            facecolors=TRAJECTORY_COLORS.get(color_key, "#999999"),
            edgecolors="black",
            linewidth=1.0,
        )
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([labels.get(episode_id, str(episode_id)) for episode_id in selected], fontsize=22)
    ax.set_xlabel("DQN step")
    ax.xaxis.label.set_size(LABEL_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=18)
    legend_labels = {
        "expression": "expression",
        "cna": "cna",
        "damaging_mutation": "damaging mutation",
        "hotspot_mutation": "hotspot mutation",
        "select": "select",
    }
    handles = [
        plt.Line2D([0], [0], color=color, linewidth=10, label=legend_labels[label])
        for label, color in TRAJECTORY_COLORS.items()
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncols=5, fontsize=LEGEND_SIZE)
    ax.grid(axis="x", alpha=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.895, bottom=0.24)
    fig.savefig(output_path, dpi=BAR_DPI, facecolor="white")
    plt.close(fig)
    print(f"Wrote {output_path}")


def _plot_lineage_heatmap(source: Path, dest: Path, title: str, *, cmap: str | LinearSegmentedColormap) -> None:
    frame = pd.read_csv(source).sort_values("n_episodes", ascending=False).head(15).copy()
    modality_columns = [column for column in frame.columns if column.startswith("n_query_")]
    context_column = next(column for column in frame.columns if column not in {*modality_columns, "n_episodes"})
    values = frame[modality_columns].astype(float)
    modality_labels = [column.removeprefix("n_query_").replace("_", "\n") for column in modality_columns]
    lineage_labels = [f"{lineage} (n={int(n)})" for lineage, n in zip(frame[context_column], frame["n_episodes"], strict=False)]
    fig_height = max(5.0, len(frame) * 0.42)
    fig, ax = plt.subplots(figsize=(9.5, fig_height))
    image = ax.imshow(values.to_numpy(), aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(modality_labels)))
    ax.set_xticklabels(modality_labels, fontsize=8)
    ax.set_yticks(range(len(lineage_labels)))
    ax.set_yticklabels(lineage_labels, fontsize=8)
    ax.set_xlabel("Modality")
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Mean queries per episode")
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            ax.text(col_index, row_index, f"{values.iat[row_index, col_index]:.1f}", ha="center", va="center", fontsize=7, color="black")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(dest, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {dest}")
