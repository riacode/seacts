from __future__ import annotations

from pathlib import Path

import pandas as pd


CELL_LINE_COLUMNS = ("ModelID", "DepMap_ID", "model_id", "cell_line_id", "Model")
CONTEXT_COLUMNS = (
    "OncotreeLineage",
    "OncotreePrimaryDisease",
    "OncotreeSubtype",
    "PrimaryDisease",
    "Lineage",
    "lineage",
    "primary_disease",
)


def join_cell_line_metadata(
    episodes: pd.DataFrame,
    metadata: pd.DataFrame,
    cell_line_column: str | None = None,
) -> pd.DataFrame:
    if "cell_line_id" not in episodes:
        raise ValueError("episodes must include a cell_line_id column.")

    metadata_cell_column = cell_line_column or _first_present(metadata, CELL_LINE_COLUMNS)
    if metadata_cell_column is None:
        raise ValueError(
            "Could not find a cell-line identifier column in metadata. "
            f"Tried: {', '.join(CELL_LINE_COLUMNS)}"
        )

    metadata_copy = metadata.copy()
    metadata_copy[metadata_cell_column] = metadata_copy[metadata_cell_column].astype(str)
    episodes_copy = episodes.copy()
    episodes_copy["cell_line_id"] = episodes_copy["cell_line_id"].astype(str)
    return episodes_copy.merge(
        metadata_copy,
        left_on="cell_line_id",
        right_on=metadata_cell_column,
        how="left",
        suffixes=("", "_metadata"),
    )


def summarize_failure_cases(episodes: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    frame = episodes.copy()
    if "hit_at_k" in frame:
        frame = frame[frame["hit_at_k"] < 1.0]
    if "dependency_regret" in frame:
        frame = frame.sort_values("dependency_regret", ascending=False)
    elif "total_reward" in frame:
        frame = frame.sort_values("total_reward", ascending=True)
    return _select_episode_columns(frame).head(top_n).reset_index(drop=True)


def summarize_success_cases(episodes: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    frame = episodes.copy()
    if "hit_at_k" in frame:
        frame = frame[frame["hit_at_k"] >= 1.0]

    sort_columns = [column for column in ("dependency_regret", "n_queries") if column in frame]
    if sort_columns:
        frame = frame.sort_values(sort_columns, ascending=[True] * len(sort_columns))
    elif "total_reward" in frame:
        frame = frame.sort_values("total_reward", ascending=False)
    return _select_episode_columns(frame).head(top_n).reset_index(drop=True)


def summarize_query_efficiency(steps: pd.DataFrame) -> pd.DataFrame:
    required = {"action_type", "gene_true_rank"}
    missing = required - set(steps.columns)
    if missing:
        raise ValueError(f"steps is missing required columns: {sorted(missing)}")

    queries = steps[steps["action_type"] == "query"].copy()
    if queries.empty:
        return pd.DataFrame(
            columns=[
                "gene_true_rank",
                "n_queries",
                "query_fraction",
                "mean_step",
                "mean_observed_value",
            ]
        )

    grouped = (
        queries.groupby("gene_true_rank", as_index=False)
        .agg(
            n_queries=("gene_true_rank", "size"),
            mean_step=("step", "mean"),
            mean_observed_value=("observed_value", "mean"),
        )
        .sort_values("gene_true_rank")
    )
    grouped["query_fraction"] = grouped["n_queries"] / grouped["n_queries"].sum()
    return grouped[
        ["gene_true_rank", "n_queries", "query_fraction", "mean_step", "mean_observed_value"]
    ]


def modality_usage_by_context(
    episodes: pd.DataFrame,
    context_column: str | None = None,
) -> pd.DataFrame:
    modality_columns = [column for column in episodes.columns if column.startswith("n_query_")]
    if not modality_columns:
        raise ValueError("episodes must include n_query_* modality columns.")

    resolved_context = context_column or _first_present(episodes, CONTEXT_COLUMNS)
    if resolved_context is None:
        raise ValueError(
            "Could not infer a context column. Pass context_column explicitly or include one of: "
            f"{', '.join(CONTEXT_COLUMNS)}"
        )

    frame = episodes.dropna(subset=[resolved_context]).copy()
    if frame.empty:
        return pd.DataFrame(columns=[resolved_context, *modality_columns, "n_episodes"])

    usage = frame.groupby(resolved_context, as_index=False)[modality_columns].mean()
    counts = frame.groupby(resolved_context, as_index=False).size().rename(columns={"size": "n_episodes"})
    return usage.merge(counts, on=resolved_context).sort_values("n_episodes", ascending=False)


def write_behavior_analysis_tables(
    episode_summary_path: str | Path,
    step_log_path: str | Path,
    output_dir: str | Path,
    metadata_path: str | Path | None = None,
    context_column: str | None = None,
) -> list[Path]:
    episodes = pd.read_csv(episode_summary_path)
    steps = pd.read_csv(step_log_path)
    if metadata_path is not None and Path(metadata_path).exists():
        episodes = join_cell_line_metadata(episodes, pd.read_csv(metadata_path))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tables = {
        "dqn_failure_cases.csv": summarize_failure_cases(episodes),
        "dqn_success_cases.csv": summarize_success_cases(episodes),
        "dqn_query_efficiency_by_true_rank.csv": summarize_query_efficiency(steps),
    }
    if metadata_path is not None and Path(metadata_path).exists():
        try:
            tables["dqn_modality_usage_by_context.csv"] = modality_usage_by_context(
                episodes,
                context_column=context_column,
            )
        except ValueError:
            # Some DepMap metadata variants do not expose a lineage-like column.
            # The core behavior tables are still useful without context grouping.
            pass

    paths = []
    for file_name, table in tables.items():
        path = output_path / file_name
        table.to_csv(path, index=False)
        paths.append(path)
    return paths


def _select_episode_columns(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "episode_id",
        "cell_line_id",
        "selected_gene",
        "selected_dependency",
        "dependency_regret",
        "hit_at_k",
        "n_queries",
        "query_cost",
        "total_reward",
    ]
    modality_columns = [column for column in frame.columns if column.startswith("n_query_")]
    context_columns = [column for column in CONTEXT_COLUMNS if column in frame.columns]
    columns = [column for column in [*preferred, *modality_columns, *context_columns] if column in frame]
    return frame[columns] if columns else frame


def _first_present(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    normalized = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None
