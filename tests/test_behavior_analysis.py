from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.behavior_analysis import (
    join_cell_line_metadata,
    modality_usage_by_context,
    summarize_failure_cases,
    summarize_query_efficiency,
    summarize_success_cases,
    write_behavior_analysis_tables,
)


def _episodes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "episode_id": 0,
                "cell_line_id": "ACH-1",
                "selected_gene": "A",
                "selected_dependency": -1.0,
                "dependency_regret": 0.0,
                "hit_at_k": 1.0,
                "n_queries": 4,
                "query_cost": 0.08,
                "total_reward": 0.92,
                "n_query_expression": 4,
                "n_query_cna": 0,
            },
            {
                "episode_id": 1,
                "cell_line_id": "ACH-2",
                "selected_gene": "B",
                "selected_dependency": -0.2,
                "dependency_regret": 0.8,
                "hit_at_k": 0.0,
                "n_queries": 12,
                "query_cost": 0.28,
                "total_reward": -0.08,
                "n_query_expression": 8,
                "n_query_cna": 4,
            },
        ]
    )


def _steps() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "episode_id": 0,
                "cell_line_id": "ACH-1",
                "step": 0,
                "action_type": "query",
                "gene": "A",
                "modality": "expression",
                "observed_value": 1.5,
                "gene_true_rank": 1,
            },
            {
                "episode_id": 1,
                "cell_line_id": "ACH-2",
                "step": 0,
                "action_type": "query",
                "gene": "B",
                "modality": "cna",
                "observed_value": 0.2,
                "gene_true_rank": 5,
            },
            {
                "episode_id": 1,
                "cell_line_id": "ACH-2",
                "step": 1,
                "action_type": "select",
                "gene": "B",
                "modality": None,
                "observed_value": None,
                "gene_true_rank": 5,
            },
        ]
    )


def test_summarize_failure_and_success_cases() -> None:
    failures = summarize_failure_cases(_episodes(), top_n=1)
    successes = summarize_success_cases(_episodes(), top_n=1)

    assert failures["episode_id"].tolist() == [1]
    assert failures.loc[0, "dependency_regret"] == 0.8
    assert successes["episode_id"].tolist() == [0]
    assert successes.loc[0, "n_queries"] == 4


def test_summarize_query_efficiency_groups_query_actions_only() -> None:
    summary = summarize_query_efficiency(_steps())

    assert summary["gene_true_rank"].tolist() == [1, 5]
    assert summary["n_queries"].tolist() == [1, 1]
    assert summary["query_fraction"].tolist() == [0.5, 0.5]


def test_join_metadata_and_context_modality_usage() -> None:
    metadata = pd.DataFrame(
        {
            "ModelID": ["ACH-1", "ACH-2"],
            "OncotreeLineage": ["Skin", "Lung"],
        }
    )

    joined = join_cell_line_metadata(_episodes(), metadata)
    usage = modality_usage_by_context(joined)

    assert "OncotreeLineage" in joined.columns
    assert set(usage["OncotreeLineage"]) == {"Skin", "Lung"}
    lung = usage[usage["OncotreeLineage"] == "Lung"].iloc[0]
    assert lung["n_query_expression"] == 8
    assert lung["n_query_cna"] == 4


def test_modality_usage_requires_context_column() -> None:
    with pytest.raises(ValueError, match="context column"):
        modality_usage_by_context(_episodes())


def test_write_behavior_analysis_tables(tmp_path: Path) -> None:
    episodes_path = tmp_path / "dqn_episode_summary.csv"
    steps_path = tmp_path / "dqn_step_log.csv"
    metadata_path = tmp_path / "Model.csv"
    output_dir = tmp_path / "analysis"

    _episodes().to_csv(episodes_path, index=False)
    _steps().to_csv(steps_path, index=False)
    pd.DataFrame(
        {
            "ModelID": ["ACH-1", "ACH-2"],
            "OncotreeLineage": ["Skin", "Lung"],
        }
    ).to_csv(metadata_path, index=False)

    paths = write_behavior_analysis_tables(
        episode_summary_path=episodes_path,
        step_log_path=steps_path,
        metadata_path=metadata_path,
        output_dir=output_dir,
    )

    assert {path.name for path in paths} == {
        "dqn_failure_cases.csv",
        "dqn_success_cases.csv",
        "dqn_query_efficiency_by_true_rank.csv",
        "dqn_modality_usage_by_context.csv",
    }
    assert all(path.exists() for path in paths)
