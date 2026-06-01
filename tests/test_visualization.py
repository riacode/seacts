from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.visualization import generate_baseline_figures, generate_behavior_figures


def test_generate_baseline_figures_writes_expected_pngs(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    data_path = tmp_path / "data_metrics.csv"
    environment_path = tmp_path / "environment_metrics.csv"
    dqn_path = tmp_path / "dqn_metrics.csv"
    dqn_trajectory_path = tmp_path / "dqn_trajectories.csv"
    output_dir = tmp_path / "figures"

    pd.DataFrame(
        [
            {
                "policy": "data_random_select",
                "selected_dependency": -0.1,
                "hit_at_k": 0.4,
                "ndcg_at_k": 0.2,
                "mrr_at_k": 0.1,
                "query_cost": 0.0,
            }
        ]
    ).to_csv(data_path, index=False)
    pd.DataFrame(
        [
            {
                "policy": "rl_env_random_select",
                "selected_dependency": -0.2,
                "hit_at_k": 0.2,
                "ndcg_at_k": 0.1,
                "mrr_at_k": 0.05,
                "query_cost": 0.0,
                "n_queries": 0,
                "total_reward": -0.2,
            },
            {
                "policy": "rl_env_query_expression_then_select",
                "selected_dependency": -0.7,
                "hit_at_k": 0.8,
                "ndcg_at_k": 0.5,
                "mrr_at_k": 0.4,
                "query_cost": 0.32,
                "n_queries": 16,
                "total_reward": 0.38,
            },
            {
                "policy": "rl_env_oracle_select",
                "selected_dependency": -0.9,
                "hit_at_k": 1.0,
                "ndcg_at_k": 1.0,
                "mrr_at_k": 1.0,
                "query_cost": 0.0,
                "n_queries": 0,
                "total_reward": 0.9,
            },
        ]
    ).to_csv(environment_path, index=False)
    pd.DataFrame(
        [
            {
                "policy": "rl_env_dqn",
                "selected_dependency": -0.6,
                "hit_at_k": 0.7,
                "ndcg_at_k": 0.4,
                "mrr_at_k": 0.3,
                "query_cost": 0.2,
                "n_queries": 10,
                "total_reward": 0.4,
            }
        ]
    ).to_csv(dqn_path, index=False)
    pd.DataFrame(
        [
            {
                "episode_id": 0,
                "cell_line_id": "ACH-1",
                "selected_gene": "A",
                "selected_index": 0,
                "selected_dependency": -0.6,
                "query_cost": 0.2,
                "n_queries": 10,
                "total_reward": 0.4,
                "n_query_expression": 8,
                "n_query_cna": 2,
            }
        ]
    ).to_csv(dqn_trajectory_path, index=False)

    figures = generate_baseline_figures(
        data_path,
        environment_path,
        output_dir,
        dqn_path,
        dqn_trajectory_path,
    )

    assert {figure.name for figure in figures} == {
        "environment_cost_vs_target_reward.png",
        "environment_total_reward.png",
        "environment_queries_vs_hit_rate.png",
        "baseline_ranking_metrics.png",
        "dqn_query_count_distribution.png",
        "dqn_modality_usage.png",
    }
    assert all(figure.exists() for figure in figures)


def test_generate_baseline_figures_raises_without_metrics(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No baseline metric CSV"):
        generate_baseline_figures(
            data_metrics_path=tmp_path / "missing_data.csv",
            environment_metrics_path=tmp_path / "missing_env.csv",
            output_dir=tmp_path / "figures",
        )


def test_generate_behavior_figures_writes_qualitative_pngs(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    episodes_path = tmp_path / "dqn_episode_summary.csv"
    steps_path = tmp_path / "dqn_step_log.csv"
    output_dir = tmp_path / "behavior_figures"

    pd.DataFrame(
        [
            {
                "episode_id": 0,
                "cell_line_id": "ACH-1",
                "selected_gene": "A",
                "dependency_regret": 0.0,
                "hit_at_k": 1.0,
                "n_queries": 1,
            },
            {
                "episode_id": 1,
                "cell_line_id": "ACH-2",
                "selected_gene": "B",
                "dependency_regret": 1.2,
                "hit_at_k": 0.0,
                "n_queries": 3,
            },
        ]
    ).to_csv(episodes_path, index=False)
    pd.DataFrame(
        [
            {
                "episode_id": 0,
                "cell_line_id": "ACH-1",
                "step": 0,
                "action_type": "query",
                "gene": "A",
                "modality": "expression",
                "observed_value": 1.0,
                "gene_true_rank": 1,
            },
            {
                "episode_id": 0,
                "cell_line_id": "ACH-1",
                "step": 1,
                "action_type": "select",
                "gene": "A",
                "modality": None,
                "observed_value": None,
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
    ).to_csv(steps_path, index=False)

    figures = generate_behavior_figures(episodes_path, steps_path, output_dir)

    assert {figure.name for figure in figures} == {
        "dqn_regret_vs_queries.png",
        "dqn_query_efficiency_by_true_rank.png",
        "dqn_example_trajectories.png",
    }
    assert all(figure.exists() for figure in figures)


def test_generate_behavior_figures_writes_context_heatmap(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    episodes_path = tmp_path / "dqn_episode_summary.csv"
    steps_path = tmp_path / "dqn_step_log.csv"
    context_path = tmp_path / "dqn_modality_usage_by_context.csv"
    output_dir = tmp_path / "behavior_figures"

    pd.DataFrame(
        [
            {
                "episode_id": 0,
                "cell_line_id": "ACH-1",
                "selected_gene": "A",
                "dependency_regret": 0.0,
                "hit_at_k": 1.0,
                "n_queries": 1,
            }
        ]
    ).to_csv(episodes_path, index=False)
    pd.DataFrame(
        [
            {
                "episode_id": 0,
                "cell_line_id": "ACH-1",
                "step": 0,
                "action_type": "query",
                "gene": "A",
                "modality": "expression",
                "observed_value": 1.0,
                "gene_true_rank": 1,
            },
            {
                "episode_id": 0,
                "cell_line_id": "ACH-1",
                "step": 1,
                "action_type": "select",
                "gene": "A",
                "modality": None,
                "observed_value": None,
                "gene_true_rank": 1,
            },
        ]
    ).to_csv(steps_path, index=False)
    pd.DataFrame(
        [
            {
                "OncotreeLineage": "Lung",
                "n_query_expression": 4.0,
                "n_query_cna": 2.0,
                "n_episodes": 8,
            },
            {
                "OncotreeLineage": "Breast",
                "n_query_expression": 1.5,
                "n_query_cna": 5.5,
                "n_episodes": 4,
            },
        ]
    ).to_csv(context_path, index=False)

    figures = generate_behavior_figures(
        episodes_path,
        steps_path,
        output_dir,
        context_usage_path=context_path,
    )

    assert "dqn_modality_usage_by_context.png" in {figure.name for figure in figures}
    assert all(figure.exists() for figure in figures)
