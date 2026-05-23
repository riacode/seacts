from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.visualization import generate_baseline_figures


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
                "policy": "rl_env_query_expression_then_select",
                "selected_dependency": -0.7,
                "hit_at_k": 0.8,
                "ndcg_at_k": 0.5,
                "mrr_at_k": 0.4,
                "query_cost": 0.32,
                "n_queries": 16,
                "total_reward": 0.38,
            }
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
