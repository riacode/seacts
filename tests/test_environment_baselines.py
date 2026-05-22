from __future__ import annotations

import pandas as pd
import pytest

from src.environment import EvidenceAcquisitionEnv
from src.environment_baselines import (
    OracleSelectPolicy,
    QueryAllAveragePolicy,
    QueryModalityPolicy,
    RandomSelectPolicy,
    build_environment_policies,
    evaluate_environment_policy,
)
from src.episodes import CandidateEpisode


def _episode() -> CandidateEpisode:
    return CandidateEpisode(
        episode_id=0,
        cell_line_id="ACH-1",
        candidate_genes=("A", "B", "C"),
        dependency_scores=(-1.0, 0.1, 0.2),
    )


def _env() -> EvidenceAcquisitionEnv:
    modalities = {
        "expression": pd.DataFrame({"A": [4.0], "B": [2.0], "C": [1.0]}, index=["ACH-1"]),
        "cna": pd.DataFrame({"A": [0.0], "B": [3.0], "C": [1.0]}, index=["ACH-1"]),
    }
    return EvidenceAcquisitionEnv(modalities)


def _costed_env() -> EvidenceAcquisitionEnv:
    modalities = {
        "expression": pd.DataFrame({"A": [4.0], "B": [2.0], "C": [1.0]}, index=["ACH-1"]),
        "cna": pd.DataFrame({"A": [0.0], "B": [3.0], "C": [1.0]}, index=["ACH-1"]),
    }
    return EvidenceAcquisitionEnv(modalities, query_costs={"expression": 0.2, "cna": 0.5})


def test_query_modality_policy_queries_one_modality_for_all_candidates() -> None:
    episode = _episode()
    rollout = QueryModalityPolicy("expression").run(_env(), episode)

    assert rollout.ranked_indices == (0, 1, 2)
    assert rollout.selected_index == 0
    assert rollout.selected_dependency == -1.0
    assert rollout.query_cost == 3.0
    assert rollout.n_queries == 3
    assert rollout.total_reward == -2.0


def test_query_modality_policy_uses_configured_query_costs() -> None:
    episode = _episode()
    rollout = QueryModalityPolicy("expression").run(_costed_env(), episode)

    assert rollout.query_cost == pytest.approx(0.6)
    assert rollout.n_queries == 3
    assert rollout.total_reward == pytest.approx(0.4)


def test_query_all_average_policy_queries_every_pair() -> None:
    episode = _episode()
    rollout = QueryAllAveragePolicy().run(_env(), episode)

    assert rollout.selected_index in {0, 1}
    assert rollout.query_cost == 6.0
    assert rollout.n_queries == 6


def test_query_all_average_policy_uses_modality_specific_costs() -> None:
    episode = _episode()
    rollout = QueryAllAveragePolicy().run(_costed_env(), episode)

    assert rollout.query_cost == pytest.approx(2.1)
    assert rollout.n_queries == 6


def test_oracle_policy_selects_most_dependent_without_queries() -> None:
    episode = _episode()
    rollout = OracleSelectPolicy().run(_env(), episode)

    assert rollout.ranked_indices == (0, 1, 2)
    assert rollout.selected_index == 0
    assert rollout.query_cost == 0.0
    assert rollout.total_reward == 1.0


def test_evaluate_environment_policy_reports_metrics() -> None:
    episode = _episode()
    result = evaluate_environment_policy(
        QueryModalityPolicy("expression"),
        _env(),
        [episode],
        top_k=1,
    )

    assert result["policy"] == "rl_env_query_expression_then_select"
    assert result["selected_dependency"] == -1.0
    assert result["hit_at_k"] == 1.0
    assert result["query_cost"] == 3.0
    assert result["n_queries"] == 3


def test_build_environment_policies_matches_modalities() -> None:
    policies = build_environment_policies(("expression", "cna"), seed=224)

    assert [policy.name for policy in policies] == [
        "rl_env_random_select",
        "rl_env_oracle_select",
        "rl_env_query_expression_then_select",
        "rl_env_query_cna_then_select",
        "rl_env_query_all_average_then_select",
    ]
    assert isinstance(policies[0], RandomSelectPolicy)
