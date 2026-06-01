from __future__ import annotations

import pandas as pd
import pytest

from src.environment import ActionType, EvidenceAcquisitionEnv, EvidenceAction
from src.episodes import CandidateEpisode


def _episode() -> CandidateEpisode:
    return CandidateEpisode(
        episode_id=7,
        cell_line_id="ACH-1",
        candidate_genes=("A", "B"),
        dependency_scores=(-1.25, 0.5),
    )


def _modalities() -> dict[str, pd.DataFrame]:
    return {
        "expression": pd.DataFrame({"A": [2.0], "B": [5.0]}, index=["ACH-1"]),
        "cna": pd.DataFrame({"A": [0.1], "B": [-0.2]}, index=["ACH-1"]),
    }


def test_reset_returns_fully_masked_initial_state() -> None:
    env = EvidenceAcquisitionEnv(_modalities())

    state = env.reset(_episode())

    assert state.episode_id == 7
    assert state.cell_line_id == "ACH-1"
    assert state.candidate_genes == ("A", "B")
    assert state.modality_names == ("expression", "cna")
    assert state.observed_values == ((None, None), (None, None))
    assert state.query_mask == ((False, False), (False, False))
    assert not state.done


def test_query_action_reveals_value_and_charges_modality_cost() -> None:
    env = EvidenceAcquisitionEnv(_modalities(), query_costs={"expression": 2.5, "cna": 1.0})
    env.reset(_episode())

    result = env.step(env.query_action(gene_index=1, modality_index=0))

    assert result.reward == -2.5
    assert not result.done
    assert result.state.observed_values == ((None, None), (5.0, None))
    assert result.state.query_mask == ((False, False), (True, False))
    assert result.info["modality"] == "expression"


def test_available_actions_excludes_queried_pairs_but_keeps_select_actions() -> None:
    env = EvidenceAcquisitionEnv(_modalities())
    env.reset(_episode())
    env.step(env.query_action(gene_index=0, modality_index=1))

    actions = env.available_actions()

    assert env.query_action(0, 1) not in actions
    assert env.query_action(0, 0) in actions
    assert env.select_action(0) in actions
    assert env.select_action(1) in actions


def test_select_action_terminates_with_dependency_reward() -> None:
    env = EvidenceAcquisitionEnv(_modalities())
    env.reset(_episode())

    result = env.step(env.select_action(gene_index=0))

    assert result.done
    assert result.reward == 1.25
    assert result.state.done
    assert result.state.selected_gene == "A"
    assert result.info["selected_dependency"] == -1.25
    assert env.available_actions() == ()


def test_select_action_scales_dependency_reward() -> None:
    env = EvidenceAcquisitionEnv(_modalities(), selection_reward_scale=1.5)
    env.reset(_episode())

    result = env.step(env.select_action(gene_index=0))

    assert result.reward == pytest.approx(1.875)


def test_repeated_query_does_not_change_state_and_gets_penalty() -> None:
    env = EvidenceAcquisitionEnv(_modalities(), repeated_query_penalty=3.0)
    env.reset(_episode())
    env.step(env.query_action(gene_index=0, modality_index=0))

    result = env.step(env.query_action(gene_index=0, modality_index=0))

    assert result.reward == -4.0
    assert result.info["repeated"] is True
    assert result.state.observed_values == ((2.0, None), (None, None))


def test_step_after_done_raises() -> None:
    env = EvidenceAcquisitionEnv(_modalities())
    env.reset(_episode())
    env.step(env.select_action(gene_index=0))

    with pytest.raises(RuntimeError, match="after it is done"):
        env.step(env.select_action(gene_index=1))


def test_query_action_requires_modality_index() -> None:
    env = EvidenceAcquisitionEnv(_modalities())
    env.reset(_episode())

    with pytest.raises(ValueError, match="modality_index"):
        env.step(EvidenceAction(ActionType.QUERY, gene_index=0))
