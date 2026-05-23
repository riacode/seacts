from __future__ import annotations

import numpy as np
import pandas as pd

from src.environment import ActionType, EvidenceAcquisitionEnv
from src.episodes import CandidateEpisode
from src.state_encoder import StateEncoder


def _episode() -> CandidateEpisode:
    return CandidateEpisode(
        episode_id=0,
        cell_line_id="ACH-1",
        candidate_genes=("A", "B"),
        dependency_scores=(-1.0, 0.2),
    )


def _env() -> EvidenceAcquisitionEnv:
    modalities = {
        "expression": pd.DataFrame({"A": [3.0], "B": [1.0]}, index=["ACH-1"]),
        "cna": pd.DataFrame({"A": [0.5], "B": [-0.5]}, index=["ACH-1"]),
    }
    return EvidenceAcquisitionEnv(modalities)


def test_action_space_round_trips_query_and_select_actions() -> None:
    env = _env()
    env.reset(_episode())
    encoder = StateEncoder(n_genes=2, n_modalities=2)

    query_index = encoder.action_space.to_index(env.query_action(1, 0))
    select_index = encoder.action_space.to_index(env.select_action(1))

    assert query_index == 2
    assert select_index == 5
    assert encoder.action_space.from_index(query_index) == env.query_action(1, 0)
    assert encoder.action_space.from_index(select_index) == env.select_action(1)


def test_valid_action_mask_excludes_observed_queries_and_keeps_selects() -> None:
    env = _env()
    state = env.reset(_episode())
    encoder = StateEncoder(n_genes=2, n_modalities=2)

    initial_mask = encoder.valid_action_mask(state)
    assert initial_mask.tolist() == [True, True, True, True, True, True]

    state = env.step(env.query_action(0, 1)).state
    updated_mask = encoder.valid_action_mask(state)

    assert updated_mask.tolist() == [True, False, True, True, True, True]


def test_encode_uses_zero_for_unobserved_values_and_masks_observations() -> None:
    env = _env()
    state = env.reset(_episode())
    state = env.step(env.query_action(0, 0)).state
    encoder = StateEncoder(n_genes=2, n_modalities=2)

    vector = encoder.encode(state)

    assert vector.dtype == np.float32
    assert len(vector) == encoder.state_size
    assert vector.tolist() == [
        3.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]


def test_done_state_has_no_valid_actions() -> None:
    env = _env()
    env.reset(_episode())
    state = env.step(env.select_action(0)).state
    encoder = StateEncoder(n_genes=2, n_modalities=2)

    assert not encoder.valid_action_mask(state).any()
    assert encoder.action_space.from_index(4).action_type == ActionType.SELECT
