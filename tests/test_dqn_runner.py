from __future__ import annotations

import pandas as pd
import pytest

from src.dqn import DQNHyperparameters, optimize_dqn_batch
from src.environment import EvidenceAcquisitionEnv
from src.episodes import CandidateEpisode
from src.replay_buffer import Transition
from src.rl_runner import evaluate_dqn_agent, train_dqn_agent
from src.state_encoder import StateEncoder


def _episodes() -> list[CandidateEpisode]:
    return [
        CandidateEpisode(
            episode_id=0,
            cell_line_id="ACH-1",
            candidate_genes=("A", "B"),
            dependency_scores=(-1.0, 0.2),
        ),
        CandidateEpisode(
            episode_id=1,
            cell_line_id="ACH-2",
            candidate_genes=("A", "B"),
            dependency_scores=(-0.8, 0.1),
        ),
    ]


def _env() -> EvidenceAcquisitionEnv:
    modalities = {
        "expression": pd.DataFrame(
            {"A": [4.0, 3.0], "B": [1.0, 0.5]},
            index=["ACH-1", "ACH-2"],
        )
    }
    return EvidenceAcquisitionEnv(modalities, query_costs={"expression": 0.1})


def test_train_and_evaluate_dqn_on_tiny_environment() -> None:
    pytest.importorskip("torch")

    env = _env()
    encoder = StateEncoder(n_genes=2, n_modalities=1)
    network, history = train_dqn_agent(
        env=env,
        episodes=_episodes(),
        encoder=encoder,
        hyperparameters=DQNHyperparameters(
            hidden_dim=16,
            batch_size=2,
            replay_capacity=20,
            target_update_steps=2,
            max_steps_per_episode=3,
            epsilon_decay_steps=1,
        ),
        seed=0,
    )

    results = evaluate_dqn_agent(
        q_network=network,
        env=env,
        episodes=_episodes(),
        encoder=encoder,
        top_k=1,
        max_steps_per_episode=3,
    )

    assert len(history) == 2
    assert results.loc[0, "policy"] == "rl_env_dqn"
    assert results.loc[0, "n_queries"] >= 0
    assert "n_query_expression" in results.columns


def test_optimize_dqn_batch_uses_double_dqn_target_selection() -> None:
    torch = pytest.importorskip("torch")

    class ConstantQ(torch.nn.Module):
        def __init__(self, values: list[float]) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))

        def forward(self, states):
            return self.bias.unsqueeze(0).repeat(states.shape[0], 1)

    q_network = ConstantQ([0.0, 10.0])
    target_network = ConstantQ([100.0, 1.0])
    optimizer = torch.optim.SGD(q_network.parameters(), lr=0.0)
    transition = Transition(
        state=pd.Series([0.0], dtype="float32").to_numpy(),
        action=0,
        reward=0.0,
        next_state=pd.Series([0.0], dtype="float32").to_numpy(),
        next_valid_actions=pd.Series([True, True]).to_numpy(),
        done=False,
    )

    loss = optimize_dqn_batch(
        q_network=q_network,
        target_network=target_network,
        optimizer=optimizer,
        transitions=[transition],
        gamma=1.0,
    )

    assert loss == pytest.approx(0.5)
