from __future__ import annotations

import sys
from pathlib import Path
from random import Random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.dqn import DQNHyperparameters, optimize_dqn_batch, select_epsilon_greedy_action
from src.environment import EvidenceAcquisitionEnv
from src.episodes import CandidateEpisode
from src.replay_buffer import Transition
from src.rl_runner import RLTrainingConfig, _log_dqn_to_wandb, evaluate_dqn_agent, train_dqn_agent
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
            learning_starts=2,
            train_frequency=1,
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


def test_epsilon_exploration_can_prioritize_select_actions() -> None:
    torch = pytest.importorskip("torch")
    network = torch.nn.Linear(1, 4)

    action = select_epsilon_greedy_action(
        q_network=network,
        state=np.array([0.0], dtype=np.float32),
        valid_actions=np.array([True, True, True, True]),
        epsilon=1.0,
        rng=Random(0),
        select_action_indices=np.array([2, 3]),
        select_exploration_probability=1.0,
    )

    assert action in {2, 3}


def test_rl_training_config_uses_stable_dqn_defaults() -> None:
    config = RLTrainingConfig()

    assert config.learning_rate == 0.0001
    assert config.learning_starts == 500
    assert config.train_frequency == 4
    assert config.target_update_steps == 500
    assert config.max_grad_norm == 10.0
    assert config.validation_interval == 50


def test_wandb_logging_records_training_history_steps(monkeypatch, tmp_path: Path) -> None:
    logged: list[tuple[dict, int | None]] = []
    summary: dict[str, float] = {}

    class FakeRun:
        def __init__(self) -> None:
            self.summary = summary

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def log(self, data, step=None):
            logged.append((data, step))

        def save(self, path):
            logged.append(({"saved": path}, None))

    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: FakeRun(),
        Table=lambda dataframe: {"rows": len(dataframe)},
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    config = SimpleNamespace(
        tracking=SimpleNamespace(
            wandb=SimpleNamespace(enabled=True, entity="seacts", project="seacts")
        )
    )

    _log_dqn_to_wandb(
        config=config,
        config_path="config.yaml",
        rl_config=RLTrainingConfig(wandb_log_interval=2),
        results=pd.DataFrame([{"policy": "rl_env_dqn", "total_reward": 0.5}]),
        training_history=[
            {"episode": 0, "total_reward": 0.1, "n_queries": 1, "epsilon": 1.0, "loss": 0.0},
            {"episode": 1, "total_reward": 0.2, "n_queries": 2, "epsilon": 0.9, "loss": 0.3},
            {"episode": 2, "total_reward": 0.3, "n_queries": 3, "epsilon": 0.8, "loss": 0.4},
            {"episode": 3, "total_reward": 0.4, "n_queries": 4, "epsilon": 0.7, "loss": 0.5},
        ],
        output_path=tmp_path / "dqn_eval_metrics.csv",
        model_path=tmp_path / "dqn_policy.pt",
    )

    train_logs = [item for item in logged if "train/total_reward" in item[0]]
    assert train_logs == [
        (
            {
                "train/total_reward": 0.1,
                "train/n_queries": 1,
                "train/epsilon": 1.0,
                "train/loss": 0.0,
            },
            0,
        ),
        (
            {
                "train/total_reward": 0.3,
                "train/n_queries": 3,
                "train/epsilon": 0.8,
                "train/loss": 0.4,
            },
            2,
        ),
        (
            {
                "train/total_reward": 0.4,
                "train/n_queries": 4,
                "train/epsilon": 0.7,
                "train/loss": 0.5,
            },
            3,
        ),
    ]
    assert summary == {"eval/total_reward": 0.5}
