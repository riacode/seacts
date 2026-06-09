from __future__ import annotations

import sys
from pathlib import Path
from random import Random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.context_encoding import LineageContextEncoder
from src.dqn import (
    ContextSelectStructuredQNetwork,
    ContextStructuredQNetwork,
    DQNHyperparameters,
    StructuredQNetwork,
    build_q_network,
    forward_q_network,
    load_structured_checkpoint_into_context,
    optimize_dqn_batch,
    requires_context_indices,
    select_epsilon_greedy_action,
    select_greedy_action,
    uses_cancer_context,
)
from src.environment import EvidenceAcquisitionEnv
from src.episodes import CandidateEpisode
from src.replay_buffer import ReplayBuffer, Transition
from src.rl_runner import (
    RLTrainingConfig,
    _apply_min_query_constraint,
    _log_dqn_to_wandb,
    _wandb_training_logger,
    collect_dqn_behavior_log,
    collect_dqn_trajectory_metrics,
    evaluate_dqn_agent,
    seed_replay_with_expert,
    seed_replay_with_modality_expert,
    train_dqn_agent,
)
from src.splits import split_dependency_by_cell_line
from src.tracking import log_dqn_behavior_results
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


def test_context_structured_q_network_forward() -> None:
    pytest.importorskip("torch")
    import torch

    encoder = StateEncoder(n_genes=2, n_modalities=1)
    network = build_q_network(
        encoder.state_size,
        encoder.action_space.size,
        hidden_dim=16,
        network_type="context_structured",
        n_genes=2,
        n_modalities=1,
        n_lineages=3,
        cancer_context_dim=8,
    )
    assert isinstance(network, ContextStructuredQNetwork)
    state = torch.zeros(1, encoder.state_size)
    context = torch.tensor([1], dtype=torch.int64)
    q_values = forward_q_network(network, state, context)
    assert q_values.shape == (1, encoder.action_space.size)


def test_context_structured_dueling_forward() -> None:
    pytest.importorskip("torch")
    import torch

    encoder = StateEncoder(n_genes=2, n_modalities=1)
    network = build_q_network(
        encoder.state_size,
        encoder.action_space.size,
        hidden_dim=16,
        network_type="context_structured_dueling",
        n_genes=2,
        n_modalities=1,
        n_lineages=3,
        cancer_context_dim=8,
    )
    assert network.dueling is True
    state = torch.zeros(1, encoder.state_size)
    context = torch.tensor([1], dtype=torch.int64)
    q_values = forward_q_network(network, state, context)
    assert q_values.shape == (1, encoder.action_space.size)


def test_load_structured_checkpoint_into_context(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    import torch

    n_genes = 2
    n_modalities = 1
    hidden_dim = 16
    structured = StructuredQNetwork(
        n_genes=n_genes,
        n_modalities=n_modalities,
        hidden_dim=hidden_dim,
        dueling=False,
    )
    checkpoint_path = tmp_path / "structured.pt"
    torch.save(structured.state_dict(), checkpoint_path)

    context = ContextStructuredQNetwork(
        n_genes=n_genes,
        n_modalities=n_modalities,
        hidden_dim=hidden_dim,
        n_lineages=4,
        context_dim=8,
        dueling=False,
    )
    loaded = load_structured_checkpoint_into_context(context, checkpoint_path)
    assert "query_head.weight" in loaded
    assert "candidate_encoder.0.weight" in loaded
    assert torch.allclose(context.query_head.weight, structured.query_head.weight)


def test_uses_cancer_context_includes_dueling() -> None:
    assert uses_cancer_context("context_structured")
    assert uses_cancer_context("context_structured_dueling")
    assert uses_cancer_context("context_select_structured")
    assert uses_cancer_context("context_fusion_structured")
    assert not uses_cancer_context("structured")


def test_context_select_structured_q_network_forward() -> None:
    pytest.importorskip("torch")
    import torch

    encoder = StateEncoder(n_genes=2, n_modalities=2)
    network = build_q_network(
        encoder.state_size,
        encoder.action_space.size,
        hidden_dim=16,
        network_type="context_select_structured",
        n_genes=2,
        n_modalities=2,
        n_lineages=3,
        cancer_context_dim=8,
    )
    assert isinstance(network, ContextSelectStructuredQNetwork)
    assert requires_context_indices(network)
    state = torch.zeros(1, encoder.state_size)
    context = torch.tensor([1], dtype=torch.int64)
    q_values = forward_q_network(network, state, context)
    assert q_values.shape == (1, encoder.action_space.size)


def test_load_structured_checkpoint_into_context_select_only(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    import torch

    n_genes = 2
    n_modalities = 2
    hidden_dim = 16
    structured = StructuredQNetwork(
        n_genes=n_genes,
        n_modalities=n_modalities,
        hidden_dim=hidden_dim,
        dueling=False,
    )
    checkpoint_path = tmp_path / "structured.pt"
    torch.save(structured.state_dict(), checkpoint_path)

    context = ContextSelectStructuredQNetwork(
        n_genes=n_genes,
        n_modalities=n_modalities,
        hidden_dim=hidden_dim,
        n_lineages=4,
        context_dim=8,
    )
    loaded = load_structured_checkpoint_into_context(context, checkpoint_path)
    assert "query_head.weight" in loaded
    assert "select_head.weight" not in loaded
    assert torch.allclose(context.query_head.weight, structured.query_head.weight)
    assert context.select_head.weight.shape != structured.select_head.weight.shape


def test_context_fusion_structured_q_network_forward() -> None:
    pytest.importorskip("torch")
    import torch

    encoder = StateEncoder(n_genes=2, n_modalities=2)
    network = build_q_network(
        encoder.state_size,
        encoder.action_space.size,
        hidden_dim=16,
        network_type="context_fusion_structured",
        n_genes=2,
        n_modalities=2,
        n_lineages=4,
        cancer_context_dim=8,
    )
    state = torch.randn(3, encoder.state_size)
    context = torch.as_tensor([0, 1, 2], dtype=torch.int64)
    q_values = forward_q_network(network, state, context)
    assert q_values.shape == (3, encoder.action_space.size)


def test_generate_context_dqn_modality_figure(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from src.visualization import generate_context_dqn_modality_figure

    context = pd.DataFrame(
        [
            {
                "total_reward": 1.02,
                "n_query_expression": 8.0,
                "n_query_cna": 4.0,
                "n_query_damaging_mutation": 0.2,
                "n_query_hotspot_mutation": 0.1,
            }
        ]
    )
    structured = pd.DataFrame(
        [
            {
                "total_reward": 1.04,
                "n_query_expression": 12.0,
                "n_query_cna": 0.0,
                "n_query_damaging_mutation": 0.5,
                "n_query_hotspot_mutation": 0.0,
            }
        ]
    )
    path = generate_context_dqn_modality_figure(
        context,
        tmp_path,
        variant="ctx_larger_init_structured",
        structured_metrics=structured,
    )
    assert path is not None
    assert path.name == "context_dqn_modality_usage.png"
    assert path.exists()


def test_train_context_structured_dqn_with_lineage_context(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    metadata_path = tmp_path / "Model.csv"
    pd.DataFrame(
        {
            "ModelID": ["ACH-1", "ACH-2"],
            "OncotreeLineage": ["Lung", "Lymphoid"],
        }
    ).to_csv(metadata_path, index=False)
    context_encoder = LineageContextEncoder(metadata_path)

    env = _env()
    encoder = StateEncoder(n_genes=2, n_modalities=1)
    hyperparameters = DQNHyperparameters(
        hidden_dim=16,
        batch_size=2,
        replay_capacity=20,
        learning_starts=2,
        train_frequency=1,
        target_update_steps=2,
        max_steps_per_episode=3,
        epsilon_decay_steps=1,
        q_network_type="context_structured",
        n_genes=2,
        n_modalities=1,
        n_lineages=context_encoder.n_lineages,
        cancer_context_dim=8,
    )
    network, history = train_dqn_agent(
        env=env,
        episodes=_episodes(),
        encoder=encoder,
        hyperparameters=hyperparameters,
        seed=0,
        context_encoder=context_encoder,
    )
    results = evaluate_dqn_agent(
        q_network=network,
        env=env,
        episodes=_episodes(),
        encoder=encoder,
        top_k=1,
        max_steps_per_episode=3,
        context_encoder=context_encoder,
    )
    assert len(history) == 2
    assert results.loc[0, "policy"] == "rl_env_dqn"
    action = select_greedy_action(
        network,
        encoder.encode(env.reset(_episodes()[0])),
        encoder.valid_action_mask(env.state),
        context_index=context_encoder.encode("ACH-1"),
    )
    assert action >= 0


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


def test_collect_dqn_trajectory_metrics_returns_per_episode_rows() -> None:
    pytest.importorskip("torch")

    env = _env()
    encoder = StateEncoder(n_genes=2, n_modalities=1)
    network, _ = train_dqn_agent(
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

    trajectories = collect_dqn_trajectory_metrics(
        q_network=network,
        env=env,
        episodes=_episodes(),
        encoder=encoder,
        max_steps_per_episode=3,
    )

    assert len(trajectories) == 2
    assert {
        "episode_id",
        "cell_line_id",
        "selected_gene",
        "n_queries",
        "n_query_expression",
    }.issubset(trajectories.columns)


def test_collect_dqn_behavior_log_records_step_actions() -> None:
    pytest.importorskip("torch")

    env = _env()
    encoder = StateEncoder(n_genes=2, n_modalities=1)
    network, _ = train_dqn_agent(
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

    episodes_df, steps_df = collect_dqn_behavior_log(
        q_network=network,
        env=env,
        episodes=_episodes(),
        encoder=encoder,
        top_k=1,
        max_steps_per_episode=3,
    )

    assert len(episodes_df) == 2
    assert not steps_df.empty
    assert {"hit_at_k", "dependency_regret"}.issubset(episodes_df.columns)
    assert {"step", "action_type", "gene", "modality", "gene_true_rank"}.issubset(steps_df.columns)


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


def test_optimize_dqn_batch_uses_n_step_discount() -> None:
    torch = pytest.importorskip("torch")

    class ConstantQ(torch.nn.Module):
        def __init__(self, values: list[float]) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))

        def forward(self, states):
            return self.bias.unsqueeze(0).repeat(states.shape[0], 1)

    q_network = ConstantQ([0.0, 0.0])
    target_network = ConstantQ([10.0, 1.0])
    optimizer = torch.optim.SGD(q_network.parameters(), lr=0.0)
    transition = Transition(
        state=np.array([0.0], dtype=np.float32),
        action=0,
        reward=1.0,
        next_state=np.array([0.0], dtype=np.float32),
        next_valid_actions=np.array([True, True]),
        done=False,
        n_steps=3,
    )

    loss = optimize_dqn_batch(
        q_network=q_network,
        target_network=target_network,
        optimizer=optimizer,
        transitions=[transition],
        gamma=0.5,
    )

    assert loss == pytest.approx(1.75)


def test_build_q_network_supports_structured_and_dueling_variants() -> None:
    torch = pytest.importorskip("torch")
    encoder = StateEncoder(n_genes=2, n_modalities=2)

    for network_type in ("dueling_mlp", "structured", "structured_dueling"):
        network = build_q_network(
            state_size=encoder.state_size,
            action_size=encoder.action_space.size,
            hidden_dim=16,
            network_type=network_type,
            n_genes=encoder.n_genes,
            n_modalities=encoder.n_modalities,
        )
        output = network(torch.zeros((3, encoder.state_size), dtype=torch.float32))
        assert tuple(output.shape) == (3, encoder.action_space.size)


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

    assert config.train_episodes == 10000
    assert config.learning_rate == 0.0001
    assert config.learning_starts == 500
    assert config.train_frequency == 4
    assert config.target_update_steps == 500
    assert config.max_grad_norm == 10.0
    assert config.epsilon_decay_steps == 20000
    assert config.validation_interval == 100
    assert config.select_exploration_probability == 0.25
    assert config.expert_seed_episodes == 1000
    assert config.expert_seed_modality == "expression"
    assert config.expert_seed_strategy == "single_modality"
    assert config.expert_seed_refinement_modality == "cna"
    assert config.expert_seed_refinement_top_k == 4
    assert config.min_queries_before_select == 0
    assert config.n_step_returns == 1
    assert config.q_network_type == "mlp"
    assert config.wandb_log_interval == 25
    assert not config.split_cell_lines
    assert config.validation_cell_line_fraction == 0.1
    assert config.eval_cell_line_fraction == 0.1


def test_split_dependency_by_cell_line_creates_disjoint_partitions() -> None:
    config = SimpleNamespace(
        seed=0,
        episodes=SimpleNamespace(
            dependency_threshold=-0.5,
            candidates_per_episode=4,
            positives_per_episode=1,
            min_candidates_per_cell_line=4,
        ),
    )
    dependency = pd.DataFrame(
        {
            "A": [-1.0] * 10,
            "B": [0.1] * 10,
            "C": [0.2] * 10,
            "D": [0.3] * 10,
        },
        index=[f"ACH-{index}" for index in range(10)],
    )

    train, validation, eval_ = split_dependency_by_cell_line(
        config,
        dependency,
        seed=7,
        validation_fraction=0.2,
        eval_fraction=0.2,
    )

    assert set(train.index).isdisjoint(validation.index)
    assert set(train.index).isdisjoint(eval_.index)
    assert set(validation.index).isdisjoint(eval_.index)
    assert len(train) == 6
    assert len(validation) == 2
    assert len(eval_) == 2


def test_apply_min_query_constraint_masks_select_actions_until_threshold() -> None:
    env = _env()
    state = env.reset(_episodes()[0])
    encoder = StateEncoder(n_genes=2, n_modalities=1)

    constrained = _apply_min_query_constraint(
        encoder.valid_action_mask(state),
        state,
        encoder,
        min_queries_before_select=1,
    )

    assert constrained.tolist() == [True, True, False, False]

    state = env.step(env.query_action(0, 0)).state
    unconstrained = _apply_min_query_constraint(
        encoder.valid_action_mask(state),
        state,
        encoder,
        min_queries_before_select=1,
    )

    assert unconstrained.tolist() == [False, True, True, True]


def test_seed_replay_with_modality_expert_adds_query_and_select_transitions() -> None:
    env = _env()
    encoder = StateEncoder(n_genes=2, n_modalities=1)
    replay = ReplayBuffer(capacity=10, seed=0)

    n_transitions = seed_replay_with_modality_expert(
        replay=replay,
        env=env,
        episodes=[_episodes()[0]],
        encoder=encoder,
        modality_name="expression",
    )

    assert n_transitions == 3
    assert len(replay) == 3
    transitions = replay.sample(3)
    assert sum(transition.done for transition in transitions) == 1


def test_seed_replay_with_refinement_expert_queries_screen_then_top_k_refinement() -> None:
    modalities = {
        "expression": pd.DataFrame({"A": [4.0], "B": [1.0], "C": [3.0]}, index=["ACH-1"]),
        "cna": pd.DataFrame({"A": [0.0], "B": [5.0], "C": [2.0]}, index=["ACH-1"]),
    }
    episode = CandidateEpisode(
        episode_id=0,
        cell_line_id="ACH-1",
        candidate_genes=("A", "B", "C"),
        dependency_scores=(-1.0, 0.2, -0.5),
    )
    env = EvidenceAcquisitionEnv(modalities, query_costs={"expression": 0.02, "cna": 0.02})
    encoder = StateEncoder(n_genes=3, n_modalities=2)
    replay = ReplayBuffer(capacity=20, seed=0)

    n_transitions = seed_replay_with_expert(
        replay=replay,
        env=env,
        episodes=[episode],
        encoder=encoder,
        strategy="expression_full_then_refinement_top_k",
        modality_name="expression",
        refinement_modality_name="cna",
        refinement_top_k=2,
    )

    assert n_transitions == 6
    assert len(replay) == 6
    assert env.state.done
    assert env.state.query_mask == (
        (True, True),
        (True, False),
        (True, True),
    )


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
        rl_config=RLTrainingConfig(train_episodes=4, wandb_log_interval=2),
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
    assert summary == {
        "train/final_total_reward": 0.4,
        "train/final_n_queries": 4,
        "train/final_epsilon": 0.7,
        "train/final_loss": 0.5,
        "eval/total_reward": 0.5,
    }


def test_wandb_training_logger_logs_validation_off_train_log_interval() -> None:
    logged: list[tuple[dict, int | None]] = []

    class FakeRun:
        def log(self, data, step=None):
            logged.append((data, step))

    logger = _wandb_training_logger(
        FakeRun(),
        RLTrainingConfig(wandb_log_interval=25, train_episodes=10_000, validation_interval=100),
    )
    logger(
        {
            "episode": 99,
            "total_reward": 0.5,
            "n_queries": 5,
            "epsilon": 0.9,
            "loss": 0.1,
            "validation_total_reward": 0.8,
            "validation_n_queries": 9.0,
            "validation_selected_dependency": -1.0,
            "validation_hit_at_k": 0.95,
        }
    )

    assert logged == [
        (
            {
                "validation/total_reward": 0.8,
                "validation/n_queries": 9.0,
                "validation/selected_dependency": -1.0,
                "validation/hit_at_k": 0.95,
            },
            99,
        )
    ]


def test_log_dqn_behavior_results_logs_analysis_tables(monkeypatch) -> None:
    logged: list[dict] = []
    summary: dict[str, float] = {}

    class FakeRun:
        def __init__(self) -> None:
            self.summary = summary

        def log(self, data):
            logged.append(data)

    fake_wandb = SimpleNamespace(
        Table=lambda **kwargs: {
            "table": len(kwargs["dataframe"]) if "dataframe" in kwargs else len(kwargs["data"])
        },
        Histogram=lambda values: {"histogram": len(values)},
        plot=SimpleNamespace(
            bar=lambda *args, **kwargs: {"bar": kwargs.get("title")},
            histogram=lambda *args, **kwargs: {"histogram_plot": kwargs.get("title")},
        ),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    episodes = pd.DataFrame(
        [
            {
                "hit_at_k": 1.0,
                "n_queries": 2,
                "dependency_regret": 0.0,
                "n_query_expression": 2,
            }
        ]
    )
    steps = pd.DataFrame(
        [
            {
                "action_type": "query",
                "step": 0,
                "gene_true_rank": 1,
            }
        ]
    )
    analysis = {"dqn_failure_cases": pd.DataFrame([{"episode_id": 1}])}

    log_dqn_behavior_results(FakeRun(), episodes, steps, analysis)

    assert logged[0]["behavior_analysis/dqn_failure_cases"] == {"table": 1}
    assert summary["behavior/hit_rate"] == 1.0
