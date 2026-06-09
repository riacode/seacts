from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.config import BaselineConfig, load_baseline_config
from src.data import load_project_data
from src.data_baseline_runner import _resolve_data_path, _resolve_optional_data_path
from src.context_encoding import LineageContextEncoder
from src.dqn import (
    ContextStructuredQNetwork,
    DQNHyperparameters,
    _context_tensor,
    build_q_network,
    epsilon_by_step,
    forward_q_network,
    load_structured_checkpoint_into_context,
    optimize_dqn_batch,
    select_epsilon_greedy_action,
    select_greedy_action,
    uses_cancer_context,
)
from src.environment import ActionType, EvidenceAcquisitionEnv
from src.episodes import CandidateEpisode, EpisodeBuilder
from src.metrics import hit_at_k, ndcg_at_k, reciprocal_rank_at_k
from src.modality_scores import (
    build_lineage_supervised_modality_scores,
    build_supervised_modality_scores,
)
from src.replay_buffer import ReplayBuffer, Transition
from src.splits import CellLineSplitConfig, maybe_split_dependency_by_cell_line
from src.state_encoder import StateEncoder


@dataclass(frozen=True)
class RLTrainingConfig:
    train_episodes: int = 10_000
    eval_episodes: int = 500
    validation_episodes: int = 100
    hidden_dim: int = 128
    learning_rate: float = 0.0001
    gamma: float = 0.95
    batch_size: int = 64
    replay_capacity: int = 20_000
    learning_starts: int = 500
    train_frequency: int = 4
    target_update_steps: int = 500
    max_grad_norm: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    max_steps_per_episode: int = 32
    select_exploration_probability: float = 0.25
    validation_interval: int = 100
    expert_seed_episodes: int = 1_000
    expert_seed_modality: str = "expression"
    expert_seed_strategy: str = "single_modality"
    expert_seed_refinement_modality: str = "cna"
    expert_seed_refinement_top_k: int = 4
    min_queries_before_select: int = 0
    n_step_returns: int = 1
    q_network_type: str = "mlp"
    cancer_context_column: str = "OncotreeLineage"
    cancer_context_dim: int = 16
    init_structured_checkpoint: str | None = None
    freeze_shared_heads: bool = False
    use_lineage_modality_scores: bool = False
    lineage_min_samples: int = 6
    query_shaping_alpha: float = 0.0
    fusion_query_boost: float = 2.0
    fusion_select_weight: float = 1.0
    select_residual_weight: float = 0.25
    wandb_log_interval: int = 25
    split_cell_lines: bool = False
    validation_cell_line_fraction: float = 0.1
    eval_cell_line_fraction: float = 0.1

    @property
    def hyperparameters(self) -> DQNHyperparameters:
        return DQNHyperparameters(
            hidden_dim=self.hidden_dim,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            batch_size=self.batch_size,
            replay_capacity=self.replay_capacity,
            learning_starts=self.learning_starts,
            train_frequency=self.train_frequency,
            target_update_steps=self.target_update_steps,
            max_grad_norm=self.max_grad_norm,
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            epsilon_decay_steps=self.epsilon_decay_steps,
            max_steps_per_episode=self.max_steps_per_episode,
            select_exploration_probability=self.select_exploration_probability,
            validation_interval=self.validation_interval,
            expert_seed_episodes=self.expert_seed_episodes,
            expert_seed_modality=self.expert_seed_modality,
            expert_seed_strategy=self.expert_seed_strategy,
            expert_seed_refinement_modality=self.expert_seed_refinement_modality,
            expert_seed_refinement_top_k=self.expert_seed_refinement_top_k,
            min_queries_before_select=self.min_queries_before_select,
            n_step_returns=self.n_step_returns,
            q_network_type=self.q_network_type,
            n_genes=0,
            n_modalities=0,
            n_lineages=0,
            cancer_context_dim=self.cancer_context_dim,
            init_structured_checkpoint=self.init_structured_checkpoint,
            freeze_shared_heads=self.freeze_shared_heads,
            fusion_query_boost=self.fusion_query_boost,
            fusion_select_weight=self.fusion_select_weight,
            select_residual_weight=self.select_residual_weight,
        )


@dataclass(frozen=True)
class DQNRollout:
    ranked_indices: tuple[int, ...]
    selected_index: int
    selected_dependency: float
    query_cost: float
    n_queries: int
    total_reward: float
    modality_query_counts: dict[str, int]
    selected_gene: str
    cell_line_id: str
    episode_id: int


def run_dqn_training_pipeline(
    config_path: str | Path,
    raw_data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    wandb_run_name: str = "dqn-training",
) -> tuple[pd.DataFrame, Path]:
    config = load_baseline_config(config_path)
    rl_config = load_rl_training_config(config_path)
    train_episodes, eval_episodes, validation_episodes, env, encoder, context_encoder = (
        _prepare_dqn_setup(
            config,
            raw_data_dir,
            rl_config,
        )
    )

    resolved_output_dir = Path(output_dir) if output_dir is not None else Path(config.output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    model_path = resolved_output_dir / "dqn_policy.pt"
    training_path = resolved_output_dir / "dqn_training_metrics.csv"
    output_path = resolved_output_dir / "dqn_eval_metrics.csv"
    trajectory_path = resolved_output_dir / "dqn_trajectory_metrics.csv"

    with _wandb_dqn_run(config, config_path, rl_config, output_path, run_name=wandb_run_name) as wandb_run:
        q_network, training_history = train_dqn_agent(
            env=env,
            episodes=train_episodes,
            encoder=encoder,
            hyperparameters=_hyperparameters_with_context(
                rl_config.hyperparameters,
                encoder,
                context_encoder,
            ),
            seed=config.seed,
            validation_episodes=validation_episodes,
            top_k=config.evaluation.top_k,
            context_encoder=context_encoder,
            training_log_callback=_wandb_training_logger(wandb_run, rl_config),
            query_shaping_alpha=rl_config.query_shaping_alpha,
        )

        torch = _torch()
        torch.save(q_network.state_dict(), model_path)
        pd.DataFrame(training_history).to_csv(training_path, index=False)

        results = evaluate_dqn_agent(
            q_network=q_network,
            env=env,
            episodes=eval_episodes,
            encoder=encoder,
            top_k=config.evaluation.top_k,
            max_steps_per_episode=rl_config.max_steps_per_episode,
            min_queries_before_select=rl_config.min_queries_before_select,
            context_encoder=context_encoder,
        )
        results.to_csv(output_path, index=False)
        collect_dqn_trajectory_metrics(
            q_network=q_network,
            env=env,
            episodes=eval_episodes,
            encoder=encoder,
            max_steps_per_episode=rl_config.max_steps_per_episode,
            min_queries_before_select=rl_config.min_queries_before_select,
            context_encoder=context_encoder,
        ).to_csv(trajectory_path, index=False)
        _log_dqn_final_to_wandb(wandb_run, results, training_history, model_path)
    return results, output_path


def load_rl_training_config(config_path: str | Path) -> RLTrainingConfig:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    training = raw.get("rl_training", {})
    return RLTrainingConfig(
        train_episodes=int(training.get("train_episodes", 10_000)),
        eval_episodes=int(training.get("eval_episodes", 500)),
        validation_episodes=int(training.get("validation_episodes", 100)),
        hidden_dim=int(training.get("hidden_dim", 128)),
        learning_rate=float(training.get("learning_rate", 0.0001)),
        gamma=float(training.get("gamma", 0.95)),
        batch_size=int(training.get("batch_size", 64)),
        replay_capacity=int(training.get("replay_capacity", 20_000)),
        learning_starts=int(training.get("learning_starts", 500)),
        train_frequency=int(training.get("train_frequency", 4)),
        target_update_steps=int(training.get("target_update_steps", 500)),
        max_grad_norm=float(training.get("max_grad_norm", 10.0)),
        epsilon_start=float(training.get("epsilon_start", 1.0)),
        epsilon_end=float(training.get("epsilon_end", 0.05)),
        epsilon_decay_steps=int(training.get("epsilon_decay_steps", 20_000)),
        max_steps_per_episode=int(training.get("max_steps_per_episode", 32)),
        select_exploration_probability=float(
            training.get("select_exploration_probability", 0.25)
        ),
        validation_interval=int(training.get("validation_interval", 100)),
        expert_seed_episodes=int(training.get("expert_seed_episodes", 1_000)),
        expert_seed_modality=str(training.get("expert_seed_modality", "expression")),
        expert_seed_strategy=str(training.get("expert_seed_strategy", "single_modality")),
        expert_seed_refinement_modality=str(
            training.get("expert_seed_refinement_modality", "cna")
        ),
        expert_seed_refinement_top_k=int(training.get("expert_seed_refinement_top_k", 4)),
        min_queries_before_select=int(training.get("min_queries_before_select", 0)),
        n_step_returns=int(training.get("n_step_returns", 1)),
        q_network_type=str(training.get("q_network_type", "mlp")),
        cancer_context_column=str(training.get("cancer_context_column", "OncotreeLineage")),
        cancer_context_dim=int(training.get("cancer_context_dim", 16)),
        init_structured_checkpoint=training.get("init_structured_checkpoint"),
        freeze_shared_heads=bool(training.get("freeze_shared_heads", False)),
        use_lineage_modality_scores=bool(training.get("use_lineage_modality_scores", False)),
        lineage_min_samples=int(training.get("lineage_min_samples", 6)),
        query_shaping_alpha=float(training.get("query_shaping_alpha", 0.0)),
        fusion_query_boost=float(training.get("fusion_query_boost", 2.0)),
        fusion_select_weight=float(training.get("fusion_select_weight", 1.0)),
        select_residual_weight=float(training.get("select_residual_weight", 0.25)),
        wandb_log_interval=int(training.get("wandb_log_interval", 25)),
        split_cell_lines=bool(training.get("split_cell_lines", False)),
        validation_cell_line_fraction=float(training.get("validation_cell_line_fraction", 0.1)),
        eval_cell_line_fraction=float(training.get("eval_cell_line_fraction", 0.1)),
    )


def train_dqn_agent(
    env: EvidenceAcquisitionEnv,
    episodes: list[CandidateEpisode],
    encoder: StateEncoder,
    hyperparameters: DQNHyperparameters,
    seed: int,
    validation_episodes: list[CandidateEpisode] | None = None,
    top_k: int = 3,
    context_encoder: LineageContextEncoder | None = None,
    training_log_callback: Callable[[dict[str, float | int]], None] | None = None,
    query_shaping_alpha: float = 0.0,
) -> tuple[Any, list[dict[str, float | int]]]:
    if not episodes:
        raise ValueError("Cannot train DQN on zero episodes.")

    torch = _torch()
    torch.manual_seed(seed)
    q_network = _build_q_network(encoder, hyperparameters)
    _maybe_init_context_from_structured(q_network, hyperparameters)
    target_network = _build_q_network(encoder, hyperparameters)
    _maybe_init_context_from_structured(target_network, hyperparameters)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = torch.optim.Adam(q_network.parameters(), lr=hyperparameters.learning_rate)
    replay = ReplayBuffer(hyperparameters.replay_capacity, seed=seed)
    rng = Random(seed)
    history: list[dict[str, float | int]] = []
    expert_seeded = seed_replay_with_expert(
        replay=replay,
        env=env,
        episodes=episodes[: hyperparameters.expert_seed_episodes],
        encoder=encoder,
        strategy=hyperparameters.expert_seed_strategy,
        modality_name=hyperparameters.expert_seed_modality,
        refinement_modality_name=hyperparameters.expert_seed_refinement_modality,
        refinement_top_k=hyperparameters.expert_seed_refinement_top_k,
    )
    global_step = expert_seeded  # expert counts in epsilon
    best_validation_reward = float("-inf")
    best_state_dict: dict[str, Any] | None = None

    for episode_number, episode in enumerate(episodes):
        state = env.reset(episode)
        context_index = _episode_context_index(context_encoder, episode)
        total_reward = 0.0
        query_count = 0
        losses: list[float] = []
        episode_transitions: list[Transition] = []

        for step_in_episode in range(hyperparameters.max_steps_per_episode):
            state_vector = encoder.encode(state)
            valid_actions = encoder.valid_action_mask(state)
            valid_actions = _apply_min_query_constraint(
                valid_actions,
                state,
                encoder,
                hyperparameters.min_queries_before_select,
            )
            if step_in_episode == hyperparameters.max_steps_per_episode - 1:
                valid_actions = _select_only_mask(encoder)  # force terminal SELECT
            epsilon = epsilon_by_step(global_step, hyperparameters)
            action_index = select_epsilon_greedy_action(
                q_network,
                state_vector,
                valid_actions,
                epsilon,
                rng,
                select_action_indices=encoder.action_space.select_indices(),
                select_exploration_probability=hyperparameters.select_exploration_probability,
                context_index=context_index,
            )
            action = encoder.action_space.from_index(action_index)
            result = env.step(action)
            step_reward = float(result.reward)
            if (
                action.action_type == ActionType.QUERY
                and query_shaping_alpha > 0.0
            ):
                true_rank = _true_dependency_rank(episode.dependency_scores, action.gene_index)
                n_genes = len(episode.candidate_genes)
                rank_fraction = (true_rank - 1) / max(n_genes - 1, 1)
                step_reward += query_shaping_alpha * (1.0 - rank_fraction)  # rank query bonus
            next_state_vector = encoder.encode(result.state)
            next_valid_actions = encoder.valid_action_mask(result.state)
            next_valid_actions = _apply_min_query_constraint(
                next_valid_actions,
                result.state,
                encoder,
                hyperparameters.min_queries_before_select,
            )
            transition = Transition(
                state=state_vector,
                action=action_index,
                reward=step_reward,
                next_state=next_state_vector,
                next_valid_actions=next_valid_actions,
                done=result.done,
                context_index=context_index,
            )
            _append_n_step_transition(
                replay,
                episode_transitions,
                transition,
                hyperparameters.gamma,
                hyperparameters.n_step_returns,
            )

            total_reward += step_reward
            query_count += int(action.action_type == ActionType.QUERY)
            state = result.state
            global_step += 1

            if (
                len(replay) >= max(hyperparameters.batch_size, hyperparameters.learning_starts)
                and global_step % max(hyperparameters.train_frequency, 1) == 0
            ):
                loss = optimize_dqn_batch(
                    q_network,
                    target_network,
                    optimizer,
                    replay.sample(hyperparameters.batch_size),
                    hyperparameters.gamma,
                    max_grad_norm=hyperparameters.max_grad_norm,
                )
                losses.append(loss)

            if global_step % hyperparameters.target_update_steps == 0:
                target_network.load_state_dict(q_network.state_dict())
            if result.done:
                break
        _flush_n_step_transitions(
            replay,
            episode_transitions,
            hyperparameters.gamma,
        )

        row: dict[str, float | int] = {
            "episode": episode_number,
            "total_reward": total_reward,
            "n_queries": query_count,
            "epsilon": epsilon_by_step(global_step, hyperparameters),
            "loss": mean(losses) if losses else 0.0,
        }
        if _should_validate(episode_number, len(episodes), hyperparameters, validation_episodes):
            validation = evaluate_dqn_agent(
                q_network=q_network,
                env=env,
                episodes=validation_episodes or [],
                encoder=encoder,
                top_k=top_k,
                max_steps_per_episode=hyperparameters.max_steps_per_episode,
                min_queries_before_select=hyperparameters.min_queries_before_select,
                context_encoder=context_encoder,
            ).iloc[0]
            row.update(
                {
                    "validation_total_reward": float(validation["total_reward"]),
                    "validation_n_queries": float(validation["n_queries"]),
                    "validation_selected_dependency": float(validation["selected_dependency"]),
                    "validation_hit_at_k": float(validation["hit_at_k"]),
                }
            )
            if row["validation_total_reward"] > best_validation_reward:
                best_validation_reward = float(row["validation_total_reward"])
                best_state_dict = _clone_state_dict(q_network)
        history.append(row)
        if training_log_callback is not None:
            training_log_callback(row)

    if best_state_dict is not None:
        q_network.load_state_dict(best_state_dict)  # restore best val
    return q_network, history


def evaluate_dqn_agent(
    q_network: Any,
    env: EvidenceAcquisitionEnv,
    episodes: list[CandidateEpisode],
    encoder: StateEncoder,
    top_k: int,
    max_steps_per_episode: int,
    min_queries_before_select: int = 0,
    context_encoder: LineageContextEncoder | None = None,
) -> pd.DataFrame:
    if not episodes:
        raise ValueError("Cannot evaluate DQN on zero episodes.")

    rollouts = [
        _run_greedy_dqn_episode(
            q_network,
            env,
            episode,
            encoder,
            max_steps_per_episode,
            min_queries_before_select=min_queries_before_select,
            context_encoder=context_encoder,
        )
        for episode in episodes
    ]
    row: dict[str, float | str] = {
        "policy": "rl_env_dqn",
        "selected_dependency": mean(row.selected_dependency for row in rollouts),
        "hit_at_k": mean(
            hit_at_k(list(episode.dependency_scores), list(row.ranked_indices), top_k)
            for episode, row in zip(episodes, rollouts, strict=True)
        ),
        "ndcg_at_k": mean(
            ndcg_at_k(list(episode.dependency_scores), list(row.ranked_indices), top_k)
            for episode, row in zip(episodes, rollouts, strict=True)
        ),
        "mrr_at_k": mean(
            reciprocal_rank_at_k(list(episode.dependency_scores), list(row.ranked_indices), top_k)
            for episode, row in zip(episodes, rollouts, strict=True)
        ),
        "query_cost": mean(row.query_cost for row in rollouts),
        "n_queries": mean(row.n_queries for row in rollouts),
        "total_reward": mean(row.total_reward for row in rollouts),
    }
    for modality_name in env.modality_names:
        row[f"n_query_{modality_name}"] = mean(
            float(rollout.modality_query_counts.get(modality_name, 0.0)) for rollout in rollouts
        )
    return pd.DataFrame([row])


def build_dqn_eval_env(
    config: BaselineConfig,
    raw_data_dir: str | Path | None,
    rl_config: RLTrainingConfig,
) -> tuple[EvidenceAcquisitionEnv, StateEncoder, list[CandidateEpisode], LineageContextEncoder | None]:
    _, eval_episodes, _, env, encoder, context_encoder = _prepare_dqn_setup(
        config,
        raw_data_dir,
        rl_config,
    )
    return env, encoder, eval_episodes, context_encoder


def collect_dqn_behavior_log(
    q_network: Any,
    env: EvidenceAcquisitionEnv,
    episodes: list[CandidateEpisode],
    encoder: StateEncoder,
    top_k: int,
    max_steps_per_episode: int,
    min_queries_before_select: int = 0,
    context_encoder: LineageContextEncoder | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_rows: list[dict[str, float | int | str | bool]] = []
    step_rows: list[dict[str, float | int | str | None]] = []
    for episode in episodes:
        rollout, steps = _run_greedy_dqn_episode(
            q_network,
            env,
            episode,
            encoder,
            max_steps_per_episode,
            record_steps=True,
            min_queries_before_select=min_queries_before_select,
            context_encoder=context_encoder,
        )
        episode_rows.append(_greedy_episode_row(episode, rollout, env.modality_names, top_k))
        step_rows.extend(steps)
    return pd.DataFrame(episode_rows), pd.DataFrame(step_rows)


def collect_dqn_trajectory_metrics(
    q_network: Any,
    env: EvidenceAcquisitionEnv,
    episodes: list[CandidateEpisode],
    encoder: StateEncoder,
    max_steps_per_episode: int,
    min_queries_before_select: int = 0,
    context_encoder: LineageContextEncoder | None = None,
) -> pd.DataFrame:
    rows = [
        _greedy_episode_row(
            episode,
            _run_greedy_dqn_episode(
                q_network,
                env,
                episode,
                encoder,
                max_steps_per_episode,
                min_queries_before_select=min_queries_before_select,
                context_encoder=context_encoder,
            ),
            env.modality_names,
        )
        for episode in episodes
    ]
    return pd.DataFrame(rows)


def _prepare_dqn_setup(
    config: BaselineConfig,
    raw_data_dir: str | Path | None,
    rl_config: RLTrainingConfig,
) -> tuple[
    list[CandidateEpisode],
    list[CandidateEpisode],
    list[CandidateEpisode],
    EvidenceAcquisitionEnv,
    StateEncoder,
    LineageContextEncoder | None,
]:
    data = _load_training_data(config, raw_data_dir)
    train_dependency = data.dependency
    eval_dependency = data.dependency
    validation_dependency = data.dependency
    if rl_config.split_cell_lines:
        train_dependency, validation_dependency, eval_dependency = maybe_split_dependency_by_cell_line(
            config,
            data.dependency,
            CellLineSplitConfig(
                enabled=True,
                validation_fraction=rl_config.validation_cell_line_fraction,
                eval_fraction=rl_config.eval_cell_line_fraction,
            ),
        )

    train_episodes = _build_episode_set(config, train_dependency, rl_config.train_episodes, config.seed)
    eval_episodes = _build_episode_set(
        config,
        eval_dependency,
        rl_config.eval_episodes,
        config.seed + 1,
    )
    validation_episodes = _build_episode_set(
        config,
        validation_dependency,
        rl_config.validation_episodes,
        config.seed + 2,
    )
    modalities = _build_training_modalities(
        config,
        data,
        rl_config,
        raw_data_dir,
        train_episodes,
    )
    env = EvidenceAcquisitionEnv(
        modalities,
        query_costs=config.environment.query_costs,
        repeated_query_penalty=config.environment.repeated_query_penalty,
        selection_reward_scale=config.environment.selection_reward_scale,
    )
    encoder = StateEncoder(
        n_genes=config.episodes.candidates_per_episode,
        n_modalities=len(env.modality_names),
    )
    context_encoder = _load_context_encoder(config, raw_data_dir, rl_config)
    return train_episodes, eval_episodes, validation_episodes, env, encoder, context_encoder


def _greedy_episode_row(
    episode: CandidateEpisode,
    rollout: DQNRollout,
    modality_names: tuple[str, ...],
    top_k: int | None = None,
) -> dict[str, float | int | str | bool]:
    row: dict[str, float | int | str | bool] = {
        "episode_id": rollout.episode_id,
        "cell_line_id": rollout.cell_line_id,
        "selected_gene": rollout.selected_gene,
        "selected_index": rollout.selected_index,
        "selected_dependency": rollout.selected_dependency,
        "query_cost": rollout.query_cost,
        "n_queries": rollout.n_queries,
        "total_reward": rollout.total_reward,
        **{
            f"n_query_{name}": rollout.modality_query_counts.get(name, 0)
            for name in modality_names
        },
    }
    if top_k is not None:
        row["hit_at_k"] = hit_at_k(
            list(episode.dependency_scores),
            list(rollout.ranked_indices),
            top_k,
        )
        row["dependency_regret"] = rollout.selected_dependency - min(episode.dependency_scores)
    return row


def _run_greedy_dqn_episode(
    q_network: Any,
    env: EvidenceAcquisitionEnv,
    episode: CandidateEpisode,
    encoder: StateEncoder,
    max_steps_per_episode: int,
    record_steps: bool = False,
    min_queries_before_select: int = 0,
    context_encoder: LineageContextEncoder | None = None,
) -> DQNRollout | tuple[DQNRollout, list[dict[str, float | int | str | None]]]:
    state = env.reset(episode)
    context_index = _episode_context_index(context_encoder, episode)
    total_reward = 0.0
    query_cost = 0.0
    n_queries = 0
    modality_query_counts = {name: 0 for name in env.modality_names}
    selection_state = state
    step_rows: list[dict[str, float | int | str | None]] = []
    step_index = 0
    selected_index = 0

    for _ in range(max_steps_per_episode):
        state_vector = encoder.encode(state)
        valid_actions = _apply_min_query_constraint(
            encoder.valid_action_mask(state),
            state,
            encoder,
            min_queries_before_select,
        )
        action_index = select_greedy_action(
            q_network,
            state_vector,
            valid_actions,
            context_index=context_index,
        )
        action = encoder.action_space.from_index(action_index)
        if action.action_type == ActionType.SELECT:
            selection_state = state  # rank at decision time
        result = env.step(action)
        total_reward += result.reward
        modality_name: str | None = None
        observed_value: float | None = None
        if action.action_type == ActionType.QUERY:
            if action.modality_index is None:
                raise ValueError("Query actions require a modality index.")
            modality_name = env.modality_names[action.modality_index]
            query_cost -= result.reward  # reward is negative cost
            n_queries += 1
            modality_query_counts[modality_name] += 1
            observed_value = result.state.observed_values[action.gene_index][action.modality_index]
        if record_steps:
            step_rows.append(
                _step_log_row(
                    episode,
                    step_index,
                    action,
                    modality_name,
                    observed_value,
                    query_cost,
                    total_reward,
                    n_queries,
                )
            )
        state = result.state
        step_index += 1
        if result.done:
            selected_index = action.gene_index
            break
    else:
        selection_state = env.state
        selected_index = _force_select(q_network, env, encoder, context_index=context_index)  # horizon timeout
        result = env.step(env.select_action(selected_index))
        total_reward += result.reward
        if record_steps:
            step_rows.append(
                _step_log_row(
                    episode,
                    step_index,
                    env.select_action(selected_index),
                    None,
                    None,
                    query_cost,
                    total_reward,
                    n_queries,
                )
            )

    ranked = _rank_select_actions(q_network, selection_state, encoder, context_index=context_index)
    selected_dependency = float(episode.dependency_scores[selected_index])
    rollout = DQNRollout(
        ranked_indices=tuple(ranked),
        selected_index=selected_index,
        selected_dependency=selected_dependency,
        query_cost=query_cost,
        n_queries=n_queries,
        total_reward=total_reward,
        modality_query_counts=modality_query_counts,
        selected_gene=episode.candidate_genes[selected_index],
        cell_line_id=episode.cell_line_id,
        episode_id=episode.episode_id,
    )
    if record_steps:
        return rollout, step_rows
    return rollout


def _step_log_row(
    episode: CandidateEpisode,
    step: int,
    action: Any,
    modality_name: str | None,
    observed_value: float | None,
    cumulative_query_cost: float,
    cumulative_reward: float,
    n_queries_so_far: int,
) -> dict[str, float | int | str | None]:
    gene_index = action.gene_index
    return {
        "episode_id": episode.episode_id,
        "cell_line_id": episode.cell_line_id,
        "step": step,
        "action_type": action.action_type.value,
        "gene": episode.candidate_genes[gene_index],
        "modality": modality_name,
        "observed_value": observed_value,
        "gene_true_rank": _true_dependency_rank(episode.dependency_scores, gene_index),
        "n_queries_so_far": n_queries_so_far,
        "cumulative_query_cost": cumulative_query_cost,
        "cumulative_reward": cumulative_reward,
    }


def _true_dependency_rank(dependency_scores: tuple[float, ...], gene_index: int) -> int:
    order = sorted(range(len(dependency_scores)), key=lambda index: dependency_scores[index])
    return order.index(gene_index) + 1


def _rank_select_actions(
    q_network: Any,
    state: Any,
    encoder: StateEncoder,
    context_index: int = 0,
) -> list[int]:
    torch = _torch()
    with torch.no_grad():
        state_tensor = torch.as_tensor(encoder.encode(state), dtype=torch.float32).unsqueeze(0)
        context_tensor = _context_tensor(context_index, q_network)
        q_values = forward_q_network(q_network, state_tensor, context_tensor).squeeze(0).detach().cpu().numpy()
    select_indices = encoder.action_space.select_indices()
    return [
        int(index - encoder.n_genes * encoder.n_modalities)  # past query actions
        for index in select_indices[np.argsort(q_values[select_indices])[::-1]]
    ]


def _force_select(
    q_network: Any,
    env: EvidenceAcquisitionEnv,
    encoder: StateEncoder,
    context_index: int = 0,
) -> int:
    mask = _select_only_mask(encoder)
    action_index = select_greedy_action(
        q_network,
        encoder.encode(env.state),
        mask,
        context_index=context_index,
    )
    return encoder.action_space.from_index(action_index).gene_index


def _select_only_mask(encoder: StateEncoder) -> np.ndarray:
    mask = np.zeros(encoder.action_space.size, dtype=bool)
    mask[encoder.action_space.select_indices()] = True
    return mask


def _append_n_step_transition(
    replay: ReplayBuffer,
    pending: list[Transition],
    transition: Transition,
    gamma: float,
    n_steps: int,
) -> None:
    pending.append(transition)
    if len(pending) >= max(n_steps, 1):
        replay.append(_collapse_n_step(pending[: max(n_steps, 1)], gamma))
        pending.pop(0)


def _flush_n_step_transitions(
    replay: ReplayBuffer,
    pending: list[Transition],
    gamma: float,
) -> None:
    while pending:
        replay.append(_collapse_n_step(pending, gamma))
        pending.pop(0)


def _collapse_n_step(transitions: list[Transition], gamma: float) -> Transition:
    reward = 0.0
    for offset, transition in enumerate(transitions):
        reward += (gamma**offset) * transition.reward  # n-step return
        if transition.done:
            transitions = transitions[: offset + 1]
            break
    last = transitions[-1]
    first = transitions[0]
    return Transition(
        state=first.state,
        action=first.action,
        reward=reward,
        next_state=last.next_state,
        next_valid_actions=last.next_valid_actions,
        done=last.done,
        n_steps=len(transitions),
        context_index=first.context_index,
    )


def _apply_min_query_constraint(
    valid_actions: np.ndarray,
    state: Any,
    encoder: StateEncoder,
    min_queries_before_select: int,
) -> np.ndarray:
    if min_queries_before_select <= 0 or state.done:
        return valid_actions
    n_queries = sum(int(observed) for row in state.query_mask for observed in row)
    if n_queries >= min_queries_before_select:
        return valid_actions

    constrained = valid_actions.copy()
    constrained[encoder.action_space.select_indices()] = False  # block early SELECT
    if not constrained.any():
        return valid_actions
    return constrained



def seed_replay_with_modality_expert(
    replay: ReplayBuffer,
    env: EvidenceAcquisitionEnv,
    episodes: list[CandidateEpisode],
    encoder: StateEncoder,
    modality_name: str,
) -> int:
    return seed_replay_with_expert(
        replay=replay,
        env=env,
        episodes=episodes,
        encoder=encoder,
        strategy="single_modality",
        modality_name=modality_name,
    )


def seed_replay_with_expert(
    replay: ReplayBuffer,
    env: EvidenceAcquisitionEnv,
    episodes: list[CandidateEpisode],
    encoder: StateEncoder,
    strategy: str,
    modality_name: str,
    refinement_modality_name: str = "cna",
    refinement_top_k: int = 4,
) -> int:
    if not episodes:
        return 0
    if strategy == "single_modality":
        return _seed_replay_with_single_modality_expert(
            replay,
            env,
            episodes,
            encoder,
            modality_name,
        )
    if strategy == "expression_full_then_refinement_top_k":
        return _seed_replay_with_refinement_expert(
            replay=replay,
            env=env,
            episodes=episodes,
            encoder=encoder,
            screening_modality_name=modality_name,
            refinement_modality_name=refinement_modality_name,
            refinement_top_k=refinement_top_k,
        )
    raise ValueError(f"Unknown expert seed strategy: {strategy}")


def _seed_replay_with_single_modality_expert(
    replay: ReplayBuffer,
    env: EvidenceAcquisitionEnv,
    episodes: list[CandidateEpisode],
    encoder: StateEncoder,
    modality_name: str,
) -> int:
    try:
        modality_index = env.modality_names.index(modality_name)
    except ValueError:
        return 0

    n_transitions = 0
    for episode in episodes:
        state = env.reset(episode)
        observed_scores: list[float] = []
        for gene_index in range(len(episode.candidate_genes)):
            state_vector = encoder.encode(state)
            action = env.query_action(gene_index, modality_index)
            result = env.step(action)
            replay.append(
                Transition(
                    state=state_vector,
                    action=encoder.action_space.to_index(action),
                    reward=result.reward,
                    next_state=encoder.encode(result.state),
                    next_valid_actions=encoder.valid_action_mask(result.state),
                    done=result.done,
                )
            )
            observed_scores.append(_rank_value(result.state.observed_values[gene_index][modality_index]))
            state = result.state
            n_transitions += 1

        selected_index = int(np.argmax(np.asarray(observed_scores, dtype=np.float32)))
        state_vector = encoder.encode(state)
        action = env.select_action(selected_index)
        result = env.step(action)
        replay.append(
            Transition(
                state=state_vector,
                action=encoder.action_space.to_index(action),
                reward=result.reward,
                next_state=encoder.encode(result.state),
                next_valid_actions=encoder.valid_action_mask(result.state),
                done=result.done,
            )
        )
        n_transitions += 1
    return n_transitions


def _seed_replay_with_refinement_expert(
    replay: ReplayBuffer,
    env: EvidenceAcquisitionEnv,
    episodes: list[CandidateEpisode],
    encoder: StateEncoder,
    screening_modality_name: str,
    refinement_modality_name: str,
    refinement_top_k: int,
) -> int:
    try:
        screening_index = env.modality_names.index(screening_modality_name)
        refinement_index = env.modality_names.index(refinement_modality_name)
    except ValueError:
        return 0

    n_transitions = 0
    for episode in episodes:
        state = env.reset(episode)
        screening_scores: list[float] = []
        for gene_index in range(len(episode.candidate_genes)):
            result = _append_expert_transition(
                replay,
                env,
                encoder,
                state,
                env.query_action(gene_index, screening_index),
            )
            screening_scores.append(
                _rank_value(result.state.observed_values[gene_index][screening_index])
            )
            state = result.state
            n_transitions += 1

        top_indices = np.argsort(np.asarray(screening_scores, dtype=np.float32))[::-1][
            : max(refinement_top_k, 0)  # refine top-k only
        ]
        for gene_index in top_indices:
            result = _append_expert_transition(
                replay,
                env,
                encoder,
                state,
                env.query_action(int(gene_index), refinement_index),
            )
            state = result.state
            n_transitions += 1

        selected_index = _select_by_standardized_observed_scores(
            state.observed_values,
            queried_modalities=(screening_index, refinement_index),
        )
        _append_expert_transition(
            replay,
            env,
            encoder,
            state,
            env.select_action(selected_index),
        )
        n_transitions += 1
    return n_transitions


def _append_expert_transition(
    replay: ReplayBuffer,
    env: EvidenceAcquisitionEnv,
    encoder: StateEncoder,
    state: Any,
    action: Any,
) -> Any:
    state_vector = encoder.encode(state)
    result = env.step(action)
    replay.append(
        Transition(
            state=state_vector,
            action=encoder.action_space.to_index(action),
            reward=result.reward,
            next_state=encoder.encode(result.state),
            next_valid_actions=encoder.valid_action_mask(result.state),
            done=result.done,
        )
    )
    return result


def _select_by_standardized_observed_scores(
    observed_values: tuple[tuple[float | None, ...], ...],
    queried_modalities: tuple[int, ...],
) -> int:
    values_by_modality = {
        modality_index: [
            _rank_value(row[modality_index])
            if row[modality_index] is not None and not pd.isna(row[modality_index])
            else None
            for row in observed_values
        ]
        for modality_index in queried_modalities
    }
    standardized = {
        modality_index: _standardize_rank_values(values)
        for modality_index, values in values_by_modality.items()
    }
    scores = []
    for gene_index in range(len(observed_values)):
        observed = [
            values[gene_index]
            for values in standardized.values()
            if values[gene_index] is not None
        ]
        scores.append(mean(observed) if observed else float("-inf"))
    return int(np.argmax(np.asarray(scores, dtype=np.float32)))


def _standardize_rank_values(values: list[float | None]) -> list[float | None]:
    observed = [float(value) for value in values if value is not None]
    if not observed:
        return values
    center = mean(observed)
    scale = float(np.std(np.asarray(observed, dtype=np.float32)))
    if scale == 0.0:
        return [0.0 if value is not None else None for value in values]
    return [(float(value) - center) / scale if value is not None else None for value in values]


def _rank_value(value: float | None) -> float:
    if value is None or pd.isna(value):
        return float("-inf")
    return float(value)


def _build_training_modalities(
    config: BaselineConfig,
    data: Any,
    rl_config: RLTrainingConfig,
    raw_data_dir: str | Path | None,
    train_episodes: list[CandidateEpisode],
) -> dict[str, pd.DataFrame]:
    if not config.environment.use_supervised_modality_scores:
        return data.modalities  # raw DepMap values
    train_cell_lines = {episode.cell_line_id for episode in train_episodes}
    if rl_config.use_lineage_modality_scores:
        metadata_path = _resolve_optional_data_path(config.data.metadata_path, raw_data_dir)
        if metadata_path is None:
            raise ValueError("use_lineage_modality_scores requires data.metadata_path in the config.")
        return build_lineage_supervised_modality_scores(  # ridge query returns
            data.dependency,
            data.modalities,
            metadata_path,
            context_column=rl_config.cancer_context_column,
            train_cell_lines=train_cell_lines,
            lineage_min_samples=rl_config.lineage_min_samples,
        )
    return build_supervised_modality_scores(
        data.dependency,
        data.modalities,
        train_cell_lines=train_cell_lines,
    )


def _load_context_encoder(
    config: BaselineConfig,
    raw_data_dir: str | Path | None,
    rl_config: RLTrainingConfig,
) -> LineageContextEncoder | None:
    if not uses_cancer_context(rl_config.q_network_type):
        return None
    metadata_path = _resolve_optional_data_path(config.data.metadata_path, raw_data_dir)
    if metadata_path is None:
        raise ValueError("context_structured requires data.metadata_path in the config.")
    return LineageContextEncoder(
        metadata_path,
        context_column=rl_config.cancer_context_column,
    )


def _hyperparameters_with_context(
    hyperparameters: DQNHyperparameters,
    encoder: StateEncoder,
    context_encoder: LineageContextEncoder | None,
) -> DQNHyperparameters:
    return replace(
        hyperparameters,
        n_genes=encoder.n_genes,
        n_modalities=encoder.n_modalities,
        n_lineages=context_encoder.n_lineages if context_encoder is not None else 0,
    )


def _maybe_init_context_from_structured(q_network: Any, hyperparameters: DQNHyperparameters) -> None:
    checkpoint = hyperparameters.init_structured_checkpoint
    if not checkpoint or not uses_cancer_context(hyperparameters.q_network_type):
        return
    from src.dqn import requires_context_indices

    if not requires_context_indices(q_network):
        return
    load_structured_checkpoint_into_context(
        q_network,
        checkpoint,
        freeze_shared_heads=hyperparameters.freeze_shared_heads,
    )


def _build_q_network(encoder: StateEncoder, hyperparameters: DQNHyperparameters) -> Any:
    return build_q_network(
        encoder.state_size,
        encoder.action_space.size,
        hyperparameters.hidden_dim,
        network_type=hyperparameters.q_network_type,
        n_genes=encoder.n_genes,
        n_modalities=encoder.n_modalities,
        n_lineages=hyperparameters.n_lineages,
        cancer_context_dim=hyperparameters.cancer_context_dim,
        fusion_query_boost=hyperparameters.fusion_query_boost,
        fusion_select_weight=hyperparameters.fusion_select_weight,
        select_residual_weight=hyperparameters.select_residual_weight,
    )


def _episode_context_index(
    context_encoder: LineageContextEncoder | None,
    episode: CandidateEpisode,
) -> int:
    if context_encoder is None:
        return 0
    return context_encoder.encode(episode.cell_line_id)


def _should_validate(
    episode_number: int,
    n_train_episodes: int,
    hyperparameters: DQNHyperparameters,
    validation_episodes: list[CandidateEpisode] | None,
) -> bool:
    if not validation_episodes:
        return False
    interval = max(hyperparameters.validation_interval, 1)
    return (episode_number + 1) % interval == 0 or episode_number == n_train_episodes - 1


def _clone_state_dict(q_network: Any) -> dict[str, Any]:
    return {key: value.detach().clone() for key, value in q_network.state_dict().items()}


def _load_training_data(config: BaselineConfig, raw_data_dir: str | Path | None):
    dependency_path = _resolve_data_path(config.data.dependency_path, raw_data_dir)
    metadata_path = _resolve_optional_data_path(config.data.metadata_path, raw_data_dir)
    modality_paths = {
        name: _resolve_data_path(path, raw_data_dir)
        for name, path in config.data.modalities.items()
    }
    return load_project_data(
        dependency_path=dependency_path,
        modality_paths=modality_paths,
        metadata_path=metadata_path,
    )


def _build_episode_set(
    config: BaselineConfig,
    dependency: pd.DataFrame,
    n_episodes: int,
    seed: int,
) -> list[CandidateEpisode]:
    builder = EpisodeBuilder(
        dependency=dependency,
        dependency_threshold=config.episodes.dependency_threshold,
        candidates_per_episode=config.episodes.candidates_per_episode,
        positives_per_episode=config.episodes.positives_per_episode,
        min_candidates_per_cell_line=config.episodes.min_candidates_per_cell_line,
        seed=seed,
    )
    return builder.build(n_episodes)


def _log_dqn_to_wandb(
    config: BaselineConfig,
    config_path: str | Path,
    rl_config: RLTrainingConfig,
    results: pd.DataFrame,
    training_history: list[dict[str, float | int]],
    output_path: Path,
    model_path: Path,
) -> None:
    if not config.tracking.wandb.enabled:
        return
    with _wandb_dqn_run(config, config_path, rl_config, output_path) as run:
        if training_history:
            logger = _wandb_training_logger(run, rl_config)
            for row in training_history:
                logger(row)
        _log_dqn_final_to_wandb(run, results, training_history, model_path)


@contextmanager
def _wandb_dqn_run(
    config: BaselineConfig,
    config_path: str | Path,
    rl_config: RLTrainingConfig,
    output_path: Path,
    run_name: str = "dqn-training",
) -> Iterator[Any | None]:
    if not config.tracking.wandb.enabled:
        yield None
        return

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B tracking is enabled, but wandb is not installed. "
            "Install project dependencies or disable tracking.wandb.enabled."
        ) from error

    with wandb.init(
        entity=config.tracking.wandb.entity,
        project=config.tracking.wandb.project,
        name=run_name,
        job_type="rl-training",
        config={
            "config_path": str(config_path),
            "rl_training": rl_config.__dict__,
            "output_path": str(output_path),
        },
    ) as run:
        yield run


def _wandb_training_logger(
    run: Any | None,
    rl_config: RLTrainingConfig,
) -> Callable[[dict[str, float | int]], None] | None:
    if run is None:
        return None

    log_interval = max(rl_config.wandb_log_interval, 1)
    final_episode = rl_config.train_episodes - 1

    def log_row(row: dict[str, float | int]) -> None:
        episode = int(row["episode"])
        should_log_train = episode % log_interval == 0 or episode == final_episode
        if should_log_train:
            run.log(
                {
                    "train/total_reward": row["total_reward"],
                    "train/n_queries": row["n_queries"],
                    "train/epsilon": row["epsilon"],
                    "train/loss": row["loss"],
                },
                step=episode,
            )
        if "validation_total_reward" in row:
            run.log(
                {
                    "validation/total_reward": row["validation_total_reward"],
                    "validation/n_queries": row["validation_n_queries"],
                    "validation/selected_dependency": row[
                        "validation_selected_dependency"
                    ],
                    "validation/hit_at_k": row["validation_hit_at_k"],
                },
                step=episode,
            )

    return log_row


def _log_dqn_final_to_wandb(
    run: Any | None,
    results: pd.DataFrame,
    training_history: list[dict[str, float | int]],
    model_path: Path,
) -> None:
    if run is None:
        return

    if training_history:
        run.summary.update(
            {
                "train/final_total_reward": training_history[-1]["total_reward"],
                "train/final_n_queries": training_history[-1]["n_queries"],
                "train/final_epsilon": training_history[-1]["epsilon"],
                "train/final_loss": training_history[-1]["loss"],
            }
        )
    eval_metrics = {}
    for row in results.to_dict(orient="records"):
        for key, value in row.items():
            if key != "policy":
                eval_metrics[f"eval/{key}"] = value
    run.summary.update(eval_metrics)

    import wandb

    run.log({"dqn_eval_metrics": wandb.Table(dataframe=results)})
    run.save(str(model_path))


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "DQN training requires torch. Install project dependencies first."
        ) from error
    return torch
