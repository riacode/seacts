from __future__ import annotations

from dataclasses import dataclass
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
from src.dqn import (
    DQNHyperparameters,
    build_q_network,
    epsilon_by_step,
    optimize_dqn_batch,
    select_epsilon_greedy_action,
    select_greedy_action,
)
from src.environment import ActionType, EvidenceAcquisitionEnv
from src.episodes import CandidateEpisode, EpisodeBuilder
from src.metrics import hit_at_k, ndcg_at_k, reciprocal_rank_at_k
from src.replay_buffer import ReplayBuffer, Transition
from src.state_encoder import StateEncoder


@dataclass(frozen=True)
class RLTrainingConfig:
    train_episodes: int = 1_000
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
    epsilon_decay_steps: int = 2_000
    max_steps_per_episode: int = 32
    select_exploration_probability: float = 0.5
    validation_interval: int = 50
    expert_seed_episodes: int = 200
    expert_seed_modality: str = "expression"
    wandb_log_interval: int = 10

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


def run_dqn_training_pipeline(
    config_path: str | Path,
    raw_data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    config = load_baseline_config(config_path)
    rl_config = load_rl_training_config(config_path)
    data = _load_training_data(config, raw_data_dir)
    train_episodes = _build_episode_set(config, data.dependency, rl_config.train_episodes, config.seed)
    eval_episodes = _build_episode_set(
        config,
        data.dependency,
        rl_config.eval_episodes,
        config.seed + 1,
    )
    validation_episodes = _build_episode_set(
        config,
        data.dependency,
        rl_config.validation_episodes,
        config.seed + 2,
    )

    env = EvidenceAcquisitionEnv(
        data.modalities,
        query_costs=config.environment.query_costs,
        repeated_query_penalty=config.environment.repeated_query_penalty,
    )
    encoder = StateEncoder(
        n_genes=config.episodes.candidates_per_episode,
        n_modalities=len(env.modality_names),
    )
    q_network, training_history = train_dqn_agent(
        env=env,
        episodes=train_episodes,
        encoder=encoder,
        hyperparameters=rl_config.hyperparameters,
        seed=config.seed,
        validation_episodes=validation_episodes,
        top_k=config.evaluation.top_k,
    )

    resolved_output_dir = Path(output_dir) if output_dir is not None else Path(config.output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    model_path = resolved_output_dir / "dqn_policy.pt"
    training_path = resolved_output_dir / "dqn_training_metrics.csv"
    output_path = resolved_output_dir / "dqn_eval_metrics.csv"

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
    )
    results.to_csv(output_path, index=False)
    _log_dqn_to_wandb(config, config_path, rl_config, results, training_history, output_path, model_path)
    return results, output_path


def load_rl_training_config(config_path: str | Path) -> RLTrainingConfig:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    training = raw.get("rl_training", {})
    return RLTrainingConfig(
        train_episodes=int(training.get("train_episodes", 1_000)),
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
        epsilon_decay_steps=int(training.get("epsilon_decay_steps", 2_000)),
        max_steps_per_episode=int(training.get("max_steps_per_episode", 32)),
        select_exploration_probability=float(
            training.get("select_exploration_probability", 0.5)
        ),
        validation_interval=int(training.get("validation_interval", 50)),
        expert_seed_episodes=int(training.get("expert_seed_episodes", 200)),
        expert_seed_modality=str(training.get("expert_seed_modality", "expression")),
        wandb_log_interval=int(training.get("wandb_log_interval", 10)),
    )


def train_dqn_agent(
    env: EvidenceAcquisitionEnv,
    episodes: list[CandidateEpisode],
    encoder: StateEncoder,
    hyperparameters: DQNHyperparameters,
    seed: int,
    validation_episodes: list[CandidateEpisode] | None = None,
    top_k: int = 3,
) -> tuple[Any, list[dict[str, float | int]]]:
    if not episodes:
        raise ValueError("Cannot train DQN on zero episodes.")

    torch = _torch()
    torch.manual_seed(seed)
    q_network = build_q_network(
        encoder.state_size,
        encoder.action_space.size,
        hyperparameters.hidden_dim,
    )
    target_network = build_q_network(
        encoder.state_size,
        encoder.action_space.size,
        hyperparameters.hidden_dim,
    )
    target_network.load_state_dict(q_network.state_dict())
    optimizer = torch.optim.Adam(q_network.parameters(), lr=hyperparameters.learning_rate)
    replay = ReplayBuffer(hyperparameters.replay_capacity, seed=seed)
    rng = Random(seed)
    history: list[dict[str, float | int]] = []
    expert_seeded = seed_replay_with_modality_expert(
        replay=replay,
        env=env,
        episodes=episodes[: hyperparameters.expert_seed_episodes],
        encoder=encoder,
        modality_name=hyperparameters.expert_seed_modality,
    )
    global_step = expert_seeded
    best_validation_reward = float("-inf")
    best_state_dict: dict[str, Any] | None = None

    for episode_number, episode in enumerate(episodes):
        state = env.reset(episode)
        total_reward = 0.0
        query_count = 0
        losses: list[float] = []

        for step_in_episode in range(hyperparameters.max_steps_per_episode):
            state_vector = encoder.encode(state)
            valid_actions = encoder.valid_action_mask(state)
            if step_in_episode == hyperparameters.max_steps_per_episode - 1:
                valid_actions = _select_only_mask(encoder)
            epsilon = epsilon_by_step(global_step, hyperparameters)
            action_index = select_epsilon_greedy_action(
                q_network,
                state_vector,
                valid_actions,
                epsilon,
                rng,
                select_action_indices=encoder.action_space.select_indices(),
                select_exploration_probability=hyperparameters.select_exploration_probability,
            )
            action = encoder.action_space.from_index(action_index)
            result = env.step(action)
            next_state_vector = encoder.encode(result.state)
            next_valid_actions = encoder.valid_action_mask(result.state)
            replay.append(
                Transition(
                    state=state_vector,
                    action=action_index,
                    reward=result.reward,
                    next_state=next_state_vector,
                    next_valid_actions=next_valid_actions,
                    done=result.done,
                )
            )

            total_reward += result.reward
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

    if best_state_dict is not None:
        q_network.load_state_dict(best_state_dict)
    return q_network, history


def evaluate_dqn_agent(
    q_network: Any,
    env: EvidenceAcquisitionEnv,
    episodes: list[CandidateEpisode],
    encoder: StateEncoder,
    top_k: int,
    max_steps_per_episode: int,
) -> pd.DataFrame:
    if not episodes:
        raise ValueError("Cannot evaluate DQN on zero episodes.")

    rollouts = [
        _run_greedy_dqn_episode(q_network, env, episode, encoder, max_steps_per_episode)
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


def _run_greedy_dqn_episode(
    q_network: Any,
    env: EvidenceAcquisitionEnv,
    episode: CandidateEpisode,
    encoder: StateEncoder,
    max_steps_per_episode: int,
) -> DQNRollout:
    state = env.reset(episode)
    total_reward = 0.0
    query_cost = 0.0
    n_queries = 0
    modality_query_counts = {name: 0 for name in env.modality_names}
    selection_state = state

    for _ in range(max_steps_per_episode):
        state_vector = encoder.encode(state)
        action_index = select_greedy_action(q_network, state_vector, encoder.valid_action_mask(state))
        action = encoder.action_space.from_index(action_index)
        if action.action_type == ActionType.SELECT:
            selection_state = state
        result = env.step(action)
        total_reward += result.reward
        if action.action_type == ActionType.QUERY:
            if action.modality_index is None:
                raise ValueError("Query actions require a modality index.")
            modality_name = env.modality_names[action.modality_index]
            query_cost -= result.reward
            n_queries += 1
            modality_query_counts[modality_name] += 1
        state = result.state
        if result.done:
            selected_index = action.gene_index
            break
    else:
        selection_state = env.state
        selected_index = _force_select(q_network, env, encoder)
        result = env.step(env.select_action(selected_index))
        total_reward += result.reward

    ranked = _rank_select_actions(q_network, selection_state, encoder)
    selected_dependency = float(episode.dependency_scores[selected_index])
    return DQNRollout(
        ranked_indices=tuple(ranked),
        selected_index=selected_index,
        selected_dependency=selected_dependency,
        query_cost=query_cost,
        n_queries=n_queries,
        total_reward=total_reward,
        modality_query_counts=modality_query_counts,
    )


def _rank_select_actions(q_network: Any, state: Any, encoder: StateEncoder) -> list[int]:
    torch = _torch()
    with torch.no_grad():
        state_tensor = torch.as_tensor(encoder.encode(state), dtype=torch.float32).unsqueeze(0)
        q_values = q_network(state_tensor).squeeze(0).detach().cpu().numpy()
    select_indices = encoder.action_space.select_indices()
    return [
        int(index - encoder.n_genes * encoder.n_modalities)
        for index in select_indices[np.argsort(q_values[select_indices])[::-1]]
    ]


def _force_select(q_network: Any, env: EvidenceAcquisitionEnv, encoder: StateEncoder) -> int:
    mask = _select_only_mask(encoder)
    action_index = select_greedy_action(q_network, encoder.encode(env.state), mask)
    return encoder.action_space.from_index(action_index).gene_index


def _select_only_mask(encoder: StateEncoder) -> np.ndarray:
    mask = np.zeros(encoder.action_space.size, dtype=bool)
    mask[encoder.action_space.select_indices()] = True
    return mask


def seed_replay_with_modality_expert(
    replay: ReplayBuffer,
    env: EvidenceAcquisitionEnv,
    episodes: list[CandidateEpisode],
    encoder: StateEncoder,
    modality_name: str,
) -> int:
    if not episodes:
        return 0
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


def _rank_value(value: float | None) -> float:
    if value is None or pd.isna(value):
        return float("-inf")
    return float(value)


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
        name="dqn-training",
        job_type="rl-training",
        config={
            "config_path": str(config_path),
            "rl_training": rl_config.__dict__,
            "output_path": str(output_path),
        },
    ) as run:
        if training_history:
            final_episode = int(training_history[-1]["episode"])
            log_interval = max(rl_config.wandb_log_interval, 1)
            for row in training_history:
                episode = int(row["episode"])
                if episode % log_interval != 0 and episode != final_episode:
                    continue
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
