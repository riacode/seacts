from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

import numpy as np

from src.replay_buffer import Transition


@dataclass(frozen=True)
class DQNHyperparameters:
    hidden_dim: int = 128
    learning_rate: float = 0.001
    gamma: float = 0.95
    batch_size: int = 64
    replay_capacity: int = 20_000
    target_update_steps: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 2_000
    max_steps_per_episode: int = 32


def build_q_network(state_size: int, action_size: int, hidden_dim: int) -> Any:
    torch, nn, _ = _torch_modules()
    return nn.Sequential(
        nn.Linear(state_size, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, action_size),
    )


def select_epsilon_greedy_action(
    q_network: Any,
    state: np.ndarray,
    valid_actions: np.ndarray,
    epsilon: float,
    rng: Random,
) -> int:
    valid_indices = np.flatnonzero(valid_actions)
    if len(valid_indices) == 0:
        raise ValueError("No valid actions are available.")
    if rng.random() < epsilon:
        return int(rng.choice(valid_indices.tolist()))
    return select_greedy_action(q_network, state, valid_actions)


def select_greedy_action(q_network: Any, state: np.ndarray, valid_actions: np.ndarray) -> int:
    torch, _, _ = _torch_modules()
    valid_indices = np.flatnonzero(valid_actions)
    if len(valid_indices) == 0:
        raise ValueError("No valid actions are available.")

    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = q_network(state_tensor).squeeze(0).detach().cpu().numpy()
    masked_values = np.full_like(q_values, fill_value=-np.inf, dtype=np.float32)
    masked_values[valid_indices] = q_values[valid_indices]
    return int(np.argmax(masked_values))


def optimize_dqn_batch(
    q_network: Any,
    target_network: Any,
    optimizer: Any,
    transitions: list[Transition],
    gamma: float,
) -> float:
    torch, _, functional = _torch_modules()

    states = torch.as_tensor(np.stack([item.state for item in transitions]), dtype=torch.float32)
    actions = torch.as_tensor([item.action for item in transitions], dtype=torch.int64).unsqueeze(1)
    rewards = torch.as_tensor([item.reward for item in transitions], dtype=torch.float32)
    next_states = torch.as_tensor(
        np.stack([item.next_state for item in transitions]),
        dtype=torch.float32,
    )
    next_valid = torch.as_tensor(
        np.stack([item.next_valid_actions for item in transitions]),
        dtype=torch.bool,
    )
    done = torch.as_tensor([item.done for item in transitions], dtype=torch.bool)

    current_q = q_network(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        # Double DQN: online network selects the next action, target network evaluates it.
        next_online_q = q_network(next_states).masked_fill(~next_valid, -1.0e9)
        next_actions = next_online_q.argmax(dim=1, keepdim=True)
        next_target_q = target_network(next_states).gather(1, next_actions).squeeze(1)
        next_target_q = torch.where(done, torch.zeros_like(next_target_q), next_target_q)
        expected_q = rewards + gamma * next_target_q

    loss = functional.smooth_l1_loss(current_q, expected_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


def epsilon_by_step(step: int, hyperparameters: DQNHyperparameters) -> float:
    if hyperparameters.epsilon_decay_steps <= 0:
        return hyperparameters.epsilon_end
    progress = min(max(step, 0) / hyperparameters.epsilon_decay_steps, 1.0)
    return (
        hyperparameters.epsilon_start
        + progress * (hyperparameters.epsilon_end - hyperparameters.epsilon_start)
    )


def _torch_modules() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as error:
        raise RuntimeError(
            "DQN training requires torch. Install project dependencies first."
        ) from error
    return torch, nn, functional
