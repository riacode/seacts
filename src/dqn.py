from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.replay_buffer import Transition


@dataclass(frozen=True)
class DQNHyperparameters:
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
    n_genes: int = 0
    n_modalities: int = 0
    n_lineages: int = 0
    cancer_context_dim: int = 16
    init_structured_checkpoint: str | None = None
    freeze_shared_heads: bool = False
    fusion_query_boost: float = 2.0
    fusion_select_weight: float = 1.0
    select_residual_weight: float = 0.25


def uses_cancer_context(network_type: str) -> bool:
    return network_type in {
        "context_structured",
        "context_structured_dueling",
        "context_select_structured",
        "context_fusion_structured",
    }


def requires_context_indices(q_network: Any) -> bool:
    return isinstance(q_network, (ContextStructuredQNetwork, ContextSelectStructuredQNetwork))


def build_q_network(
    state_size: int,
    action_size: int,
    hidden_dim: int,
    network_type: str = "mlp",
    n_genes: int = 0,
    n_modalities: int = 0,
    n_lineages: int = 0,
    cancer_context_dim: int = 16,
    fusion_query_boost: float = 2.0,
    fusion_select_weight: float = 1.0,
    select_residual_weight: float = 0.25,
) -> Any:
    torch, nn, _ = _torch_modules()
    if network_type == "mlp":
        return nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size),
        )
    if network_type == "dueling_mlp":
        return DuelingMLPQNetwork(state_size, action_size, hidden_dim)
    if network_type == "structured":
        return StructuredQNetwork(
            n_genes=n_genes,
            n_modalities=n_modalities,
            hidden_dim=hidden_dim,
            dueling=False,
        )
    if network_type == "structured_dueling":
        return StructuredQNetwork(
            n_genes=n_genes,
            n_modalities=n_modalities,
            hidden_dim=hidden_dim,
            dueling=True,
        )
    if network_type in {"context_structured", "context_structured_dueling"}:
        if n_lineages <= 0:
            raise ValueError(f"{network_type} requires n_lineages > 0.")
        return ContextStructuredQNetwork(
            n_genes=n_genes,
            n_modalities=n_modalities,
            hidden_dim=hidden_dim,
            n_lineages=n_lineages,
            context_dim=cancer_context_dim,
            dueling=network_type == "context_structured_dueling",
        )
    if network_type == "context_select_structured":
        if n_lineages <= 0:
            raise ValueError(f"{network_type} requires n_lineages > 0.")
        return ContextSelectStructuredQNetwork(
            n_genes=n_genes,
            n_modalities=n_modalities,
            hidden_dim=hidden_dim,
            n_lineages=n_lineages,
            context_dim=cancer_context_dim,
        )
    if network_type == "context_fusion_structured":
        if n_lineages <= 0:
            raise ValueError(f"{network_type} requires n_lineages > 0.")
        return ContextFusionStructuredQNetwork(
            n_genes=n_genes,
            n_modalities=n_modalities,
            hidden_dim=hidden_dim,
            n_lineages=n_lineages,
            context_dim=cancer_context_dim,
            fusion_query_boost=fusion_query_boost,
            fusion_select_weight=fusion_select_weight,
            select_residual_weight=select_residual_weight,
        )
    raise ValueError(f"Unknown Q-network type: {network_type}")


def load_structured_checkpoint_into_context(
    context_network: "ContextStructuredQNetwork | ContextSelectStructuredQNetwork",
    checkpoint_path: str | Path,
    *,
    freeze_shared_heads: bool = False,
) -> list[str]:
    """Initialize compatible layers from a trained StructuredQNetwork checkpoint."""
    torch, _, _ = _torch_modules()
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Structured checkpoint not found: {path}")
    structured_state = torch.load(path, map_location="cpu", weights_only=True)
    context_state = context_network.state_dict()
    loaded: list[str] = []

    for key in ("query_head.weight", "query_head.bias", "select_head.weight", "select_head.bias"):
        if key not in structured_state or key not in context_state:
            continue
        if structured_state[key].shape != context_state[key].shape:
            continue
        context_state[key] = structured_state[key]
        loaded.append(key)

    encoder_weight_key = "candidate_encoder.0.weight"
    encoder_bias_key = "candidate_encoder.0.bias"
    if encoder_weight_key in structured_state and encoder_weight_key in context_state:
        source_weight = structured_state[encoder_weight_key]
        target_weight = context_state[encoder_weight_key].clone()
        overlap = min(source_weight.shape[1], target_weight.shape[1])
        target_weight[:, :overlap] = source_weight[:, :overlap]
        context_state[encoder_weight_key] = target_weight
        loaded.append(encoder_weight_key)
    if encoder_bias_key in structured_state and encoder_bias_key in context_state:
        if structured_state[encoder_bias_key].shape == context_state[encoder_bias_key].shape:
            context_state[encoder_bias_key] = structured_state[encoder_bias_key]
            loaded.append(encoder_bias_key)

    encoder_hidden_key = "candidate_encoder.2.weight"
    if encoder_hidden_key in structured_state and encoder_hidden_key in context_state:
        if structured_state[encoder_hidden_key].shape == context_state[encoder_hidden_key].shape:
            context_state[encoder_hidden_key] = structured_state[encoder_hidden_key]
            loaded.append(encoder_hidden_key)
        bias_key = "candidate_encoder.2.bias"
        if bias_key in structured_state and bias_key in context_state:
            if structured_state[bias_key].shape == context_state[bias_key].shape:
                context_state[bias_key] = structured_state[bias_key]
                loaded.append(bias_key)

    context_network.load_state_dict(context_state)

    if freeze_shared_heads:
        frozen_modules: list[Any] = [
            context_network.candidate_encoder,
            context_network.query_head,
        ]
        if isinstance(context_network, ContextStructuredQNetwork):
            frozen_modules.append(context_network.select_head)
        for module in frozen_modules:
            for parameter in module.parameters():
                parameter.requires_grad = False

    return loaded


class DuelingMLPQNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden_dim, 1)
        self.advantage = nn.Linear(hidden_dim, action_size)

    def forward(self, states: Any) -> Any:
        features = self.trunk(states)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class StructuredQNetwork(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_modalities: int,
        hidden_dim: int,
        dueling: bool,
    ) -> None:
        super().__init__()
        if n_genes <= 0 or n_modalities <= 0:
            raise ValueError("Structured Q-networks require n_genes and n_modalities.")
        self.n_genes = n_genes
        self.n_modalities = n_modalities
        self.dueling = dueling
        self.per_candidate_size = n_modalities * 2 + 1
        candidate_input_size = self.per_candidate_size + 1
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        head_input_size = hidden_dim * 2
        self.query_head = nn.Linear(head_input_size, n_modalities)
        self.select_head = nn.Linear(head_input_size, 1)
        self.value_head = nn.Linear(hidden_dim + 1, 1) if dueling else None

    def forward(self, states: Any) -> Any:
        batch_size = states.shape[0]
        candidate_flat_size = self.n_genes * self.per_candidate_size
        candidate_features = states[:, :candidate_flat_size].reshape(
            batch_size,
            self.n_genes,
            self.per_candidate_size,
        )
        done = states[:, candidate_flat_size:].reshape(batch_size, 1)
        done_expanded = done.unsqueeze(1).expand(-1, self.n_genes, -1)
        candidate_input = _torch_cat((candidate_features, done_expanded), dim=2)
        encoded = self.candidate_encoder(candidate_input)
        global_context = encoded.mean(dim=1)
        global_expanded = global_context.unsqueeze(1).expand(-1, self.n_genes, -1)
        head_input = _torch_cat((encoded, global_expanded), dim=2)
        query_q = self.query_head(head_input).reshape(batch_size, self.n_genes * self.n_modalities)
        select_q = self.select_head(head_input).squeeze(-1)
        advantages = _torch_cat((query_q, select_q), dim=1)
        if not self.dueling:
            return advantages
        if self.value_head is None:
            raise RuntimeError("Dueling structured network is missing value_head.")
        value_input = _torch_cat((global_context, done), dim=1)
        value = self.value_head(value_input)
        return value + advantages - advantages.mean(dim=1, keepdim=True)


class ContextStructuredQNetwork(nn.Module):
    """Structured Q-network with a learned embedding for cancer context (e.g. lineage)."""

    def __init__(
        self,
        n_genes: int,
        n_modalities: int,
        hidden_dim: int,
        n_lineages: int,
        context_dim: int,
        dueling: bool,
    ) -> None:
        super().__init__()
        if n_genes <= 0 or n_modalities <= 0:
            raise ValueError("Context structured Q-networks require n_genes and n_modalities.")
        if n_lineages <= 0:
            raise ValueError("Context structured Q-networks require n_lineages > 0.")
        self.n_genes = n_genes
        self.n_modalities = n_modalities
        self.dueling = dueling
        self.per_candidate_size = n_modalities * 2 + 1
        self.lineage_embed = nn.Embedding(n_lineages, context_dim)
        candidate_input_size = self.per_candidate_size + 1 + context_dim
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        head_input_size = hidden_dim * 2
        self.query_head = nn.Linear(head_input_size, n_modalities)
        self.select_head = nn.Linear(head_input_size, 1)
        value_input_size = hidden_dim + 1 + context_dim if dueling else 0
        self.value_head = nn.Linear(value_input_size, 1) if dueling else None

    def forward(self, states: Any, context_indices: Any) -> Any:
        batch_size = states.shape[0]
        candidate_flat_size = self.n_genes * self.per_candidate_size
        candidate_features = states[:, :candidate_flat_size].reshape(
            batch_size,
            self.n_genes,
            self.per_candidate_size,
        )
        done = states[:, candidate_flat_size:].reshape(batch_size, 1)
        context = self.lineage_embed(context_indices)
        context_per_gene = context.unsqueeze(1).expand(-1, self.n_genes, -1)
        done_expanded = done.unsqueeze(1).expand(-1, self.n_genes, -1)
        candidate_input = _torch_cat((candidate_features, done_expanded, context_per_gene), dim=2)
        encoded = self.candidate_encoder(candidate_input)
        global_context = encoded.mean(dim=1)
        global_expanded = global_context.unsqueeze(1).expand(-1, self.n_genes, -1)
        head_input = _torch_cat((encoded, global_expanded), dim=2)
        query_q = self.query_head(head_input).reshape(batch_size, self.n_genes * self.n_modalities)
        select_q = self.select_head(head_input).squeeze(-1)
        advantages = _torch_cat((query_q, select_q), dim=1)
        if not self.dueling:
            return advantages
        if self.value_head is None:
            raise RuntimeError("Dueling context structured network is missing value_head.")
        value_input = _torch_cat((global_context, done, context), dim=1)
        value = self.value_head(value_input)
        return value + advantages - advantages.mean(dim=1, keepdim=True)


class ContextSelectStructuredQNetwork(nn.Module):
    """Structured query policy; lineage embedding modulates SELECT Q-values only."""

    def __init__(
        self,
        n_genes: int,
        n_modalities: int,
        hidden_dim: int,
        n_lineages: int,
        context_dim: int,
    ) -> None:
        super().__init__()
        if n_genes <= 0 or n_modalities <= 0:
            raise ValueError("Context select structured Q-networks require n_genes and n_modalities.")
        if n_lineages <= 0:
            raise ValueError("Context select structured Q-networks require n_lineages > 0.")
        self.n_genes = n_genes
        self.n_modalities = n_modalities
        self.per_candidate_size = n_modalities * 2 + 1
        self.lineage_embed = nn.Embedding(n_lineages, context_dim)
        candidate_input_size = self.per_candidate_size + 1
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        head_input_size = hidden_dim * 2
        self.query_head = nn.Linear(head_input_size, n_modalities)
        self.select_head = nn.Linear(head_input_size + context_dim, 1)

    def forward(self, states: Any, context_indices: Any) -> Any:
        batch_size = states.shape[0]
        candidate_flat_size = self.n_genes * self.per_candidate_size
        candidate_features = states[:, :candidate_flat_size].reshape(
            batch_size,
            self.n_genes,
            self.per_candidate_size,
        )
        done = states[:, candidate_flat_size:].reshape(batch_size, 1)
        done_expanded = done.unsqueeze(1).expand(-1, self.n_genes, -1)
        candidate_input = _torch_cat((candidate_features, done_expanded), dim=2)
        encoded = self.candidate_encoder(candidate_input)
        global_context = encoded.mean(dim=1)
        global_expanded = global_context.unsqueeze(1).expand(-1, self.n_genes, -1)
        head_input = _torch_cat((encoded, global_expanded), dim=2)
        query_q = self.query_head(head_input).reshape(batch_size, self.n_genes * self.n_modalities)
        context = self.lineage_embed(context_indices)
        context_per_gene = context.unsqueeze(1).expand(-1, self.n_genes, -1)
        select_input = _torch_cat((head_input, context_per_gene), dim=2)
        select_q = self.select_head(select_input).squeeze(-1)
        return _torch_cat((query_q, select_q), dim=1)


class ContextFusionStructuredQNetwork(ContextStructuredQNetwork):
    """Context DQN with lineage-specific modality fusion for query and selection."""

    def __init__(
        self,
        n_genes: int,
        n_modalities: int,
        hidden_dim: int,
        n_lineages: int,
        context_dim: int,
        *,
        fusion_query_boost: float = 2.0,
        fusion_select_weight: float = 1.0,
        select_residual_weight: float = 0.25,
    ) -> None:
        super().__init__(
            n_genes=n_genes,
            n_modalities=n_modalities,
            hidden_dim=hidden_dim,
            n_lineages=n_lineages,
            context_dim=context_dim,
            dueling=False,
        )
        self.modality_weight_head = nn.Linear(context_dim, n_modalities)
        self.fusion_query_boost = float(fusion_query_boost)
        self.fusion_select_weight = float(fusion_select_weight)
        self.select_residual_weight = float(select_residual_weight)

    def forward(self, states: Any, context_indices: Any) -> Any:
        batch_size = states.shape[0]
        candidate_flat_size = self.n_genes * self.per_candidate_size
        candidate_features = states[:, :candidate_flat_size].reshape(
            batch_size,
            self.n_genes,
            self.per_candidate_size,
        )
        done = states[:, candidate_flat_size:].reshape(batch_size, 1)
        context = self.lineage_embed(context_indices)
        modality_weights = torch.softmax(self.modality_weight_head(context), dim=-1)

        obs_values = candidate_features[:, :, : self.n_modalities]
        obs_masks = candidate_features[:, :, self.n_modalities : 2 * self.n_modalities]

        context_per_gene = context.unsqueeze(1).expand(-1, self.n_genes, -1)
        done_expanded = done.unsqueeze(1).expand(-1, self.n_genes, -1)
        candidate_input = _torch_cat((candidate_features, done_expanded, context_per_gene), dim=2)
        encoded = self.candidate_encoder(candidate_input)
        global_context = encoded.mean(dim=1)
        global_expanded = global_context.unsqueeze(1).expand(-1, self.n_genes, -1)
        head_input = _torch_cat((encoded, global_expanded), dim=2)
        query_q = self.query_head(head_input).reshape(
            batch_size,
            self.n_genes,
            self.n_modalities,
        )
        select_residual = self.select_head(head_input).squeeze(-1)

        weighted_obs = obs_values * obs_masks * modality_weights.unsqueeze(1)
        fused_select = weighted_obs.sum(dim=-1)
        has_obs = obs_masks.sum(dim=-1) > 0
        select_q = (
            self.fusion_select_weight * fused_select
            + self.select_residual_weight * select_residual
        )
        select_q = torch.where(
            has_obs,
            select_q,
            torch.full_like(select_q, -1.0e9),
        )

        query_q = query_q + self.fusion_query_boost * modality_weights.unsqueeze(1)
        advantages = _torch_cat((query_q.reshape(batch_size, -1), select_q), dim=1)
        return advantages


def forward_q_network(
    q_network: Any,
    states: Any,
    context_indices: Any | None = None,
) -> Any:
    if requires_context_indices(q_network):
        if context_indices is None:
            raise ValueError("context Q-networks require context_indices.")
        return q_network(states, context_indices)
    return q_network(states)


def _torch_cat(items: tuple[Any, ...], dim: int) -> Any:
    return torch.cat(items, dim=dim)


def select_epsilon_greedy_action(
    q_network: Any,
    state: np.ndarray,
    valid_actions: np.ndarray,
    epsilon: float,
    rng: Random,
    select_action_indices: np.ndarray | None = None,
    select_exploration_probability: float = 0.0,
    context_index: int = 0,
) -> int:
    valid_indices = np.flatnonzero(valid_actions)
    if len(valid_indices) == 0:
        raise ValueError("No valid actions are available.")
    if rng.random() < epsilon:
        valid_select_indices = _valid_select_indices(valid_actions, select_action_indices)
        if (
            len(valid_select_indices) > 0
            and rng.random() < select_exploration_probability
        ):
            return int(rng.choice(valid_select_indices.tolist()))
        return int(rng.choice(valid_indices.tolist()))
    return select_greedy_action(
        q_network,
        state,
        valid_actions,
        context_index=context_index,
    )


def select_greedy_action(
    q_network: Any,
    state: np.ndarray,
    valid_actions: np.ndarray,
    context_index: int = 0,
) -> int:
    torch, _, _ = _torch_modules()
    valid_indices = np.flatnonzero(valid_actions)
    if len(valid_indices) == 0:
        raise ValueError("No valid actions are available.")

    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        context_tensor = _context_tensor(context_index, q_network)
        q_values = forward_q_network(q_network, state_tensor, context_tensor).squeeze(0).detach().cpu().numpy()
    masked_values = np.full_like(q_values, fill_value=-np.inf, dtype=np.float32)
    masked_values[valid_indices] = q_values[valid_indices]
    return int(np.argmax(masked_values))


def optimize_dqn_batch(
    q_network: Any,
    target_network: Any,
    optimizer: Any,
    transitions: list[Transition],
    gamma: float,
    max_grad_norm: float | None = None,
) -> float:
    torch, nn, functional = _torch_modules()

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
    context_indices = torch.as_tensor(
        [item.context_index for item in transitions],
        dtype=torch.int64,
    )

    current_q = forward_q_network(q_network, states, context_indices).gather(1, actions).squeeze(1)
    with torch.no_grad():
        # Double DQN: online network selects the next action, target network evaluates it.
        next_online_q = forward_q_network(q_network, next_states, context_indices).masked_fill(
            ~next_valid,
            -1.0e9,
        )
        next_actions = next_online_q.argmax(dim=1, keepdim=True)
        next_target_q = forward_q_network(target_network, next_states, context_indices).gather(
            1,
            next_actions,
        ).squeeze(1)
        next_target_q = torch.where(done, torch.zeros_like(next_target_q), next_target_q)
        n_steps = torch.as_tensor([item.n_steps for item in transitions], dtype=torch.float32)
        discounts = torch.pow(torch.full_like(n_steps, gamma), n_steps)
        expected_q = rewards + discounts * next_target_q

    loss = functional.smooth_l1_loss(current_q, expected_q)
    optimizer.zero_grad()
    loss.backward()
    if max_grad_norm is not None and max_grad_norm > 0.0:
        nn.utils.clip_grad_norm_(q_network.parameters(), max_grad_norm)
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


def _context_tensor(context_index: int, q_network: Any) -> Any | None:
    if not requires_context_indices(q_network):
        return None
    torch, _, _ = _torch_modules()
    return torch.as_tensor([context_index], dtype=torch.int64)


def _valid_select_indices(
    valid_actions: np.ndarray,
    select_action_indices: np.ndarray | None,
) -> np.ndarray:
    if select_action_indices is None:
        return np.array([], dtype=np.int64)
    return np.asarray(
        [index for index in select_action_indices if valid_actions[int(index)]],
        dtype=np.int64,
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
