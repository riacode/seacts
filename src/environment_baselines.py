from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from random import Random
from statistics import mean, pstdev
from typing import Protocol

import pandas as pd

from src.environment import EvidenceAcquisitionEnv
from src.episodes import CandidateEpisode
from src.metrics import hit_at_k, ndcg_at_k, reciprocal_rank_at_k


@dataclass(frozen=True)
class EnvironmentRollout:
    policy: str
    ranked_indices: tuple[int, ...]
    selected_index: int
    selected_dependency: float
    query_cost: float
    n_queries: int
    total_reward: float


class EnvironmentPolicy(Protocol):
    name: str

    def run(self, env: EvidenceAcquisitionEnv, episode: CandidateEpisode) -> EnvironmentRollout:
        ...


@dataclass
class RandomSelectPolicy:
    seed: int = 0
    name: str = "rl_env_random_select"

    def __post_init__(self) -> None:
        self.rng = Random(self.seed)

    def run(self, env: EvidenceAcquisitionEnv, episode: CandidateEpisode) -> EnvironmentRollout:
        env.reset(episode)
        ranked = list(range(len(episode.candidate_genes)))
        self.rng.shuffle(ranked)
        return _select(env, episode, self.name, ranked, query_cost=0.0, n_queries=0, query_reward=0.0)


@dataclass(frozen=True)
class OracleSelectPolicy:
    name: str = "rl_env_oracle_select"

    def run(self, env: EvidenceAcquisitionEnv, episode: CandidateEpisode) -> EnvironmentRollout:
        env.reset(episode)
        ranked = sorted(range(len(episode.dependency_scores)), key=lambda i: episode.dependency_scores[i])
        return _select(env, episode, self.name, ranked, query_cost=0.0, n_queries=0, query_reward=0.0)


@dataclass(frozen=True)
class QueryModalityPolicy:
    modality_name: str

    @property
    def name(self) -> str:
        return f"rl_env_query_{self.modality_name}_then_select"

    def run(self, env: EvidenceAcquisitionEnv, episode: CandidateEpisode) -> EnvironmentRollout:
        env.reset(episode)
        modality_index = _modality_index(env, self.modality_name)
        query_reward = 0.0
        n_queries = 0
        for gene_index in range(len(episode.candidate_genes)):
            result = env.step(env.query_action(gene_index, modality_index))
            query_reward += result.reward
            n_queries += 1

        state = env.state
        ranked = sorted(
            range(len(episode.candidate_genes)),
            key=lambda i: _rank_value(state.observed_values[i][modality_index]),
            reverse=True,
        )
        return _select(
            env,
            episode,
            self.name,
            ranked,
            query_cost=-query_reward,
            n_queries=n_queries,
            query_reward=query_reward,
        )


@dataclass(frozen=True)
class QueryAllAveragePolicy:
    name: str = "rl_env_query_all_average_then_select"

    def run(self, env: EvidenceAcquisitionEnv, episode: CandidateEpisode) -> EnvironmentRollout:
        env.reset(episode)
        query_reward = 0.0
        n_queries = 0
        for gene_index in range(len(episode.candidate_genes)):
            for modality_index in range(len(env.modality_names)):
                result = env.step(env.query_action(gene_index, modality_index))
                query_reward += result.reward
                n_queries += 1

        observed_by_modality = _transpose(env.state.observed_values)
        standardized_by_modality = [
            _standardize_observed(values) for values in observed_by_modality
        ]
        ranked = sorted(
            range(len(episode.candidate_genes)),
            key=lambda i: _average_available(
                modality_scores[i] for modality_scores in standardized_by_modality
            ),
            reverse=True,
        )
        return _select(
            env,
            episode,
            self.name,
            ranked,
            query_cost=-query_reward,
            n_queries=n_queries,
            query_reward=query_reward,
        )


def evaluate_environment_policy(
    policy: EnvironmentPolicy,
    env: EvidenceAcquisitionEnv,
    episodes: list[CandidateEpisode],
    top_k: int,
) -> dict[str, float | str]:
    if not episodes:
        raise ValueError("Cannot evaluate a policy on zero episodes.")

    rows = [policy.run(env, episode) for episode in episodes]
    return {
        "policy": policy.name,
        "selected_dependency": mean(row.selected_dependency for row in rows),
        "hit_at_k": mean(
            hit_at_k(list(episode.dependency_scores), list(row.ranked_indices), top_k)
            for episode, row in zip(episodes, rows, strict=True)
        ),
        "ndcg_at_k": mean(
            ndcg_at_k(list(episode.dependency_scores), list(row.ranked_indices), top_k)
            for episode, row in zip(episodes, rows, strict=True)
        ),
        "mrr_at_k": mean(
            reciprocal_rank_at_k(list(episode.dependency_scores), list(row.ranked_indices), top_k)
            for episode, row in zip(episodes, rows, strict=True)
        ),
        "query_cost": mean(row.query_cost for row in rows),
        "n_queries": mean(row.n_queries for row in rows),
        "total_reward": mean(row.total_reward for row in rows),
    }


def build_environment_policies(modality_names: tuple[str, ...], seed: int) -> list[EnvironmentPolicy]:
    policies: list[EnvironmentPolicy] = [
        RandomSelectPolicy(seed=seed),
        OracleSelectPolicy(),
    ]
    policies.extend(QueryModalityPolicy(name) for name in modality_names)
    policies.append(QueryAllAveragePolicy())
    return policies


def _select(
    env: EvidenceAcquisitionEnv,
    episode: CandidateEpisode,
    policy_name: str,
    ranked: list[int],
    query_cost: float,
    n_queries: int,
    query_reward: float,
) -> EnvironmentRollout:
    selected = ranked[0]
    result = env.step(env.select_action(selected))
    selected_dependency = float(episode.dependency_scores[selected])
    return EnvironmentRollout(
        policy=policy_name,
        ranked_indices=tuple(ranked),
        selected_index=selected,
        selected_dependency=selected_dependency,
        query_cost=query_cost,
        n_queries=n_queries,
        total_reward=query_reward + result.reward,
    )


def _modality_index(env: EvidenceAcquisitionEnv, modality_name: str) -> int:
    try:
        return env.modality_names.index(modality_name)
    except ValueError as error:
        raise ValueError(f"Unknown modality: {modality_name}") from error


def _rank_value(value: float | None) -> float:
    return float(value) if value is not None and pd.notna(value) else float("-inf")


def _transpose(values: tuple[tuple[float | None, ...], ...]) -> list[list[float | None]]:
    if not values:
        return []
    return [list(column) for column in zip(*values, strict=True)]


def _standardize_observed(values: list[float | None]) -> list[float | None]:
    observed = [value for value in values if value is not None]
    if not observed:
        return values
    center = mean(observed)
    scale = pstdev(observed)
    if scale == 0.0:
        return [0.0 if value is not None else None for value in values]
    return [(value - center) / scale if value is not None else None for value in values]


def _average_available(values: Iterable[float | None]) -> float:
    observed = [value for value in values if value is not None]
    if not observed:
        return float("-inf")
    return mean(observed)
