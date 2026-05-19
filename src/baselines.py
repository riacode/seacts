from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Protocol

import pandas as pd

from src.episodes import CandidateEpisode
from src.metrics import hit_at_k, ndcg_at_k, reciprocal_rank_at_k


class BaselinePolicy(Protocol):
    name: str
    query_cost: float

    def rank(self, episode: CandidateEpisode) -> list[int]:
        ...


@dataclass
class RandomPolicy:
    seed: int = 0
    name: str = "random"
    query_cost: float = 0.0

    def __post_init__(self) -> None:
        self.rng = Random(self.seed)

    def rank(self, episode: CandidateEpisode) -> list[int]:
        indices = list(range(len(episode.candidate_genes)))
        self.rng.shuffle(indices)
        return indices


@dataclass
class OraclePolicy:
    name: str = "oracle_dependency"
    query_cost: float = 0.0

    def rank(self, episode: CandidateEpisode) -> list[int]:
        return sorted(range(len(episode.dependency_scores)), key=lambda i: episode.dependency_scores[i])


@dataclass
class ModalityScorePolicy:
    modality_name: str
    modality: pd.DataFrame
    query_cost: float

    @property
    def name(self) -> str:
        return f"{self.modality_name}_score"

    def rank(self, episode: CandidateEpisode) -> list[int]:
        scores = []
        for idx, gene in enumerate(episode.candidate_genes):
            value = self.modality.loc[episode.cell_line_id, gene]
            scores.append((idx, float(value) if pd.notna(value) else float("-inf")))
        return [idx for idx, _ in sorted(scores, key=lambda item: item[1], reverse=True)]


@dataclass
class AverageModalityPolicy:
    modalities: dict[str, pd.DataFrame]
    query_cost: float
    name: str = "average_all_modalities"

    def rank(self, episode: CandidateEpisode) -> list[int]:
        scores = []
        for idx, gene in enumerate(episode.candidate_genes):
            values = [
                float(modality.loc[episode.cell_line_id, gene])
                for modality in self.modalities.values()
                if pd.notna(modality.loc[episode.cell_line_id, gene])
            ]
            score = sum(values) / len(values) if values else float("-inf")
            scores.append((idx, score))
        return [idx for idx, _ in sorted(scores, key=lambda item: item[1], reverse=True)]


def evaluate_policy(
    policy: BaselinePolicy,
    episodes: list[CandidateEpisode],
    top_k: int,
) -> dict[str, float | str]:
    rows: list[dict[str, float]] = []
    for episode in episodes:
        ranked = policy.rank(episode)
        dependencies = list(episode.dependency_scores)
        selected = ranked[0]
        rows.append(
            {
                "selected_dependency": dependencies[selected],
                "hit_at_k": hit_at_k(dependencies, ranked, top_k),
                "ndcg_at_k": ndcg_at_k(dependencies, ranked, top_k),
                "mrr_at_k": reciprocal_rank_at_k(dependencies, ranked, top_k),
                "query_cost": policy.query_cost,
            }
        )

    summary = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
    return {"policy": policy.name, **summary}
