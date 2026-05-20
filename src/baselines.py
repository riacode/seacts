from __future__ import annotations

from dataclasses import dataclass
from random import Random
from statistics import mean, pstdev
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
            value = _score_value(self.modality, episode.cell_line_id, gene)
            scores.append((idx, float(value) if pd.notna(value) else float("-inf")))
        return [idx for idx, _ in sorted(scores, key=lambda item: item[1], reverse=True)]


@dataclass
class AverageModalityPolicy:
    modalities: dict[str, pd.DataFrame]
    query_cost: float
    name: str = "average_all_modalities"

    def rank(self, episode: CandidateEpisode) -> list[int]:
        modality_scores: list[list[float | None]] = []
        for modality in self.modalities.values():
            values = [
                float(value)
                if pd.notna(value := _score_value(modality, episode.cell_line_id, gene))
                else None
                for gene in episode.candidate_genes
            ]
            modality_scores.append(_standardize_observed(values))

        scores: list[tuple[int, float]] = []
        for idx in range(len(episode.candidate_genes)):
            values = [
                modality_score[idx]
                for modality_score in modality_scores
                if modality_score[idx] is not None
            ]
            score = mean(values) if values else float("-inf")
            scores.append((idx, score))
        return [idx for idx, _ in sorted(scores, key=lambda item: item[1], reverse=True)]


def evaluate_policy(
    policy: BaselinePolicy,
    episodes: list[CandidateEpisode],
    top_k: int,
) -> dict[str, float | str]:
    if not episodes:
        raise ValueError("Cannot evaluate a policy on zero episodes.")

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


def _standardize_observed(values: list[float | None]) -> list[float | None]:
    observed = [value for value in values if value is not None]
    if not observed:
        return values
    center = mean(observed)
    scale = pstdev(observed)
    if scale == 0.0:
        return [0.0 if value is not None else None for value in values]
    return [(value - center) / scale if value is not None else None for value in values]


def _score_value(modality: pd.DataFrame, cell_line_id: str, gene: str) -> float:
    value = modality.loc[cell_line_id, gene]
    if not isinstance(value, pd.Series):
        return float(value)

    numeric = pd.to_numeric(value, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.mean())
