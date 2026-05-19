from __future__ import annotations

from math import log2
from statistics import mean


def dependency_to_relevance(dependency_score: float) -> float:
    """Convert DepMap-style dependency scores to positive relevance.

    DepMap CRISPR effects are usually more dependent when more negative, so -score is a
    convenient ranking relevance for candidate-set metrics.
    """
    return -float(dependency_score)


def hit_at_k(dependencies: list[float], ranked_indices: list[int], k: int) -> float:
    top_true = set(sorted(range(len(dependencies)), key=lambda i: dependencies[i])[:k])
    top_pred = set(ranked_indices[:k])
    return float(bool(top_true & top_pred))


def ndcg_at_k(dependencies: list[float], ranked_indices: list[int], k: int) -> float:
    def dcg(indices: list[int]) -> float:
        total = 0.0
        for rank, idx in enumerate(indices[:k], start=1):
            relevance = max(dependency_to_relevance(dependencies[idx]), 0.0)
            total += relevance / log2(rank + 1)
        return total

    ideal = sorted(range(len(dependencies)), key=lambda i: dependencies[i])
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return dcg(ranked_indices) / ideal_dcg


def reciprocal_rank_at_k(dependencies: list[float], ranked_indices: list[int], k: int) -> float:
    best_gene = min(range(len(dependencies)), key=lambda i: dependencies[i])
    for rank, idx in enumerate(ranked_indices[:k], start=1):
        if idx == best_gene:
            return 1.0 / rank
    return 0.0


def summarize_selection_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: mean(float(row[key]) for row in rows) for key in keys}
