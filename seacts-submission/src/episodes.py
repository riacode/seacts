from __future__ import annotations

from dataclasses import dataclass
from random import Random

import pandas as pd


@dataclass(frozen=True)
class CandidateEpisode:
    episode_id: int
    cell_line_id: str
    candidate_genes: tuple[str, ...]
    dependency_scores: tuple[float, ...]

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "episode_id": self.episode_id,
                "cell_line_id": self.cell_line_id,
                "gene": self.candidate_genes,
                "dependency": self.dependency_scores,
            }
        )


class EpisodeBuilder:

    def __init__(
        self,
        dependency: pd.DataFrame,
        dependency_threshold: float,
        candidates_per_episode: int,
        positives_per_episode: int,
        min_candidates_per_cell_line: int = 1,
        seed: int = 0,
    ) -> None:
        if positives_per_episode >= candidates_per_episode:
            raise ValueError("positives_per_episode must be smaller than candidates_per_episode.")
        self.dependency = dependency
        self.dependency_threshold = dependency_threshold
        self.candidates_per_episode = candidates_per_episode
        self.positives_per_episode = positives_per_episode
        self.min_candidates_per_cell_line = min_candidates_per_cell_line
        self.rng = Random(seed)

    def build(self, n_episodes: int) -> list[CandidateEpisode]:
        eligible = self._eligible_cell_lines()
        if not eligible:
            raise ValueError("No cell lines have enough positive and negative candidate genes.")

        episodes: list[CandidateEpisode] = []
        for episode_id in range(n_episodes):
            cell_line_id = self.rng.choice(eligible)
            scores = self.dependency.loc[cell_line_id].dropna()

            positive_genes = scores[scores <= self.dependency_threshold].index.tolist()  # essential = low dep
            negative_genes = scores[scores > self.dependency_threshold].index.tolist()

            selected_positive = self.rng.sample(positive_genes, self.positives_per_episode)
            selected_negative = self.rng.sample(
                negative_genes,
                self.candidates_per_episode - self.positives_per_episode,
            )
            genes = selected_positive + selected_negative
            self.rng.shuffle(genes)  # hide label order

            episodes.append(
                CandidateEpisode(
                    episode_id=episode_id,
                    cell_line_id=str(cell_line_id),
                    candidate_genes=tuple(str(gene) for gene in genes),
                    dependency_scores=tuple(float(scores[gene]) for gene in genes),
                )
            )
        return episodes

    def _eligible_cell_lines(self) -> list[str]:
        eligible: list[str] = []
        for cell_line_id, scores in self.dependency.iterrows():
            available = scores.dropna()
            positives = int((available <= self.dependency_threshold).sum())
            negatives = int((available > self.dependency_threshold).sum())
            if (
                len(available) >= self.min_candidates_per_cell_line
                and positives >= self.positives_per_episode
                and negatives >= self.candidates_per_episode - self.positives_per_episode
            ):
                eligible.append(str(cell_line_id))
        return eligible
