from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isnan

import pandas as pd

from src.episodes import CandidateEpisode


class ActionType(str, Enum):
    QUERY = "query"
    SELECT = "select"


@dataclass(frozen=True)
class EvidenceAction:
    action_type: ActionType
    gene_index: int
    modality_index: int | None = None


@dataclass(frozen=True)
class EvidenceState:
    episode_id: int
    cell_line_id: str
    candidate_genes: tuple[str, ...]
    modality_names: tuple[str, ...]
    observed_values: tuple[tuple[float | None, ...], ...]
    query_mask: tuple[tuple[bool, ...], ...]
    done: bool
    selected_gene: str | None = None


@dataclass(frozen=True)
class StepResult:
    state: EvidenceState
    reward: float
    done: bool
    info: dict[str, float | int | str | bool | None]


class EvidenceAcquisitionEnv:
    """Sequential evidence-acquisition environment for one candidate episode at a time."""

    def __init__(
        self,
        modalities: dict[str, pd.DataFrame],
        query_costs: dict[str, float] | None = None,
        repeated_query_penalty: float = 0.0,
        selection_reward_scale: float = 1.0,
    ) -> None:
        if not modalities:
            raise ValueError("EvidenceAcquisitionEnv requires at least one modality.")
        self.modalities = modalities
        self.modality_names = tuple(modalities.keys())
        self.query_costs = {
            name: float(query_costs[name]) if query_costs and name in query_costs else 1.0
            for name in self.modality_names
        }
        self.repeated_query_penalty = float(repeated_query_penalty)
        self.selection_reward_scale = float(selection_reward_scale)
        self._episode: CandidateEpisode | None = None
        self._observed_values: list[list[float | None]] = []
        self._query_mask: list[list[bool]] = []
        self._done = False
        self._selected_gene: str | None = None

    def reset(self, episode: CandidateEpisode) -> EvidenceState:
        self._episode = episode
        n_genes = len(episode.candidate_genes)
        n_modalities = len(self.modality_names)
        self._observed_values = [[None for _ in range(n_modalities)] for _ in range(n_genes)]
        self._query_mask = [[False for _ in range(n_modalities)] for _ in range(n_genes)]
        self._done = False
        self._selected_gene = None
        return self.state

    @property
    def state(self) -> EvidenceState:
        episode = self._require_episode()
        return EvidenceState(
            episode_id=episode.episode_id,
            cell_line_id=episode.cell_line_id,
            candidate_genes=episode.candidate_genes,
            modality_names=self.modality_names,
            observed_values=tuple(tuple(row) for row in self._observed_values),
            query_mask=tuple(tuple(row) for row in self._query_mask),
            done=self._done,
            selected_gene=self._selected_gene,
        )

    def query_action(self, gene_index: int, modality_index: int) -> EvidenceAction:
        return EvidenceAction(ActionType.QUERY, gene_index=gene_index, modality_index=modality_index)

    def select_action(self, gene_index: int) -> EvidenceAction:
        return EvidenceAction(ActionType.SELECT, gene_index=gene_index)

    def available_actions(self) -> tuple[EvidenceAction, ...]:
        if self._done:
            return ()
        episode = self._require_episode()
        actions: list[EvidenceAction] = []
        for gene_index in range(len(episode.candidate_genes)):
            for modality_index in range(len(self.modality_names)):
                if not self._query_mask[gene_index][modality_index]:
                    actions.append(self.query_action(gene_index, modality_index))
        actions.extend(self.select_action(gene_index) for gene_index in range(len(episode.candidate_genes)))
        return tuple(actions)

    def step(self, action: EvidenceAction) -> StepResult:
        if self._done:
            raise RuntimeError("Cannot step an environment after it is done. Call reset first.")
        self._validate_gene_index(action.gene_index)

        if action.action_type == ActionType.QUERY:
            return self._step_query(action)
        if action.action_type == ActionType.SELECT:
            return self._step_select(action)
        raise ValueError(f"Unknown action type: {action.action_type}")

    def _step_query(self, action: EvidenceAction) -> StepResult:
        if action.modality_index is None:
            raise ValueError("Query actions require a modality_index.")
        self._validate_modality_index(action.modality_index)

        modality_name = self.modality_names[action.modality_index]
        repeated = self._query_mask[action.gene_index][action.modality_index]
        reward = -self.query_costs[modality_name]
        if repeated:
            reward -= self.repeated_query_penalty
        else:
            self._observed_values[action.gene_index][action.modality_index] = self._lookup_value(
                action.gene_index,
                action.modality_index,
            )
            self._query_mask[action.gene_index][action.modality_index] = True

        return StepResult(
            state=self.state,
            reward=reward,
            done=False,
            info={
                "action_type": ActionType.QUERY.value,
                "gene_index": action.gene_index,
                "modality": modality_name,
                "repeated": repeated,
            },
        )

    def _step_select(self, action: EvidenceAction) -> StepResult:
        episode = self._require_episode()
        selected_dependency = float(episode.dependency_scores[action.gene_index])
        self._done = True
        self._selected_gene = episode.candidate_genes[action.gene_index]
        return StepResult(
            state=self.state,
            reward=-selected_dependency * self.selection_reward_scale,
            done=True,
            info={
                "action_type": ActionType.SELECT.value,
                "gene_index": action.gene_index,
                "selected_gene": self._selected_gene,
                "selected_dependency": selected_dependency,
            },
        )

    def _lookup_value(self, gene_index: int, modality_index: int) -> float | None:
        episode = self._require_episode()
        modality_name = self.modality_names[modality_index]
        modality = self.modalities[modality_name]
        gene = episode.candidate_genes[gene_index]
        value = _modality_value(modality, episode.cell_line_id, gene)
        if isnan(value):
            return None
        return value

    def _require_episode(self) -> CandidateEpisode:
        if self._episode is None:
            raise RuntimeError("Environment has not been reset with an episode.")
        return self._episode

    def _validate_gene_index(self, gene_index: int) -> None:
        episode = self._require_episode()
        if not 0 <= gene_index < len(episode.candidate_genes):
            raise IndexError(f"gene_index {gene_index} is out of range.")

    def _validate_modality_index(self, modality_index: int) -> None:
        if not 0 <= modality_index < len(self.modality_names):
            raise IndexError(f"modality_index {modality_index} is out of range.")

def _modality_value(modality: pd.DataFrame, cell_line_id: str, gene: str) -> float:
    value = modality.loc[cell_line_id, gene]
    if not isinstance(value, pd.Series):
        return float(value)

    numeric = pd.to_numeric(value, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.mean())
