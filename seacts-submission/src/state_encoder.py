from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.environment import ActionType, EvidenceAction, EvidenceState


@dataclass(frozen=True)
class ActionSpace:
    n_genes: int
    n_modalities: int

    @property
    def size(self) -> int:
        return self.n_genes * self.n_modalities + self.n_genes

    def to_index(self, action: EvidenceAction) -> int:
        if action.action_type == ActionType.QUERY:
            if action.modality_index is None:
                raise ValueError("Query actions require a modality index.")
            return action.gene_index * self.n_modalities + action.modality_index
        if action.action_type == ActionType.SELECT:
            return self.n_genes * self.n_modalities + action.gene_index  # after query block
        raise ValueError(f"Unknown action type: {action.action_type}")

    def from_index(self, index: int) -> EvidenceAction:
        if not 0 <= index < self.size:
            raise IndexError(f"Action index {index} is out of range.")
        query_actions = self.n_genes * self.n_modalities
        if index < query_actions:
            gene_index, modality_index = divmod(index, self.n_modalities)
            return EvidenceAction(
                action_type=ActionType.QUERY,
                gene_index=gene_index,
                modality_index=modality_index,
            )
        return EvidenceAction(
            action_type=ActionType.SELECT,
            gene_index=index - query_actions,
        )

    def select_indices(self) -> np.ndarray:
        return np.arange(
            self.n_genes * self.n_modalities,
            self.size,
            dtype=np.int64,
        )


@dataclass(frozen=True)
class StateEncoder:
    n_genes: int
    n_modalities: int

    @property
    def action_space(self) -> ActionSpace:
        return ActionSpace(self.n_genes, self.n_modalities)

    @property
    def state_size(self) -> int:
        per_candidate_features = self.n_modalities * 2 + 1  # values, masks, slot
        return self.n_genes * per_candidate_features + 1

    def encode(self, state: EvidenceState) -> np.ndarray:
        _validate_state_shape(state, self.n_genes, self.n_modalities)
        features: list[float] = []
        denominator = max(self.n_genes - 1, 1)
        for gene_index, (values, mask_row) in enumerate(
            zip(state.observed_values, state.query_mask, strict=True)
        ):
            features.extend(_observed_value(value) for value in values)
            features.extend(1.0 if observed else 0.0 for observed in mask_row)
            features.append(gene_index / denominator)
        features.append(1.0 if state.done else 0.0)
        return np.asarray(features, dtype=np.float32)

    def valid_action_mask(self, state: EvidenceState) -> np.ndarray:
        _validate_state_shape(state, self.n_genes, self.n_modalities)
        mask = np.zeros(self.action_space.size, dtype=bool)
        if state.done:
            return mask

        for gene_index, query_row in enumerate(state.query_mask):
            for modality_index, already_queried in enumerate(query_row):
                if not already_queried:
                    mask[gene_index * self.n_modalities + modality_index] = True
        mask[self.action_space.select_indices()] = True  # SELECT always valid
        return mask


def _observed_value(value: float | None) -> float:
    if value is None or np.isnan(value):
        return 0.0
    return float(value)


def _validate_state_shape(state: EvidenceState, n_genes: int, n_modalities: int) -> None:
    if len(state.candidate_genes) != n_genes:
        raise ValueError(
            f"Expected {n_genes} candidate genes, got {len(state.candidate_genes)}."
        )
    if len(state.modality_names) != n_modalities:
        raise ValueError(f"Expected {n_modalities} modalities, got {len(state.modality_names)}.")
