from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge

from src.context_encoding import LineageContextEncoder


def build_supervised_modality_scores(
    dependency: pd.DataFrame,
    modalities: dict[str, pd.DataFrame],
    train_cell_lines: set[str] | list[str] | tuple[str, ...] | None = None,
    min_samples: int = 8,
) -> dict[str, pd.DataFrame]:
    return {
        name: _fit_modality_scores(
            dependency,
            modality,
            train_cell_lines=train_cell_lines,
            min_samples=min_samples,
        )
        for name, modality in modalities.items()
    }


def build_lineage_supervised_modality_scores(
    dependency: pd.DataFrame,
    modalities: dict[str, pd.DataFrame],
    metadata_path: str | Path,
    *,
    context_column: str = "OncotreeLineage",
    train_cell_lines: set[str] | list[str] | tuple[str, ...] | None = None,
    min_samples: int = 8,
    lineage_min_samples: int = 6,
) -> dict[str, pd.DataFrame]:
    encoder = LineageContextEncoder(metadata_path, context_column=context_column)
    global_train = (
        None
        if train_cell_lines is None
        else {str(cell_line) for cell_line in train_cell_lines}
    )
    global_scores = build_supervised_modality_scores(
        dependency,
        modalities,
        train_cell_lines=train_cell_lines,
        min_samples=min_samples,
    )
    lineages = sorted(encoder.lineage_to_index)
    scores_by_modality: dict[str, pd.DataFrame] = {}

    for modality_name, modality in modalities.items():
        merged = global_scores[modality_name].copy()
        for lineage in lineages:
            lineage_cell_lines = [
                cell_line
                for cell_line in dependency.index
                if encoder.lineage_for(str(cell_line)) == lineage
            ]
            if not lineage_cell_lines:
                continue
            lineage_train = (
                lineage_cell_lines
                if global_train is None
                else [cell_line for cell_line in lineage_cell_lines if str(cell_line) in global_train]
            )
            if len(lineage_train) < lineage_min_samples:
                continue
            lineage_scores = _fit_modality_scores(
                dependency,
                modality,
                train_cell_lines=lineage_train,
                min_samples=min(lineage_min_samples, min_samples),
            )
            merged.loc[lineage_cell_lines] = lineage_scores.loc[lineage_cell_lines]  # lineage overrides global
        scores_by_modality[modality_name] = merged

    return scores_by_modality


def _fit_modality_scores(
    dependency: pd.DataFrame,
    modality: pd.DataFrame,
    train_cell_lines: set[str] | list[str] | tuple[str, ...] | None,
    min_samples: int,
) -> pd.DataFrame:
    cell_lines = sorted(set(dependency.index) & set(modality.index))
    training_cell_lines = (
        cell_lines
        if train_cell_lines is None
        else sorted(set(cell_lines) & set(str(cell_line) for cell_line in train_cell_lines))
    )
    genes = sorted(set(dependency.columns) & set(modality.columns))
    scores = pd.DataFrame(index=cell_lines, columns=genes, dtype=float)

    for gene in genes:
        prediction_frame = pd.DataFrame(
            {
                "dependency": dependency.loc[cell_lines, gene],
                "modality": modality.loc[cell_lines, gene],
            }
        ).apply(pd.to_numeric, errors="coerce")
        training_frame = prediction_frame.loc[training_cell_lines].dropna()
        if len(training_frame) < min_samples or training_frame["modality"].nunique() <= 1:
            continue

        model = Ridge(alpha=1.0)
        model.fit(training_frame[["modality"]], training_frame["dependency"])
        predictions = model.predict(
            prediction_frame[["modality"]].fillna(training_frame["modality"].mean())  # impute missing
        )
        scores[gene] = -predictions  # invert to score

    return scores
