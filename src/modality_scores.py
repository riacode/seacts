from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge


def build_supervised_modality_scores(
    dependency: pd.DataFrame,
    modalities: dict[str, pd.DataFrame],
    train_cell_lines: set[str] | list[str] | tuple[str, ...] | None = None,
    min_samples: int = 8,
) -> dict[str, pd.DataFrame]:
    """Convert raw modality matrices into per-gene dependency evidence scores.

    Each gene gets a small one-feature ridge model: dependency ~ modality_value.
    Returned values are negative predicted dependency, so larger means stronger
    predicted dependency and can be ranked directly by existing policies.
    """
    return {
        name: _fit_modality_scores(
            dependency,
            modality,
            train_cell_lines=train_cell_lines,
            min_samples=min_samples,
        )
        for name, modality in modalities.items()
    }


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
            prediction_frame[["modality"]].fillna(training_frame["modality"].mean())
        )
        scores[gene] = -predictions

    return scores
