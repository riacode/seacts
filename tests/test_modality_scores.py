from __future__ import annotations

import pandas as pd

from src.modality_scores import build_supervised_modality_scores


def test_build_supervised_modality_scores_predicts_dependency_direction() -> None:
    dependency = pd.DataFrame(
        {"A": [-1.0, -0.8, 0.2, 0.3], "B": [0.2, 0.1, -0.7, -0.9]},
        index=["ACH-1", "ACH-2", "ACH-3", "ACH-4"],
    )
    expression = pd.DataFrame(
        {"A": [10.0, 9.0, 1.0, 2.0], "B": [1.0, 2.0, 8.0, 9.0]},
        index=dependency.index,
    )

    scores = build_supervised_modality_scores(
        dependency,
        {"expression": expression},
        min_samples=3,
    )

    expression_scores = scores["expression"]
    assert expression_scores.loc["ACH-1", "A"] > expression_scores.loc["ACH-3", "A"]
    assert expression_scores.loc["ACH-4", "B"] > expression_scores.loc["ACH-1", "B"]


def test_build_supervised_modality_scores_can_hold_out_prediction_cell_lines() -> None:
    dependency = pd.DataFrame(
        {"A": [-1.0, -0.9, -0.8, 0.4]},
        index=["ACH-1", "ACH-2", "ACH-3", "ACH-4"],
    )
    expression = pd.DataFrame(
        {"A": [10.0, 9.0, 8.0, 100.0]},
        index=dependency.index,
    )

    scores = build_supervised_modality_scores(
        dependency,
        {"expression": expression},
        train_cell_lines={"ACH-1", "ACH-2", "ACH-3"},
        min_samples=3,
    )

    held_out_score = scores["expression"].loc["ACH-4", "A"]
    training_score = scores["expression"].loc["ACH-1", "A"]
    assert held_out_score > training_score
