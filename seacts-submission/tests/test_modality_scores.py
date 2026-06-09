from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.modality_scores import build_lineage_supervised_modality_scores


def test_build_lineage_supervised_modality_scores_uses_lineage_models(tmp_path: Path) -> None:
    metadata_path = tmp_path / "Model.csv"
    pd.DataFrame(
        {
            "ModelID": ["CL-A", "CL-B", "CL-C", "CL-D"],
            "OncotreeLineage": ["Lung", "Lung", "Lymphoid", "Lymphoid"],
        }
    ).to_csv(metadata_path, index=False)

    dependency = pd.DataFrame(
        {
            "G1": [-1.0, -0.8, -0.2, -0.1],
            "G2": [-0.5, -0.4, -1.2, -1.0],
        },
        index=["CL-A", "CL-B", "CL-C", "CL-D"],
    )
    expression = pd.DataFrame(
        {
            "G1": [4.0, 3.5, 1.0, 0.5],
            "G2": [1.0, 0.5, 5.0, 4.5],
        },
        index=["CL-A", "CL-B", "CL-C", "CL-D"],
    )

    scores = build_lineage_supervised_modality_scores(
        dependency,
        {"expression": expression},
        metadata_path,
        train_cell_lines={"CL-A", "CL-B", "CL-C", "CL-D"},
        lineage_min_samples=2,
    )
    frame = scores["expression"]
    assert frame.loc["CL-A", "G1"] > frame.loc["CL-C", "G1"]
    assert frame.loc["CL-C", "G2"] > frame.loc["CL-A", "G2"]
