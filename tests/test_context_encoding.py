from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.context_encoding import LineageContextEncoder, UNKNOWN_LINEAGE


def test_lineage_context_encoder_maps_cell_lines(tmp_path: Path) -> None:
    metadata_path = tmp_path / "Model.csv"
    pd.DataFrame(
        {
            "ModelID": ["ACH-1", "ACH-2", "ACH-3"],
            "OncotreeLineage": ["Lung", "Lung", "Lymphoid"],
        }
    ).to_csv(metadata_path, index=False)

    encoder = LineageContextEncoder(metadata_path)
    assert encoder.encode("ACH-1") == encoder.encode("ACH-2")
    assert encoder.encode("ACH-1") != encoder.encode("ACH-3")
    assert encoder.lineage_for("ACH-99") == UNKNOWN_LINEAGE


def test_lineage_context_encoder_requires_metadata_column(tmp_path: Path) -> None:
    metadata_path = tmp_path / "Model.csv"
    pd.DataFrame({"ModelID": ["ACH-1"], "Other": ["x"]}).to_csv(metadata_path, index=False)
    with pytest.raises(ValueError, match="context column"):
        LineageContextEncoder(metadata_path, context_column="MissingColumn")
