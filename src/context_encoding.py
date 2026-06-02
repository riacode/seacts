from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.behavior_analysis import CELL_LINE_COLUMNS, CONTEXT_COLUMNS


UNKNOWN_LINEAGE = "Unknown"


class LineageContextEncoder:
    """Map DepMap cell-line IDs to categorical cancer-context indices for RL."""

    def __init__(
        self,
        metadata_path: str | Path,
        context_column: str = "OncotreeLineage",
        unknown_label: str = UNKNOWN_LINEAGE,
    ) -> None:
        metadata = pd.read_csv(metadata_path)
        resolved_column = _resolve_context_column(metadata, context_column)
        metadata_cell_column = _first_present(metadata, CELL_LINE_COLUMNS)
        if metadata_cell_column is None:
            raise ValueError(
                "Metadata must include a cell-line id column such as ModelID or DepMap_ID."
            )

        cell_line_to_lineage: dict[str, str] = {}
        for _, row in metadata.iterrows():
            cell_line_id = str(row[metadata_cell_column])
            lineage = row.get(resolved_column)
            if pd.isna(lineage) or lineage == "":
                lineage = unknown_label
            cell_line_to_lineage[cell_line_id] = str(lineage)

        lineages = sorted(set(cell_line_to_lineage.values()) | {unknown_label})
        self.context_column = resolved_column
        self.unknown_label = unknown_label
        self.lineage_to_index = {lineage: index for index, lineage in enumerate(lineages)}
        self.index_to_lineage = {index: lineage for lineage, index in self.lineage_to_index.items()}
        self.cell_line_to_lineage = cell_line_to_lineage
        self.n_lineages = len(self.lineage_to_index)

    def encode(self, cell_line_id: str) -> int:
        lineage = self.cell_line_to_lineage.get(cell_line_id, self.unknown_label)
        return self.lineage_to_index[lineage]

    def lineage_for(self, cell_line_id: str) -> str:
        return self.cell_line_to_lineage.get(cell_line_id, self.unknown_label)


def _resolve_context_column(metadata: pd.DataFrame, context_column: str) -> str:
    normalized = {column.lower(): column for column in metadata.columns}
    if context_column.lower() in normalized:
        return normalized[context_column.lower()]
    resolved = _first_present(metadata, CONTEXT_COLUMNS)
    if resolved is None:
        raise ValueError(
            f"Metadata is missing context column '{context_column}' and no fallback "
            f"from {CONTEXT_COLUMNS}."
        )
    return resolved


def _first_present(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    normalized = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None
