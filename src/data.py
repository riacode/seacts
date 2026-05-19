from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


CELL_LINE_COLS = ("cell_line_id", "DepMap_ID", "model_id")
GENE_COLS = ("gene", "gene_symbol", "hugo_symbol")
VALUE_COLS = ("value", "score", "dependency", "effect")


@dataclass(frozen=True)
class ProjectData:
    dependency: pd.DataFrame
    modalities: dict[str, pd.DataFrame]
    metadata: pd.DataFrame | None = None

    @property
    def cell_lines(self) -> list[str]:
        return sorted(self.dependency.index.astype(str).tolist())

    @property
    def genes(self) -> list[str]:
        return sorted(self.dependency.columns.astype(str).tolist())


def _first_present(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    normalized = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def read_cell_line_by_gene_matrix(path: str | Path) -> pd.DataFrame:
    """Read either a wide cell-line-by-gene matrix or a long table into a matrix."""
    frame = pd.read_csv(path)
    cell_col = _first_present(frame.columns, CELL_LINE_COLS)
    gene_col = _first_present(frame.columns, GENE_COLS)
    value_col = _first_present(frame.columns, VALUE_COLS)

    if cell_col and gene_col and value_col:
        matrix = frame.pivot_table(index=cell_col, columns=gene_col, values=value_col, aggfunc="mean")
    elif cell_col:
        matrix = frame.set_index(cell_col)
    else:
        matrix = frame.set_index(frame.columns[0])

    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    return matrix.apply(pd.to_numeric, errors="coerce")


def load_project_data(
    dependency_path: str | Path,
    modality_paths: dict[str, str | Path],
    metadata_path: str | Path | None = None,
) -> ProjectData:
    dependency_path = Path(dependency_path)
    if not dependency_path.exists():
        raise FileNotFoundError(f"Dependency matrix not found: {dependency_path}")

    missing_modalities = {
        name: Path(path) for name, path in modality_paths.items() if not Path(path).exists()
    }
    if missing_modalities:
        missing = ", ".join(f"{name}={path}" for name, path in missing_modalities.items())
        raise FileNotFoundError(f"Configured modality file(s) not found: {missing}")

    dependency = read_cell_line_by_gene_matrix(dependency_path)
    modalities = {
        name: read_cell_line_by_gene_matrix(path)
        for name, path in modality_paths.items()
    }

    metadata = None
    if metadata_path is not None and Path(metadata_path).exists():
        metadata = pd.read_csv(metadata_path)

    common_cell_lines = set(dependency.index)
    common_genes = set(dependency.columns)
    for modality in modalities.values():
        common_cell_lines &= set(modality.index)
        common_genes &= set(modality.columns)

    if not common_cell_lines:
        raise ValueError("No overlapping cell lines between dependency data and modalities.")
    if not common_genes:
        raise ValueError("No overlapping genes between dependency data and modalities.")

    cell_lines = sorted(common_cell_lines)
    genes = sorted(common_genes)
    return ProjectData(
        dependency=dependency.loc[cell_lines, genes],
        modalities={name: modality.loc[cell_lines, genes] for name, modality in modalities.items()},
        metadata=metadata,
    )
