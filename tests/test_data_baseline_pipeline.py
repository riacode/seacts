from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_baselines import AverageModalityPolicy, ModalityScorePolicy, OraclePolicy, evaluate_policy
from src.config import load_baseline_config
from src.data import load_project_data, read_cell_line_by_gene_matrix
from src.episodes import CandidateEpisode


def test_read_matrix_normalizes_depmap_gene_suffixes(tmp_path: Path) -> None:
    path = tmp_path / "matrix.csv"
    pd.DataFrame(
        {
            "DepMap_ID": ["ACH-1", "ACH-2"],
            "SOX10 (6663)": [1.0, 2.0],
            "BRAF (673)": [3.0, 4.0],
        }
    ).to_csv(path, index=False)

    matrix = read_cell_line_by_gene_matrix(path)

    assert matrix.index.tolist() == ["ACH-1", "ACH-2"]
    assert matrix.columns.tolist() == ["SOX10", "BRAF"]


def test_read_matrix_rejects_duplicate_gene_names_after_normalization(tmp_path: Path) -> None:
    path = tmp_path / "matrix.csv"
    pd.DataFrame(
        {
            "DepMap_ID": ["ACH-1"],
            "GENE (1)": [1.0],
            "GENE (2)": [2.0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate columns"):
        read_cell_line_by_gene_matrix(path)


def test_read_matrix_averages_duplicate_cell_line_rows(tmp_path: Path) -> None:
    path = tmp_path / "matrix.csv"
    pd.DataFrame(
        {
            "DepMap_ID": ["ACH-1", "ACH-1"],
            "SOX10 (6663)": [1.0, 3.0],
        }
    ).to_csv(path, index=False)

    matrix = read_cell_line_by_gene_matrix(path)

    assert matrix.index.tolist() == ["ACH-1"]
    assert matrix.loc["ACH-1", "SOX10"] == 2.0


def test_read_matrix_filters_default_omics_rows_and_drops_id_columns(tmp_path: Path) -> None:
    path = tmp_path / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
    pd.DataFrame(
        {
            "ModelID": ["ACH-1", "ACH-1", "ACH-2"],
            "ModelConditionID": ["MC-1", "MC-2", "MC-3"],
            "SequencingID": ["CDS-1", "CDS-2", "CDS-3"],
            "IsDefaultEntryForModel": ["No", "Yes", "Yes"],
            "SOX10 (6663)": [100.0, 2.0, 4.0],
        }
    ).to_csv(path, index=False)

    matrix = read_cell_line_by_gene_matrix(path)

    assert matrix.index.tolist() == ["ACH-1", "ACH-2"]
    assert matrix.columns.tolist() == ["SOX10"]
    assert matrix.loc["ACH-1", "SOX10"] == 2.0


def test_load_project_data_intersects_normalized_gene_names(tmp_path: Path) -> None:
    dependency_path = tmp_path / "CRISPRGeneEffect.csv"
    expression_path = tmp_path / "expression.csv"
    pd.DataFrame(
        {
            "DepMap_ID": ["ACH-1"],
            "SOX10 (6663)": [-1.0],
            "BRAF (673)": [0.2],
        }
    ).to_csv(dependency_path, index=False)
    pd.DataFrame(
        {
            "DepMap_ID": ["ACH-1"],
            "SOX10": [5.0],
            "BRAF": [1.0],
        }
    ).to_csv(expression_path, index=False)

    data = load_project_data(dependency_path, {"expression": expression_path})

    assert data.genes == ["BRAF", "SOX10"]
    assert data.dependency.loc["ACH-1", "SOX10"] == -1.0


def test_average_modality_policy_standardizes_modalities_before_averaging() -> None:
    episode = CandidateEpisode(
        episode_id=0,
        cell_line_id="ACH-1",
        candidate_genes=("A", "B", "C"),
        dependency_scores=(-1.0, 0.2, 0.3),
    )
    high_scale = pd.DataFrame({"A": [1000.0], "B": [900.0], "C": [0.0]}, index=["ACH-1"])
    low_scale = pd.DataFrame({"A": [0.0], "B": [1.0], "C": [1.0]}, index=["ACH-1"])
    policy = AverageModalityPolicy({"high": high_scale, "low": low_scale}, query_cost=2.0)

    assert policy.rank(episode)[0] == 1


def test_modality_score_policy_handles_duplicate_lookup_series() -> None:
    episode = CandidateEpisode(
        episode_id=0,
        cell_line_id="ACH-1",
        candidate_genes=("A", "B"),
        dependency_scores=(-1.0, 0.2),
    )
    modality = pd.DataFrame({"A": [1.0, 3.0], "B": [5.0, 1.0]}, index=["ACH-1", "ACH-1"])
    policy = ModalityScorePolicy("test", modality, query_cost=1.0)

    assert policy.rank(episode) == [1, 0]


def test_evaluate_policy_rejects_empty_episode_list() -> None:
    with pytest.raises(ValueError, match="zero episodes"):
        evaluate_policy(OraclePolicy(), [], top_k=3)


def test_baseline_config_loads_environment_costs() -> None:
    config = load_baseline_config("configs/depmap_baselines.yaml")

    assert config.environment.use_supervised_modality_scores
    assert config.environment.query_costs == {
        "expression": 0.02,
        "cna": 0.03,
        "damaging_mutation": 0.04,
        "hotspot_mutation": 0.04,
    }
    assert config.environment.repeated_query_penalty == 0.0
