from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    dependency_path: Path
    metadata_path: Path | None
    modalities: dict[str, Path]


@dataclass(frozen=True)
class EpisodeConfig:
    n_episodes: int
    candidates_per_episode: int
    positives_per_episode: int
    dependency_threshold: float
    min_candidates_per_cell_line: int


@dataclass(frozen=True)
class EvaluationConfig:
    top_k: int
    full_query_cost: float


@dataclass(frozen=True)
class BaselineConfig:
    seed: int
    data: DataConfig
    episodes: EpisodeConfig
    evaluation: EvaluationConfig
    output_dir: Path


def _resolve_path(config_path: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent.parent / path).resolve()


def load_baseline_config(path: str | Path) -> BaselineConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    data = raw["data"]
    episodes = raw["episodes"]
    evaluation = raw["evaluation"]

    return BaselineConfig(
        seed=int(raw.get("seed", 0)),
        data=DataConfig(
            dependency_path=_resolve_path(config_path, data["dependency_path"]),
            metadata_path=_resolve_path(config_path, data.get("metadata_path")),
            modalities={
                name: _resolve_path(config_path, modality_path)
                for name, modality_path in data.get("modalities", {}).items()
            },
        ),
        episodes=EpisodeConfig(
            n_episodes=int(episodes["n_episodes"]),
            candidates_per_episode=int(episodes["candidates_per_episode"]),
            positives_per_episode=int(episodes["positives_per_episode"]),
            dependency_threshold=float(episodes["dependency_threshold"]),
            min_candidates_per_cell_line=int(episodes.get("min_candidates_per_cell_line", 1)),
        ),
        evaluation=EvaluationConfig(
            top_k=int(evaluation.get("top_k", 3)),
            full_query_cost=float(evaluation.get("full_query_cost", 0.0)),
        ),
        output_dir=_resolve_path(config_path, raw.get("output_dir", "outputs/baselines")),
    )
