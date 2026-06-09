from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random

import pandas as pd
import yaml

from src.config import BaselineConfig
from src.episodes import EpisodeBuilder


@dataclass(frozen=True)
class CellLineSplitConfig:
    enabled: bool = False
    validation_fraction: float = 0.1
    eval_fraction: float = 0.1


def load_cell_line_split_config(config_path: str | Path) -> CellLineSplitConfig:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    training = raw.get("rl_training", {})
    return CellLineSplitConfig(
        enabled=bool(training.get("split_cell_lines", False)),
        validation_fraction=float(training.get("validation_cell_line_fraction", 0.1)),
        eval_fraction=float(training.get("eval_cell_line_fraction", 0.1)),
    )


def split_dependency_by_cell_line(
    config: BaselineConfig,
    dependency: pd.DataFrame,
    seed: int,
    validation_fraction: float,
    eval_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = eligible_cell_lines_for_episodes(config, dependency)
    if len(eligible) < 3:
        raise ValueError("Cell-line splitting requires at least three eligible cell lines.")

    rng = Random(seed)
    shuffled = eligible.copy()
    rng.shuffle(shuffled)

    n_eval = _fraction_count(len(shuffled), eval_fraction)
    n_validation = _fraction_count(len(shuffled) - n_eval, validation_fraction)
    if n_eval + n_validation >= len(shuffled):
        raise ValueError("Validation/eval cell-line fractions leave no training cell lines.")

    eval_cell_lines = shuffled[:n_eval]
    validation_cell_lines = shuffled[n_eval : n_eval + n_validation]
    train_cell_lines = shuffled[n_eval + n_validation :]
    return (
        dependency.loc[sorted(train_cell_lines)],
        dependency.loc[sorted(validation_cell_lines)],
        dependency.loc[sorted(eval_cell_lines)],
    )


def maybe_split_dependency_by_cell_line(
    config: BaselineConfig,
    dependency: pd.DataFrame,
    split_config: CellLineSplitConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not split_config.enabled:
        return dependency, dependency, dependency
    return split_dependency_by_cell_line(
        config,
        dependency,
        seed=config.seed,
        validation_fraction=split_config.validation_fraction,
        eval_fraction=split_config.eval_fraction,
    )


def eligible_cell_lines_for_episodes(
    config: BaselineConfig,
    dependency: pd.DataFrame,
) -> list[str]:
    builder = EpisodeBuilder(
        dependency=dependency,
        dependency_threshold=config.episodes.dependency_threshold,
        candidates_per_episode=config.episodes.candidates_per_episode,
        positives_per_episode=config.episodes.positives_per_episode,
        min_candidates_per_cell_line=config.episodes.min_candidates_per_cell_line,
        seed=config.seed,
    )
    return builder._eligible_cell_lines()


def _fraction_count(n_items: int, fraction: float) -> int:
    if fraction <= 0.0:
        return 0
    return max(1, int(round(n_items * fraction)))
