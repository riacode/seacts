#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.visualization import (
    append_cancer_context_dqn_metrics,
    append_context_sweep_metrics,
    generate_baseline_figures,
    load_structured_dqn_poster_metrics,
    _resolve_context_sweep_eval_path,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "outputs" / "depmap_baselines"
CANCER_CONTEXT_DIR = RESULTS / "dqn_cancer_context"
CONTEXT_SWEEP_DIR = RESULTS / "dqn_context_sweeps"
POSTER_MAIN = ROOT / "poster_outputs" / "01_main_results"

POSTER_MAIN_FIGURES = (
    "environment_total_reward.png",
    "baseline_ranking_metrics.png",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate poster main-results figures.")
    parser.add_argument(
        "--context-variant",
        default="",
        help="Context sweep folder name (e.g. ctx_larger_ctx32). Default: best on disk.",
    )
    args = parser.parse_args()

    dqn_results, trajectory_path = load_structured_dqn_poster_metrics(RESULTS)
    sweep_variant = args.context_variant.strip() or None
    dqn_results, used_sweep = append_context_sweep_metrics(
        dqn_results,
        CONTEXT_SWEEP_DIR,
        variant=sweep_variant,
    )
    if used_sweep:
        print(f"Context DQN from sweep variant: {used_sweep}")
        metrics_path = _resolve_context_sweep_eval_path(CONTEXT_SWEEP_DIR, used_sweep)
        if metrics_path is not None:
            reward = float(pd.read_csv(metrics_path).iloc[0]["total_reward"])
            print(f"  total_reward={reward:.4f}  ({metrics_path})")
    else:
        dqn_results = append_cancer_context_dqn_metrics(
            dqn_results,
            CANCER_CONTEXT_DIR,
            variants=("larger",),
        )
        print("Context DQN from dqn_cancer_context/larger (no sweep metrics found)")

    tmp_dir = ROOT / "poster_outputs" / "_tmp_main_results"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    figures = generate_baseline_figures(
        data_metrics_path=RESULTS / "data_baseline_metrics.csv",
        environment_metrics_path=RESULTS / "environment_baseline_metrics.csv",
        dqn_results=dqn_results,
        dqn_trajectory_path=trajectory_path,
        output_dir=tmp_dir,
        poster_subset=True,
    )
    print("Generated:", *[str(f) for f in figures], sep="\n  ")

    POSTER_MAIN.mkdir(parents=True, exist_ok=True)
    for name in POSTER_MAIN_FIGURES:
        src = tmp_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Expected figure not produced: {src}")
        dst = POSTER_MAIN / name
        shutil.copy2(src, dst)
        print(f"Copied -> {dst}")

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
