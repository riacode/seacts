#!/usr/bin/env python3
"""Regenerate poster_outputs/03_lineage_and_context figures."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.visualization import (
    _resolve_context_sweep_eval_path,
    generate_context_dqn_modality_figure,
    generate_context_sweep_figures,
    load_context_sweep_eval_table,
    load_structured_dqn_poster_metrics,
    resolve_best_context_sweep_variant,
    resolve_context_sweep_variant_with_behavior,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "depmap_baselines"
CONTEXT_SWEEP_DIR = RESULTS / "dqn_context_sweeps"
POSTER_SECTION = ROOT / "poster_outputs" / "03_lineage_and_context"

STRUCTURED_BEHAVIOR_CANDIDATES = (
    RESULTS / "best_structured_1step_larger" / "behavior_figures",
    RESULTS / "best_structured_1step_larger" / "best_structured_1step_larger" / "behavior_figures",
    RESULTS / "dqn_sweeps" / "best_structured_1step_larger" / "behavior_figures",
)


def _first_existing_dir(candidates: tuple[Path, ...]) -> Path | None:
    for path in candidates:
        if path.is_dir():
            return path
    return None


def _structured_reward() -> float:
    try:
        structured, _ = load_structured_dqn_poster_metrics(RESULTS)
        return float(structured.iloc[0]["total_reward"])
    except FileNotFoundError:
        return 1.035


def _copy_behavior_figures(
    behavior_dir: Path,
    *,
    variant: str,
    heatmap_name: str,
    trajectory_name: str,
    label: str = "context",
) -> None:
    src = behavior_dir / "dqn_modality_usage_by_context.png"
    if src.exists():
        dst = POSTER_SECTION / heatmap_name
        shutil.copy2(src, dst)
        print(f"Copied {label} heatmap ({variant}) -> {dst}")
    src = behavior_dir / "dqn_example_trajectories.png"
    if src.exists():
        dst = POSTER_SECTION / trajectory_name
        shutil.copy2(src, dst)
        print(f"Copied {label} trajectories ({variant}) -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate poster lineage/context section.")
    parser.add_argument(
        "--context-variant",
        default="",
        help="Preferred context sweep variant (default: best eval on disk).",
    )
    args = parser.parse_args()

    POSTER_SECTION.mkdir(parents=True, exist_ok=True)

    structured_reward = _structured_reward()
    sweep_figs = generate_context_sweep_figures(
        CONTEXT_SWEEP_DIR,
        POSTER_SECTION,
        structured_total_reward=structured_reward,
    )
    if sweep_figs:
        print("Generated:", *[str(path) for path in sweep_figs], sep="\n  ")
    else:
        print("No context sweep eval metrics found; skipped context_sweep_reward_and_modality.png")

    preferred_variant = args.context_variant.strip() or None
    best_eval_variant = resolve_best_context_sweep_variant(
        CONTEXT_SWEEP_DIR,
        variant=preferred_variant,
    )
    if best_eval_variant:
        metrics_path = _resolve_context_sweep_eval_path(CONTEXT_SWEEP_DIR, best_eval_variant)
        reward = float(pd.read_csv(metrics_path).iloc[0]["total_reward"]) if metrics_path else float("nan")
        print(f"Best eval context variant: {best_eval_variant}  total_reward={reward:.4f}")
        if metrics_path is not None:
            context_metrics = pd.read_csv(metrics_path)
            try:
                structured_metrics, _ = load_structured_dqn_poster_metrics(RESULTS)
            except FileNotFoundError:
                structured_metrics = None
            modality_fig = generate_context_dqn_modality_figure(
                context_metrics,
                POSTER_SECTION,
                variant=best_eval_variant,
                structured_metrics=structured_metrics,
            )
            if modality_fig is not None:
                print(f"Generated context modality usage -> {modality_fig}")
    else:
        print("No context sweep eval metrics found.")

    structured_behavior = _first_existing_dir(STRUCTURED_BEHAVIOR_CANDIDATES)
    if structured_behavior:
        _copy_behavior_figures(
            structured_behavior,
            variant="best_structured_1step_larger",
            heatmap_name="structured_modality_by_oncotree_lineage.png",
            trajectory_name="structured_dqn_example_trajectories.png",
            label="structured",
        )

    behavior_variant, context_behavior = resolve_context_sweep_variant_with_behavior(
        CONTEXT_SWEEP_DIR,
        preferred_variant=best_eval_variant,
    )
    if context_behavior and behavior_variant:
        if behavior_variant != best_eval_variant:
            print(
                f"Lineage plots: using {behavior_variant} "
                f"(best eval {best_eval_variant} has no behavior_figures yet)"
            )
        else:
            print(f"Lineage plots: using best eval variant {behavior_variant}")
        _copy_behavior_figures(
            context_behavior,
            variant=behavior_variant,
            heatmap_name="context_dqn_modality_by_oncotree_lineage.png",
            trajectory_name="context_dqn_example_trajectories.png",
        )
    elif behavior_variant:
        metrics = _resolve_context_sweep_eval_path(CONTEXT_SWEEP_DIR, behavior_variant)
        table = load_context_sweep_eval_table(CONTEXT_SWEEP_DIR)
        ranked = ", ".join(table["variant"].astype(str).head(5))
        stale = (
            POSTER_SECTION / "context_dqn_modality_by_oncotree_lineage.png",
            POSTER_SECTION / "context_dqn_example_trajectories.png",
        )
        for path in stale:
            if path.exists():
                path.unlink()
                print(f"Removed stale poster figure (no sweep behavior logs): {path.name}")
        print(
            f"No behavior figures for any ranked context variant "
            f"(checked: {ranked}).\n"
            f"  Log behavior for the best eval run, then pull and re-run:\n"
            f"  modal run modal_log_dqn_behavior.py --variant {behavior_variant}\n"
            f"  ./scripts/pull_modal_results.sh\n"
            f"  conda run -n seacts python scripts/regenerate_poster.py"
        )
        if metrics:
            print(f"  (eval metrics exist at {metrics})")
    else:
        print("No context variant available for lineage heatmaps/trajectories.")


if __name__ == "__main__":
    main()
