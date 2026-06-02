#!/usr/bin/env python3
"""Regenerate all poster figure sections with consistent context-sweep selection."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from poster_figures import (
    generate_ablation_storyline_figures,
    generate_context_storyline_figure,
    generate_representative_context_trajectories,
    regenerate_lineage_heatmaps,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate final poster figures.")
    parser.add_argument(
        "--context-variant",
        default="",
        help="Pin context sweep variant (default: best eval on disk).",
    )
    args = parser.parse_args()

    scripts = (
        ROOT / "scripts" / "generate_poster_architecture_diagram.py",
        ROOT / "scripts" / "generate_poster_outputs.py",
        ROOT / "scripts" / "generate_poster_context_section.py",
    )
    variant_args = ["--context-variant", args.context_variant] if args.context_variant.strip() else []

    for script in scripts:
        cmd = [sys.executable, str(script), *variant_args]
        print(f"\n=== {script.name} ===")
        subprocess.run(cmd, check=True, cwd=ROOT)

    print("\n=== poster_figures: ablation storyline ===")
    generate_ablation_storyline_figures()
    print("\n=== poster_figures: context storyline ===")
    generate_context_storyline_figure()
    print("\n=== poster_figures: representative context trajectories ===")
    generate_representative_context_trajectories()
    print("\n=== poster_figures: lineage heatmaps ===")
    regenerate_lineage_heatmaps()


if __name__ == "__main__":
    main()
