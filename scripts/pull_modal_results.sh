#!/usr/bin/env bash
# Pull new/updated files from Modal seacts-results into ./outputs (additive only).
# Does not delete existing local figures or results.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VOLUME="seacts-results"
OUT="outputs"
BASE="$OUT/depmap_baselines"
FIG="$OUT/figures"
STAGING="$(mktemp -d "${TMPDIR:-/tmp}/seacts-pull.XXXXXX")"

cleanup() {
  rm -rf "$STAGING"
}
trap cleanup EXIT

mkdir -p "$BASE" "$FIG"

_merge_tree() {
  # Copy src/ into dest/; overwrite same paths, keep files only present in dest.
  local src="$1"
  local dest="$2"
  mkdir -p "$dest"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    rsync -a "$src"/ "$dest"/
  else
    rsync -a "$src"/ "$dest"/
  fi
}

echo "==> Poster / comparison figures (modal_visualizations)"
mkdir -p "$STAGING/figures"
modal volume get "$VOLUME" figures "$STAGING"
if [[ -d "$STAGING/figures/figures" ]]; then
  _merge_tree "$STAGING/figures/figures" "$FIG"
  _merge_tree "$STAGING/figures" "$FIG"
else
  _merge_tree "$STAGING/figures" "$FIG"
fi

echo "==> Architecture ablation figure subfolder (merge)"
mkdir -p "$STAGING/fig-ablation"
modal volume get "$VOLUME" figures/dqn_ablation "$STAGING/fig-ablation"
_merge_tree "$STAGING/fig-ablation" "$FIG/dqn_ablation"

echo "==> Cancer-context extension figures (merge)"
mkdir -p "$STAGING/fig-cancer"
modal volume get "$VOLUME" figures/dqn_cancer_context "$STAGING/fig-cancer"
_merge_tree "$STAGING/fig-cancer" "$FIG/dqn_cancer_context"

echo "==> Cancer-context training + behavior (merge)"
mkdir -p "$STAGING/cancer"
modal volume get "$VOLUME" depmap_baselines/dqn_cancer_context "$STAGING/cancer"
_merge_tree "$STAGING/cancer" "$BASE/dqn_cancer_context"

echo "==> Context DQN sweep metrics (merge)"
mkdir -p "$STAGING/ctx-sweep" "$BASE/dqn_context_sweeps"
if modal volume get "$VOLUME" depmap_baselines/dqn_context_sweeps "$STAGING/ctx-sweep" 2>/dev/null; then
  if [[ -d "$STAGING/ctx-sweep/dqn_context_sweeps" ]]; then
    _merge_tree "$STAGING/ctx-sweep/dqn_context_sweeps" "$BASE/dqn_context_sweeps"
  else
    _merge_tree "$STAGING/ctx-sweep" "$BASE/dqn_context_sweeps"
  fi
fi

echo "==> Structured sweep winner (merge)"
mkdir -p "$STAGING/sweep"
modal volume get "$VOLUME" depmap_baselines/best_structured_1step_larger "$STAGING/sweep"
_merge_tree "$STAGING/sweep" "$BASE/best_structured_1step_larger"

echo "==> Context DQN sweep behavior logs (merge)"
CTX_VARIANT="ctx_larger_init_structured"
CTX_DIR="$BASE/dqn_context_sweeps/$CTX_VARIANT"
mkdir -p "$STAGING/ctx-behavior" "$CTX_DIR/behavior_figures"
if modal volume get "$VOLUME" "depmap_baselines/dqn_context_sweeps/$CTX_VARIANT/behavior_figures" \
  "$STAGING/ctx-behavior" 2>/dev/null; then
  if [[ -d "$STAGING/ctx-behavior/behavior_figures" ]]; then
    _merge_tree "$STAGING/ctx-behavior/behavior_figures" "$CTX_DIR/behavior_figures"
  elif [[ -d "$STAGING/ctx-behavior" ]]; then
    _merge_tree "$STAGING/ctx-behavior" "$CTX_DIR/behavior_figures"
  fi
fi
mkdir -p "$STAGING/ctx-analysis" "$CTX_DIR/behavior_analysis"
if modal volume get "$VOLUME" "depmap_baselines/dqn_context_sweeps/$CTX_VARIANT/behavior_analysis" \
  "$STAGING/ctx-analysis" 2>/dev/null; then
  if [[ -d "$STAGING/ctx-analysis/behavior_analysis" ]]; then
    _merge_tree "$STAGING/ctx-analysis/behavior_analysis" "$CTX_DIR/behavior_analysis"
  elif [[ -d "$STAGING/ctx-analysis" ]]; then
    _merge_tree "$STAGING/ctx-analysis" "$CTX_DIR/behavior_analysis"
  fi
fi

echo "==> Optional: sweep folder on volume (skip if missing)"
mkdir -p "$STAGING/sweeps" "$BASE/dqn_sweeps"
if modal volume get "$VOLUME" depmap_baselines/dqn_sweeps/best_structured_1step_larger \
  "$STAGING/sweeps" 2>/dev/null; then
  _merge_tree "$STAGING/sweeps" "$BASE/dqn_sweeps"
fi

echo ""
echo "Done (additive merge into $OUT). New/updated PNGs:"
find "$OUT" -name "*.png" | sort
