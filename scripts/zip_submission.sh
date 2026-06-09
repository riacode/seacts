#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT="${1:-seacts-submission.zip}"

zip -r "$OUT" \
  README.md \
  report.md \
  pyproject.toml \
  requirements.txt \
  setup.py \
  conda_env_local.yml \
  conda_env_modal.yml \
  configs/ \
  data/README.md \
  src/ \
  scripts/local/ \
  scripts/zip_submission.sh \
  tests/ \
  modal_ablate_dqn.py \
  modal_cancer_context_dqn.py \
  modal_data.py \
  modal_data_baselines.py \
  modal_environment_baselines.py \
  modal_log_dqn_behavior.py \
  modal_sweep_context_dqn.py \
  modal_sweep_dqn.py \
  modal_train_dqn.py \
  modal_visualizations.py \
  -x '*__pycache__*' '*/*.pyc' '*/*.pt' '*.DS_Store'

echo "Wrote $REPO_ROOT/$OUT"
