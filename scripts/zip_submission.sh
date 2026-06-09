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
  scripts/ \
  tests/ \
  modal_*.py \
  -x '*__pycache__*' '*/*.pyc' '*/*.pt' '*.DS_Store'

echo "Wrote $REPO_ROOT/$OUT"
