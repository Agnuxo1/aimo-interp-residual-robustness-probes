#!/usr/bin/env bash
# Run from the src/ directory. Set PY to the runtime-matching interpreter (see setup_env.sh).
set -euo pipefail
export AIMO_FEATURES=${AIMO_FEATURES:-features} OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore
${PY:-python} analysis.py "$@" 2>&1 | grep -v Warning
