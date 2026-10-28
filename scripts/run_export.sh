#!/usr/bin/env bash
# Run from the src/ directory. Set PY to the runtime-matching interpreter (see setup_env.sh).
set -euo pipefail
export AIMO_FEATURES=${AIMO_FEATURES:-features} OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore
rm -rf artifacts
${PY:-python} export_probe.py "$@"
ls -la artifacts
