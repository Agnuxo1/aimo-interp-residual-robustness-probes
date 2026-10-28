#!/usr/bin/env bash
# Run from the src/ directory. Set PY to the runtime-matching interpreter (see setup_env.sh).
set -euo pipefail
export AIMO_FEATURES=${AIMO_FEATURES:-features} OMP_NUM_THREADS=4
true
ls -la $AIMO_FEATURES
${PY:-python} cv_probe.py --seeds 5 "$@"
