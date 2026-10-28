#!/usr/bin/env bash
# Run from src/. Causal erasure experiment (E6). Usage: bash ../scripts/run_ablate.sh <features_dir>
set -euo pipefail
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore
export AIMO_FEATURES=${1:?features dir} OMP_NUM_THREADS=4
${PY:-python} ablate.py
