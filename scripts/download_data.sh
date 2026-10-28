#!/usr/bin/env bash
# Download the public AIMO-Interp label sources into ../data (not redistributed in this repo).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
HF=https://huggingface.co/datasets/aimo-interp
curl -sL $HF/train-main-v2/resolve/main/data/train-00000-of-00001.parquet -o data/train-main-v2.parquet
for d in val-sample augmented-sample-math-agg-filtered aimo-interp-challenge-sample-full; do
  curl -sL $HF/$d/resolve/main/data/validation-00000-of-00001.parquet -o data/$d.parquet
done
[ -d baselines ] || git clone -q https://github.com/aimo-interp/baselines
git -C baselines checkout -q 7e8839966750059b6b1d247a12ab56552c79b342
echo "data ready"
