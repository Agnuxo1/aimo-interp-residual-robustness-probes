#!/usr/bin/env bash
# Run from the src/ directory. Set PY to the runtime-matching interpreter (see setup_env.sh).
# e2e/ must contain components/ and scripts/ from aimo-interp/getting-started plus input/ and reference/ (cases + labels).
# End-to-end contract test of the Small-Track bundle with the official ingestion/scoring programs.
set -euo pipefail
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore
PY=${PY:-python}
ART=${1:?artifact dir}
rm -rf bundle && mkdir bundle && cp ../submission/solution.py ../submission/small.txt aimo_features.py bundle/ && cp -r "$ART" bundle/probe_artifacts
(cd bundle && $PY -c "import zipfile,os
z=zipfile.ZipFile('../bundle-small.zip','w',zipfile.ZIP_DEFLATED)
[z.write(os.path.join(r,f),os.path.relpath(os.path.join(r,f),'.')) for r,_,fs in os.walk('.') for f in fs if '__pycache__' not in r]
z.close()")
cd e2e
for run in 1 2; do
  t0=$(date +%s.%N)
  $PY scripts/run_local.py ../bundle-small.zip --small --input-dir input --reference-dir reference 2>&1 | grep -v -E "Loading weights|Warning" | tail -8
  echo "run $run wall_s=$(echo "$(date +%s.%N) - $t0" | bc)"
done
