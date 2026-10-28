#!/usr/bin/env bash
# Build a venv that mirrors the Codabench runtime (Dockerfile.competition) and
# cache the four Small-Track checkpoints. Idempotent; logs to setup.log.
# Usage: AIMO_HOME=/path/to/workdir bash setup_env.sh   (venv in $AIMO_HOME/.venv, HF cache in $AIMO_HOME/hf)
set -euo pipefail
P=${AIMO_HOME:-$HOME/aimo-interp}
mkdir -p "$P"
export HF_HOME="$P/hf"
if [ ! -x "$P/.venv/bin/python" ]; then
  python3 -m venv "$P/.venv"
  "$P/.venv/bin/pip" install -q --upgrade pip
  "$P/.venv/bin/pip" install -q "torch==2.12.1" --index-url https://download.pytorch.org/whl/cu132
  "$P/.venv/bin/pip" install -q "accelerate==1.13.0" "huggingface-hub==1.22.0" "joblib==1.5.3" \
      "pandas==3.0.3" "safetensors==0.8.0" "scikit-learn==1.8.0" "tokenizers==0.22.2" \
      "transformers==5.13.0" pyarrow
fi
"$P/.venv/bin/python" -c "import torch,transformers,sklearn;print(torch.__version__,transformers.__version__,sklearn.__version__,torch.version.cuda)"
"$P/.venv/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen3.5-4B", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "allenai/Olmo-3-7B-Think", "Skywork/Skywork-OR1-Math-7B"]:
    p = snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.py", "tokenizer*"])
    print("cached", m, p, flush=True)
PY
echo SETUP_DONE
