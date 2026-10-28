"""Extract residual-stream features for every public problem and each Small-Track model."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from aimo_features import encode, load_model, safe_model_id

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--problems", default="data/problems.jsonl")
ap.add_argument("--out", default="features")
args = ap.parse_args()

torch.manual_seed(0)
problems = [json.loads(line) for line in open(args.problems, encoding="utf-8")]
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for model_id in args.models:
    t0 = time.time()
    tok, model = load_model(model_id)
    load_s = time.time() - t0
    last, mean, ntok = [], [], []
    t1 = time.time()
    for p in problems:
        a, b = encode(tok, model, p["problem"])
        last.append(a.astype(np.float16))
        mean.append(b.astype(np.float16))
    fwd_s = time.time() - t1
    np.savez_compressed(out / f"{safe_model_id(model_id)}.npz", ids=np.array([p["id"] for p in problems]),
                        last=np.stack(last), mean=np.stack(mean))
    print(json.dumps({"model": model_id, "n": len(problems), "shape": list(np.stack(last).shape),
                      "load_s": round(load_s, 1), "forward_s": round(fwd_s, 1),
                      "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}), flush=True)
    del model
    torch.cuda.empty_cache()
