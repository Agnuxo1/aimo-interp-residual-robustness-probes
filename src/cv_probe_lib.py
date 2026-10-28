"""Shared data loading for probe cross-validation and export."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES_DIR = os.environ.get("AIMO_FEATURES", "features")

SMALL = {
    "Qwen/Qwen3.5-4B": "Qwen_Qwen3.5-4B",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B",
    "allenai/Olmo-3-7B-Think": "allenai_Olmo-3-7B-Think",
    "Skywork/Skywork-OR1-Math-7B": "Skywork_Skywork-OR1-Math-7B",
}

lab = pd.read_csv("data/labels.csv").dropna(subset=["label"])
lab["eval_model"] = lab.model.str.split("|").str[0]
probs = {json.loads(l)["id"]: json.loads(l)["problem"] for l in open("data/problems.jsonl", encoding="utf-8")}

FEATS = {}
for mid, safe in SMALL.items():
    z = np.load(Path(FEATURES_DIR) / f"{safe}.npz")
    FEATS[mid] = {"index": {p: i for i, p in enumerate(z["ids"])}, "last": z["last"].astype(np.float32),
                  "mean": z["mean"].astype(np.float32)}


def rows_for(target, pooled):
    own = lab[lab.eval_model == target]
    if not pooled:
        return own[["problem_id", "label", "source"]].assign(own=1)
    other = lab[lab.eval_model != target]
    # Problem-level proxy: majority label of other labelling models, only where the target has no own label.
    other = other[~other.problem_id.isin(own.problem_id)]
    # Each labelling model votes once per problem (rows are unique per problem x model); ties are dropped.
    agg = other.groupby("problem_id").label.mean().reset_index()
    agg = agg[agg.label != 0.5].assign(label=lambda d: (d.label > 0.5).astype(int))
    return pd.concat([own[["problem_id", "label", "source"]].assign(own=1), agg.assign(own=0, source="proxy")], ignore_index=True)


def featurize(target, kind, pids, folds=None):
    f = FEATS[target]
    idx = [f["index"][p] for p in pids]
    if kind == "prior":  # no features: the classifier can only learn each model's base rate
        return np.zeros((len(pids), 1), dtype=np.float32)
    if kind == "length":
        t = [probs[p] for p in pids]
        return np.array([[len(s), s.count("$"), s.count("\\"), len(s.split())] for s in t], dtype=np.float32)
    pool, layer = kind.split(":")
    X = f[pool][idx]
    L = X.shape[1] - 1
    if layer == "band":  # mean of standardised middle-to-late layers (0.4L..0.8L)
        lo, hi = int(0.4 * L), int(0.8 * L)
        B = X[:, lo:hi + 1]
        B = (B - B.mean(axis=0)) / (B.std(axis=0) + 1e-6)
        return B.mean(axis=1)
    return X[:, int(float(layer) * L)]  # relative depth


