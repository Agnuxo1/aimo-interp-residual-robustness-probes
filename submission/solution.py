"""AIMO-Interp Small Track: residual-stream robustness probe ensemble.

For each problem we run one forward pass of the evaluated model, read its
residual stream (last prompt token and token-mean; see aimo_features.py) and
average the probabilities of a few linear probes exported as plain numpy
arrays (standardise -> PCA -> logistic regression).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from aimo_features import encode, load_model, safe_model_id

ARTIFACTS = Path(__file__).resolve().parent / "probe_artifacts"


def _probe_prob(art, pre: str, last: np.ndarray, mean: np.ndarray) -> float:
    X = {"last": last, "mean": mean}[str(art[pre + "pool"])]
    L = X.shape[0] - 1
    layer = str(art[pre + "layer"])
    if layer == "band":
        lo, hi = int(0.4 * L), int(0.8 * L)
        x = ((X[lo:hi + 1] - art[pre + "band_mu"]) / art[pre + "band_sd"]).mean(axis=0)
    else:
        x = X[int(float(layer) * L)]
    z = (x - art[pre + "scaler_mean"]) / art[pre + "scaler_scale"]
    z = (z - art[pre + "pca_mean"]) @ art[pre + "pca_components"].T
    logit = float(z @ art[pre + "coef"] + art[pre + "intercept"])
    return 1.0 / (1.0 + np.exp(-logit))


def robust_probability(art, last: np.ndarray, mean: np.ndarray) -> float:
    n = int(art["n_probes"])
    return float(np.mean([_probe_prob(art, f"p{i}_", last, mean) for i in range(n)]))


def are_robust(model_id: str, reasoning_effort: str = "default", problems: list[str] | None = None) -> list[bool]:
    problems = problems or []
    path = ARTIFACTS / f"{safe_model_id(model_id)}.npz"
    if not path.is_file():
        return [True for _ in problems]
    art = np.load(path, allow_pickle=False)
    threshold = float(art["threshold"])
    tokenizer, model = load_model(model_id)
    out = []
    for p in problems:
        try:
            last, mean = encode(tokenizer, model, p)
            out.append(bool(robust_probability(art, last, mean) >= threshold))
        except Exception:
            out.append(True)
    del model
    torch.cuda.empty_cache()
    return out
