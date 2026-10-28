"""Fit the selected probe ensemble on all labelled rows and export plain-numpy artifacts.

One .npz per evaluated model; sub-probe i is stored with key prefix "p{i}_".
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import cv_probe_lib as lib

ap = argparse.ArgumentParser()
ap.add_argument("--kinds", required=True)          # e.g. last:band+mean:0.75+mean:1
ap.add_argument("--C", type=float, default=0.1)
ap.add_argument("--pca", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="artifacts")
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
summary = {}
for target, safe in lib.SMALL.items():
    R = lib.rows_for(target, True)
    f = lib.FEATS[target]
    y = R.label.to_numpy().astype(int)
    arrays = dict(n_probes=np.array(len(args.kinds.split("+"))), threshold=np.array(0.5))
    for i, kind in enumerate(args.kinds.split("+")):
        pool, layer = kind.split(":")
        X = f[pool][[f["index"][p] for p in R.problem_id]]
        L = X.shape[1] - 1
        pre = f"p{i}_"
        if layer == "band":
            lo, hi = int(0.4 * L), int(0.8 * L)
            mu, sd = X[:, lo:hi + 1].mean(axis=0), X[:, lo:hi + 1].std(axis=0) + 1e-6
            Z = ((X[:, lo:hi + 1] - mu) / sd).mean(axis=1)
            arrays[pre + "band_mu"], arrays[pre + "band_sd"] = mu.astype(np.float32), sd.astype(np.float32)
        else:
            Z = X[:, int(float(layer) * L)]
        sc = StandardScaler().fit(Z)
        Zs = sc.transform(Z)
        pca = PCA(n_components=args.pca, random_state=args.seed).fit(Zs)
        lr = LogisticRegression(C=args.C, class_weight="balanced", max_iter=2000, random_state=args.seed)
        lr.fit(pca.transform(Zs), y)
        arrays.update({pre + "pool": np.array(pool), pre + "layer": np.array(layer),
                       pre + "scaler_mean": sc.mean_.astype(np.float32), pre + "scaler_scale": sc.scale_.astype(np.float32),
                       pre + "pca_mean": pca.mean_.astype(np.float32), pre + "pca_components": pca.components_.astype(np.float32),
                       pre + "coef": lr.coef_[0].astype(np.float32), pre + "intercept": np.array(lr.intercept_[0], dtype=np.float32)})
    np.savez(out / f"{safe}.npz", **arrays)
    # Reference scores on the first five training problems, used to check the bundle reproduces them.
    summary[target] = dict(n=int(len(y)), robust=int(y.sum()), own=int(R.own.sum()))
print(json.dumps(dict(config=vars(args), models=summary), indent=1))
