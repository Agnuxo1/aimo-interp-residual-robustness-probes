"""E6: causal directional ablation inside the forward pass.

Question: is the robustness information that late-layer probes read out routed
through the direction a mid-depth probe finds? For each grouped fold we
  1. fit, on training problems only, a token-mean probe at mid depth (l = L/2)
     and map it back to a residual-stream direction (rank 1) or to its full
     16-dim PCA subspace (rank 16);
  2. fit downstream probes (token-mean at L, last token at L) on unablated
     training features;
  3. re-run the model on the held-out problems while projecting that
     direction/subspace out of the residual stream at layer l at every token,
     and score the downstream probes on the ablated activations.
Controls: no ablation (must reproduce the stored features) and random
directions/subspaces of the same rank (3 seeds each).
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import cv_probe_lib as lib
from aimo_features import build_prompt, load_model

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=list(lib.SMALL))
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--C", type=float, default=0.1)
ap.add_argument("--out", default="experiments/ablation_oof.csv")
args = ap.parse_args()
torch.manual_seed(args.seed)


def fit_probe(Z, y, seed):
    sc = StandardScaler().fit(Z)
    pca = PCA(n_components=min(16, len(y) - 1), random_state=seed).fit(sc.transform(Z))
    lr = LogisticRegression(C=args.C, class_weight="balanced", max_iter=2000, random_state=seed)
    lr.fit(pca.transform(sc.transform(Z)), y)
    return sc, pca, lr


def prob(probe, Z):
    sc, pca, lr = probe
    return lr.predict_proba(pca.transform(sc.transform(Z)))[:, 1]


def orthonormal(M):
    q, _ = np.linalg.qr(M.T)
    return q.T  # rows orthonormal


def decoder_layers(model):
    n = model.config.get_text_config().num_hidden_layers
    best = None
    for m in model.modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) == n:
            best = m
            break
    assert best is not None, "decoder layer list not found"
    return best


@torch.inference_mode()
def run_ablated(tok, model, layers, problem, layer_idx, basis):
    """Forward pass with the span of `basis` (k x d, orthonormal rows) removed from hidden_states[layer_idx]."""
    handle = None
    if basis is not None:
        B = torch.tensor(basis, dtype=torch.float32, device=model.device)

        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hf = h.float()
            hf = hf - (hf @ B.T) @ B
            h2 = hf.to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        handle = layers[layer_idx - 1].register_forward_hook(hook)  # hidden_states[i] = output of layer i-1
    try:
        ids = tok(build_prompt(tok, problem), return_tensors="pt", add_special_tokens=False).to(model.device)
        hs = model(**ids, output_hidden_states=True, use_cache=False).hidden_states
        return hs[-1][0, -1].float().cpu().numpy(), hs[-1][0].float().mean(0).cpu().numpy()
    finally:
        if handle is not None:
            handle.remove()


recs = []
for target in args.models:
    t0 = time.time()
    R = lib.rows_for(target, True).reset_index(drop=True)
    y = R.label.to_numpy().astype(int)
    g = R.problem_id.to_numpy()
    f = lib.FEATS[target]
    idx = np.array([f["index"][p] for p in g])
    L = f["mean"].shape[1] - 1
    lm = L // 2
    Zmid, Zmean_L, Zlast_L = f["mean"][idx, lm], f["mean"][idx, L], f["last"][idx, L]
    rng = np.random.default_rng(args.seed)
    uniq = rng.permutation(np.unique(g))
    fold = np.array([{u: i % args.folds for i, u in enumerate(uniq)}[x] for x in g])
    tok, model = load_model(target)
    layers = decoder_layers(model)
    d = Zmid.shape[1]
    for k in range(args.folds):
        te = np.where((fold == k) & (R.own.to_numpy() == 1))[0]
        tr = fold != k
        if len(te) == 0:
            continue
        sc, pca, lr = fit_probe(Zmid[tr], y[tr], args.seed)
        w = (pca.components_.T @ lr.coef_[0]) / sc.scale_          # gradient of the logit in raw residual space
        sub = pca.components_ / sc.scale_                            # PCA subspace mapped back to raw space
        bases = {"none": None, "probe_r1": orthonormal(w[None]), "probe_r16": orthonormal(sub)}
        for s in range(3):
            r = np.random.default_rng(100 + s)
            bases[f"random_r1_s{s}"] = orthonormal(r.standard_normal((1, d)))
            bases[f"random_r16_s{s}"] = orthonormal(r.standard_normal((sub.shape[0], d)))
        down = {"mean_L": fit_probe(Zmean_L[tr], y[tr], args.seed), "last_L": fit_probe(Zlast_L[tr], y[tr], args.seed)}
        problems = dict(json.loads(l).values() for l in open("data/problems.jsonl", encoding="utf-8"))
        for cond, basis in bases.items():
            for i in te:
                last_L, mean_L = run_ablated(tok, model, layers, problems[g[i]], lm, basis)
                recs.append(dict(model=target, problem_id=g[i], source=R.source[i], y=int(y[i]), fold=k, cond=cond,
                                 p_mean_L=float(prob(down["mean_L"], mean_L[None])[0]),
                                 p_last_L=float(prob(down["last_L"], last_L[None])[0]),
                                 drift=float(np.abs(mean_L - Zmean_L[i]).max()) if cond == "none" else np.nan))
    print(json.dumps({"model": target, "layer": lm, "L": L, "seconds": round(time.time() - t0, 1)}), flush=True)
    del model, layers, tok
    gc.collect()
    torch.cuda.empty_cache()

df = pd.DataFrame(recs)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(args.out, index=False)
summary = []
for cond, gdf in df.groupby("cond"):
    row = {"cond": cond}
    for col in ["p_mean_L", "p_last_L"]:
        row[col + "_auc"] = roc_auc_score(gdf.y, gdf[col])
        row[col + "_bacc"] = balanced_accuracy_score(gdf.y, gdf[col] >= 0.5)
    summary.append(row)
s = pd.DataFrame(summary)
s.to_csv(Path(args.out).with_name("ablation_summary.csv"), index=False)
print(s.round(3).to_string())
print("max |stored - recomputed| (no ablation):", df.drift.max())
