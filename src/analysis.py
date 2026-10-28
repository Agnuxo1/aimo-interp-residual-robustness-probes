"""Ensembles, layer sweep, permutation control, paired bootstrap and nested selection.

Protocol: per evaluated model, K folds grouped by problem; metrics on the
model's own labels (or on own + problem-level proxy labels with --eval-all).
Out-of-fold predictions are kept so paired comparisons use identical rows.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cv_probe_lib import SMALL, featurize, probs, rows_for

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, choices=["ensembles", "sweep", "perm", "bootstrap", "weights", "nested"])
ap.add_argument("--kind", default="last:band+mean:0.75+mean:1")
ap.add_argument("--C", type=float, default=0.1)
ap.add_argument("--perms", type=int, default=200)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--eval-all", action="store_true", help="evaluate on own AND proxy (problem-level) labels")
ap.add_argument("--own-weight", type=float, default=1.0)
args = ap.parse_args()
OUT = Path("experiments")
OUT.mkdir(exist_ok=True)

CANDIDATES = [["last:band"], ["mean:1"], ["last:0.5"], ["last:band", "mean:1"], ["last:band", "mean:1", "last:0.5"],
              ["last:band", "mean:0.75", "mean:1"], ["last:band", "length"], ["last:band", "mean:1", "length"]]
CONTROLS = [["prior"], ["length"], ["tfidf"], ["mean:0"]]


def clf_for(kind, C, seed, n_train):
    if kind == "prior":
        return make_pipeline(DummyClassifier(strategy="prior"))
    if kind == "tfidf":
        return make_pipeline(TfidfVectorizer(min_df=2, ngram_range=(1, 2)),
                             LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed))
    steps = [StandardScaler()]
    if kind != "length":
        steps.append(PCA(n_components=min(16, n_train - 1), random_state=seed))
    steps.append(LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed))
    return make_pipeline(*steps)


def fit_predict(kinds, C, seed, target, R, y, tr, te):
    """Mean probability over the probes in `kinds`, trained on rows tr, scored on rows te."""
    g = R.problem_id.to_numpy()
    w = np.where(R.own.to_numpy()[tr] == 1, args.own_weight, 1.0)
    ps = []
    for k in kinds:
        X = [probs[p] for p in g] if k == "tfidf" else featurize(target, k, g)
        pick = (lambda m: [X[i] for i in np.where(m)[0]]) if k == "tfidf" else (lambda m: X[m])
        c = clf_for(k, C, seed, int(tr.sum()))
        if k == "prior":
            c.fit(pick(tr), y[tr])
        else:
            c.fit(pick(tr), y[tr], logisticregression__sample_weight=w)
        ps.append(c.predict_proba(pick(te))[:, 1])
    return np.mean(ps, axis=0)


def grouped_folds(g, folds, rng):
    uniq = rng.permutation(np.unique(g))
    gmap = {u: i % folds for i, u in enumerate(uniq)}
    return np.array([gmap[x] for x in g])


def oof(kinds, C, seed, shuffle=False):
    rng = np.random.default_rng(seed)
    recs = []
    for target in SMALL:
        R = rows_for(target, True).reset_index(drop=True)
        y = R.label.to_numpy().astype(int)
        if shuffle:
            y = rng.permutation(y)
        g = R.problem_id.to_numpy()
        fold = grouped_folds(g, args.folds, rng)
        evalmask = (R.own.to_numpy() == 1) | args.eval_all
        for f in range(args.folds):
            te = (fold == f) & evalmask
            tr = fold != f
            if te.sum() == 0 or len(set(y[tr])) < 2:
                continue
            p = fit_predict(kinds, C, seed, target, R, y, tr, te)
            for i, pi in zip(np.where(te)[0], p):
                recs.append(dict(model=target, problem_id=g[i], source=R.source[i], y=y[i], p=pi, seed=seed))
    return pd.DataFrame(recs)


def metrics(d):
    yp = (d.p >= 0.5).astype(int)
    o = (d.source == "train-main-v2").to_numpy()
    if o.sum() == 0 or d.y[o].nunique() < 2:
        o = np.ones(len(d), bool)
    within = [roc_auc_score(m.y, m.p) for _, m in d.groupby("model") if m.y.nunique() == 2]
    return dict(acc=(yp == d.y).mean(), bacc=balanced_accuracy_score(d.y, yp), auc=roc_auc_score(d.y, d.p),
                auc_within=float(np.mean(within)),
                oly_bacc=balanced_accuracy_score(d.y[o], yp[o]), oly_auc=roc_auc_score(d.y[o], d.p[o]))


def summarize(kinds, C):
    ms = [metrics(oof(kinds, C, s)) for s in range(args.seeds)]
    return {**{k: float(np.mean([m[k] for m in ms])) for k in ms[0]},
            **{k + "_sd": float(np.std([m[k] for m in ms])) for k in ms[0]}}


def show(r):
    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}), flush=True)


if args.task == "ensembles":
    rows = []
    for kinds in CANDIDATES + CONTROLS:
        for C in ([0.01, 0.1] if kinds[0] != "prior" else [0.1]):
            r = dict(kinds="+".join(kinds), C=C, **summarize(kinds, C))
            rows.append(r)
            show(r)
    pd.DataFrame(rows).to_csv(OUT / "ensembles.csv", index=False)

elif args.task == "weights":
    rows = []
    for w in [1.0, 2.0, 4.0]:
        args.own_weight = w
        r = dict(own_weight=w, kinds=args.kind, C=args.C, **summarize(args.kind.split("+"), args.C))
        rows.append(r)
        show(r)
    pd.DataFrame(rows).to_csv(OUT / "own_weight.csv", index=False)

elif args.task == "sweep":
    rows = []
    for pool in ["last", "mean"]:
        for depth in np.round(np.linspace(0, 1, 21), 2):
            r = dict(pool=pool, depth=float(depth), **summarize([f"{pool}:{depth}"], args.C))
            rows.append(r)
            show(r)
    pd.DataFrame(rows).to_csv(OUT / "layer_sweep.csv", index=False)

elif args.task == "perm":
    kinds = args.kind.split("+")
    obs = summarize(kinds, args.C)
    nd = pd.DataFrame([metrics(oof(kinds, args.C, 1000 + i, shuffle=True)) for i in range(args.perms)])
    nd.to_csv(OUT / "permutation_null.csv", index=False)
    p = {k: float((1 + (nd[k] >= obs[k]).sum()) / (1 + len(nd))) for k in ["auc", "auc_within", "bacc", "oly_auc"]}
    json.dump(dict(kinds=args.kind, C=args.C, **obs, p_values=p, perms=args.perms),
              open(OUT / "permutation_observed.json", "w"), indent=1)
    print(json.dumps(dict(obs=obs, p=p)))

elif args.task == "bootstrap":
    # Paired bootstrap over problems: selected probe vs each control, seed-averaged OOF probabilities.
    def avg_oof(kinds):
        d = pd.concat([oof(kinds, args.C, s) for s in range(args.seeds)])
        return d.groupby(["model", "problem_id", "source", "y"], as_index=False).p.mean()

    def within_auc(y, p, m):
        v = [roc_auc_score(y[m == k], p[m == k]) for k in np.unique(m) if len(set(y[m == k])) == 2]
        return float(np.mean(v))

    sel = avg_oof(args.kind.split("+"))
    out = {}
    rng = np.random.default_rng(0)
    for ctrl in CONTROLS:
        c = avg_oof(ctrl)
        m = sel.merge(c, on=["model", "problem_id", "source", "y"], suffixes=("_s", "_c"))
        pids = m.problem_id.unique()
        rows_of = {q: np.where(m.problem_id.to_numpy() == q)[0] for q in pids}
        Y, PS, PC, M = m.y.to_numpy(), m.p_s.to_numpy(), m.p_c.to_numpy(), m.model.to_numpy()
        stats = {"auc": lambda y, s, mm: roc_auc_score(y, s), "auc_within": within_auc,
                 "bacc": lambda y, s, mm: balanced_accuracy_score(y, s >= .5)}
        diffs = {k: [] for k in stats}
        for _ in range(2000):
            ix = np.concatenate([rows_of[q] for q in rng.choice(pids, size=len(pids), replace=True)])
            if len(set(Y[ix])) < 2:
                continue
            for k, fn in stats.items():
                try:
                    diffs[k].append(fn(Y[ix], PS[ix], M[ix]) - fn(Y[ix], PC[ix], M[ix]))
                except (ValueError, ZeroDivisionError):
                    pass
        point = {k: fn(Y, PS, M) - fn(Y, PC, M) for k, fn in stats.items()}
        out["+".join(ctrl)] = {k: dict(delta=float(point[k]), ci95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                                       p_le_0=float(np.mean(np.array(v) <= 0))) for k, v in diffs.items()}
    tag = "_all" if args.eval_all else ""
    sel.to_csv(OUT / f"oof_selected{tag}.csv", index=False)
    json.dump(dict(selected=args.kind, C=args.C, eval_all=args.eval_all, n_rows=int(len(sel)),
                   n_problems=int(sel.problem_id.nunique()), vs=out), open(OUT / f"bootstrap_ci{tag}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))

elif args.task == "nested":
    # Honest estimate of the whole selection procedure: the ensemble is re-selected inside every outer
    # training split by an inner grouped CV (by balanced accuracy), then scored on the untouched outer fold.
    recs, picks = [], []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        for target in SMALL:
            R = rows_for(target, True).reset_index(drop=True)
            y = R.label.to_numpy().astype(int)
            g = R.problem_id.to_numpy()
            own = R.own.to_numpy() == 1
            fold = grouped_folds(g, args.folds, rng)
            for f in range(args.folds):
                te, tr = (fold == f) & own, fold != f
                if te.sum() == 0:
                    continue
                inner = grouped_folds(g, 4, np.random.default_rng(seed * 100 + f))
                best, best_score = None, -1
                for kinds in CANDIDATES:
                    yt, pp = [], []
                    for k in range(4):
                        ite, itr = tr & (inner == k) & own, tr & (inner != k)
                        if ite.sum() == 0 or len(set(y[itr])) < 2:
                            continue
                        pp += list(fit_predict(kinds, args.C, seed, target, R, y, itr, ite)); yt += list(y[ite])
                    score = balanced_accuracy_score(yt, np.array(pp) >= .5) if len(set(yt)) == 2 else 0.5
                    if score > best_score:
                        best, best_score = kinds, score
                picks.append("+".join(best))
                p = fit_predict(best, args.C, seed, target, R, y, tr, te)
                for i, pi in zip(np.where(te)[0], p):
                    recs.append(dict(model=target, problem_id=g[i], source=R.source[i], y=y[i], p=pi, seed=seed))
    d = pd.DataFrame(recs)
    ms = [metrics(d[d.seed == s]) for s in range(args.seeds)]
    res = {**{k: float(np.mean([m[k] for m in ms])) for k in ms[0]}, **{k + "_sd": float(np.std([m[k] for m in ms])) for k in ms[0]},
           "picks": pd.Series(picks).value_counts().to_dict()}
    json.dump(res, open(OUT / "nested.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
