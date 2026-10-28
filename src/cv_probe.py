"""Leave-problem-out evaluation of residual-stream robustness probes and controls.

Evaluation population: every labelled row whose labelling model is one of the
four Small-Track checkpoints (own-model labels). Training rows may also come
from other labelling models on the same problem ("pooled" proxy labels).
All splits are grouped by problem, so no problem appears in train and test.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--perms", type=int, default=0)
ap.add_argument("--out", default="experiments/cv_results.csv")
args = ap.parse_args()

from cv_probe_lib import SMALL, featurize, probs, rows_for


def make_clf(kind, C, seed, n_train=10**6):
    if kind == "tfidf":
        return make_pipeline(TfidfVectorizer(min_df=2, ngram_range=(1, 2)),
                             LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed))
    steps = [StandardScaler()]
    if kind != "length":
        steps.append(PCA(n_components=min(16, n_train - 1), random_state=seed))
    steps.append(LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed))
    return make_pipeline(*steps)


def run(kind, pooled, C, seed, shuffle=False):
    rng = np.random.default_rng(seed)
    y_true, y_pred, y_score, models, olymp = [], [], [], [], []
    for target in SMALL:
        R = rows_for(target, pooled).reset_index(drop=True)
        y = R.label.to_numpy().astype(int)
        if shuffle:
            y = rng.permutation(y)
        groups = R.problem_id.to_numpy()
        X = [probs[p] for p in groups] if kind == "tfidf" else featurize(target, kind, groups)
        uniq = rng.permutation(np.unique(groups))
        gmap = {g: i % args.folds for i, g in enumerate(uniq)}
        fold = np.array([gmap[g] for g in groups])
        for k in range(args.folds):
            te = (fold == k) & (R.own.to_numpy() == 1)
            tr = fold != k
            if te.sum() == 0 or len(set(y[tr])) < 2:
                continue
            clf = make_clf(kind, C, seed, int(tr.sum()))
            Xtr = [X[i] for i in np.where(tr)[0]] if kind == "tfidf" else X[tr]
            Xte = [X[i] for i in np.where(te)[0]] if kind == "tfidf" else X[te]
            clf.fit(Xtr, y[tr])
            s = clf.predict_proba(Xte)[:, 1]
            y_true += list(y[te]); y_score += list(s); y_pred += list((s >= 0.5).astype(int))
            models += [target] * int(te.sum())
            olymp += list(R.source.to_numpy()[te] == "train-main-v2")
    yt, yp, ys, ol = map(np.array, (y_true, y_pred, y_score, olymp))
    return dict(acc=(yt == yp).mean(), bacc=balanced_accuracy_score(yt, yp),
                auc=roc_auc_score(yt, ys) if len(set(yt)) > 1 else np.nan, n=len(yt),
                oly_bacc=balanced_accuracy_score(yt[ol], yp[ol]), oly_auc=roc_auc_score(yt[ol], ys[ol]))


configs = [("length", False), ("length", True), ("tfidf", True)]
for pool in ["last", "mean"]:
    for layer in ["0", "0.25", "0.5", "0.75", "1", "band"]:
        configs += [(f"{pool}:{layer}", False), (f"{pool}:{layer}", True)]

out = []
for kind, pooled in configs:
    for C in [0.01, 0.1]:
        res = [run(kind, pooled, C, s) for s in range(args.seeds)]
        r = {k: np.mean([x[k] for x in res]) for k in ["acc", "bacc", "auc", "oly_bacc", "oly_auc"]}
        sd = {k + "_sd": np.std([x[k] for x in res]) for k in ["acc", "bacc", "auc"]}
        row = dict(features=kind, pooled=pooled, C=C, n=res[0]["n"], **r, **sd)
        out.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
df = pd.DataFrame(out).sort_values("bacc", ascending=False)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(args.out, index=False)
print(df.head(12).to_string())
