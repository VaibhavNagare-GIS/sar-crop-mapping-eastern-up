"""Phase 3d: why is accuracy limited? TRAIN fields only, field level, 4 classes (no Fallow).

Usage: python p3_diag.py SPEC [--leaf 10] [--seeds 3] [--kind raw] [--data data]
SPEC as in p3_sweep.py (folders joined by '+').
 1 accuracy by field size     small fields vs large fields
 2 learning curve             does more training data still help?
 3 label cleaning             drop training fields the model strongly disagrees with; test labels stay untouched
 4 class grouping             confusion matrix, and merging Mustard + Lentil
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

import features
import p3_sweep as sw

BINS = [(0, 15, "<15 px (<0.15 ha)"), (15, 30, "15-29 px"), (30, 60, "30-59 px"), (60, 10 ** 9, ">=60 px (>=0.6 ha)")]
FRACS = (0.2, 0.4, 0.6, 0.8, 1.0)


def f1m(y, p, classes):
    return f1_score(y, p, labels=classes, average="macro", zero_division=0)


def size_table(y, pred, proba, npix):
    classes = np.unique(y)
    rows = []
    for lo, hi, name in BINS:
        m = (npix >= lo) & (npix < hi)
        if m.sum() < 5:
            continue
        auc = np.nan
        if all((y[m] == c).any() and (y[m] != c).any() for c in classes):
            auc = np.mean([roc_auc_score(y[m] == c, proba[m, i]) for i, c in enumerate(classes)])
        rows.append(dict(size=name, fields=int(m.sum()), macro_F1=f1m(y[m], pred[m], classes),
                         bal_acc=balanced_accuracy_score(y[m], pred[m]), AUC=auc))
    return pd.DataFrame(rows)


def learning_curve(X, y, g, model, leaf, trees, seeds):
    classes, rows = np.unique(y), []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        preds = {f: np.empty(len(y), y.dtype) for f in FRACS}
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
            for f in FRACS:
                sub = rng.choice(tr, int(round(f * len(tr))), replace=False)
                est = features.make_model(model, leaf, trees, seed).fit(X[sub], y[sub])
                preds[f][te] = est.predict(X[te])
        rows += [dict(frac=f, seed=seed, f1=f1m(y, preds[f], classes)) for f in FRACS]
    return pd.DataFrame(rows).groupby("frac").f1.agg(["mean", "std"]).rename(columns={"mean": "macro_F1", "std": "sd"})


def label_cleaning(X, y, g, model, leaf, trees, seeds, taus=(0.0, 0.15, 0.25)):
    """Inside each training part: out-of-fold probability of the field's OWN label; drop fields below tau; retrain; score on the untouched fold."""
    classes, rows = np.unique(y), []
    yi = np.searchsorted(classes, y)
    for seed in range(seeds):
        preds = {t: np.empty(len(y), y.dtype) for t in taus}
        dropped = {t: 0 for t in taus}
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
            inner = cross_val_predict(features.make_model(model, leaf, trees, seed), X[tr], y[tr], groups=g[tr],
                                      cv=StratifiedGroupKFold(3, shuffle=True, random_state=seed), method="predict_proba")
            own = inner[np.arange(len(tr)), yi[tr]]
            for t in taus:
                keep = tr[own >= t]
                dropped[t] += len(tr) - len(keep)
                preds[t][te] = features.make_model(model, leaf, trees, seed).fit(X[keep], y[keep]).predict(X[te])
        rows += [dict(tau=t, seed=seed, f1=f1m(y, preds[t], classes), dropped_share=dropped[t] / (5 * len(y) * 0.8)) for t in taus]
    return pd.DataFrame(rows).groupby("tau")[["f1", "dropped_share"]].mean().rename(columns={"f1": "macro_F1"})


def run(spec, data="data", leaf=10, seeds=3, kind="raw", model="rf", trees=100):
    parts, meta = sw.load(spec)
    keep = (meta.label != sw.FALLOW).to_numpy()
    parts, meta = [p[keep] for p in parts], meta[keep]
    X, y, g = sw.build(parts, meta, "field", kind)
    fields = np.sort(meta.field.unique())
    npix = pd.read_csv(Path(data) / "phase1" / "fields_study.csv").set_index("field").npix.reindex(fields).to_numpy()
    if np.isnan(npix).any():
        raise SystemExit("Some fields are missing from phase1/fields_study.csv")
    classes = np.unique(y)
    proba = np.mean([features.cv_predict(X, y, g, seed=s, trees=trees, leaf=leaf, model=model, proba=True) for s in range(seeds)], axis=0)
    pred = classes[proba.argmax(1)]
    out = {"size": size_table(y, pred, proba, npix), "curve": learning_curve(X, y, g, model, leaf, trees, seeds),
           "clean": label_cleaning(X, y, g, model, leaf, trees, seeds)}
    cm = pd.crosstab(pd.Series(y, name="true").map(lambda c: sw.SHORT[c]), pd.Series(pred, name="pred").map(lambda c: sw.SHORT[c]), normalize="index").round(2)
    ym = np.where(np.isin(y, (2, 3)), 2, y)                                    # merge Mustard + Lentil
    pm = np.mean([features.cv_predict(X, ym, g, seed=s, trees=trees, leaf=leaf, model=model, proba=True) for s in range(seeds)], axis=0)
    cm_cls = np.unique(ym)
    out["confusion"] = cm
    out["merge"] = pd.Series(dict(macro_F1_4class=f1m(y, pred, classes), macro_F1_merged_3class=f1m(ym, cm_cls[pm.argmax(1)], cm_cls),
                                  AUC_4class=np.mean([roc_auc_score(y == c, proba[:, i]) for i, c in enumerate(classes)]),
                                  AUC_merged_3class=np.mean([roc_auc_score(ym == c, pm[:, i]) for i, c in enumerate(cm_cls)])))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--leaf", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--kind", default="raw", choices=features.KINDS)
    ap.add_argument("--model", default="rf", choices=features.MODELS)
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    r = run(a.spec, a.data, a.leaf, a.seeds, a.kind, a.model)
    for title, key in [("1. Accuracy by field size", "size"), ("2. Learning curve (share of training fields used)", "curve"),
                       ("3. Training-only label cleaning (tau = minimum own-label probability; 0 = none)", "clean"),
                       ("4a. Confusion, rows = truth (share)", "confusion"), ("4b. Merging Mustard + Lentil (3 classes vs 4: not directly comparable)", "merge")]:
        print(f"\n{title}\n{r[key].round(3).to_string()}")
