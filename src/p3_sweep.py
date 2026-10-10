"""Phase 3c: feature, model and setting sweeps. TRAIN fields only (the test split is never read).

Usage: python p3_sweep.py [options] SPEC [SPEC ...]
SPEC   = features folder(s) joined by '+': each folder adds its own features, e.g. ASC+DESC or S1+S2
         data/phase2/2021-10-01+data/phase2/2021-10-01_s2
--leaves 1,5,10   forest/tree smoothness values to try        --seeds 5    CV repeats (different folds)
--kind raw|smooth|summary|both   feature form                 --models rf,et,hgb,lr
--level field|pixel|both   field = classify each field from the mean of its pixels
--classes 3,4,5   3 = Wheat, Mustard+Lentil, Sugarcane | 4 = Wheat, Mustard, Lentil, Sugarcane | 5 = 4 + Fallow
--common   score every SPEC on the pixels they all share (needed when one SPEC has fewer chips, e.g. descending)
Reports macro-F1, balanced accuracy, overall accuracy vs always-biggest-class, and one-vs-rest AUC per class.
Appends every row to data/phase3/sweep.csv.
"""
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

import features

FALLOW = 4
SHORT = {1: "Wh", 2: "Mu", 3: "Le", 4: "Fa", 6: "Su"}
META = ["field", "label", "block", "split"]


def load(spec, splits=("train",)):
    """-> (parts, meta). parts = one raw-feature table per folder, all with the SAME pid index; TRAIN fields only unless
    `splits` says otherwise (only p4_final.py ever asks for test)."""
    tabs = [pd.read_csv(Path(p) / "features.csv", dtype={"chip": str}).set_index("pid") for p in spec.split("+")]
    keep = tabs[0].index
    for t in tabs[1:]:
        keep = keep.intersection(t.index)
    meta = tabs[0].loc[keep, META]
    for t in tabs[1:]:
        if not (t.loc[keep, "label"] == meta.label).all():
            raise SystemExit(f"Folders in '{spec}' disagree on labels: they were not made from the same pixel table")
    parts = [t.loc[keep, sum(features.col_groups(t), [])] for t in tabs]
    ok = meta.split.isin(splits).to_numpy().copy()   # pandas 3 returns a read-only view
    for p in parts:                                   # drop pixels where any sensor/orbit has no valid window at all
        for cols in features.col_groups(p):
            ok &= ~p[cols].isna().all(axis=1).to_numpy()
    return [p[ok] for p in parts], meta[ok]


def restrict(parts, meta, idx):
    return [p.loc[idx] for p in parts], meta.loc[idx]


def build(parts, meta, level, kind):
    """-> X, y, groups. field level: features from the per-field MEAN of the raw bins."""
    if level == "pixel":
        return np.hstack([features.make_X(p, kind)[0] for p in parts]), meta.label.to_numpy().astype(int), meta.block.to_numpy()
    g = meta.field.to_numpy()
    X = np.hstack([features.make_X(p.groupby(g).mean(), kind)[0] for p in parts])
    fm = meta.groupby("field").agg(label=("label", "first"), block=("block", "first"))   # same sorted order as groupby above
    return X, fm.label.to_numpy().astype(int), fm.block.to_numpy()


def score(proba, y):
    classes = np.unique(y)
    pred = classes[proba.argmax(1)]
    aucs = {int(c): roc_auc_score(y == c, proba[:, i]) for i, c in enumerate(classes)}
    return dict(f1=f1_score(y, pred, average="macro"), bal=balanced_accuracy_score(y, pred),
                oa=accuracy_score(y, pred), auc=float(np.mean(list(aucs.values()))), aucs=aucs)


def run(specs, leaves=(10,), seeds=5, kind="raw", levels=("field",), classes=("4",), models=("rf",), common=False,
        trees=100, out="data/phase3/sweep.csv"):
    data = {s: load(s) for s in specs}
    if common:
        idx = None
        for parts, meta in data.values():
            idx = meta.index if idx is None else idx.intersection(meta.index)
        data = {s: restrict(*v, idx) for s, v in data.items()}
        print(f"--common: {len(idx)} pixels shared by all specs")
    rows = []
    for spec, (parts, meta) in data.items():
        name = "+".join(Path(p).name for p in spec.split("+"))
        for cs in classes:
            m = (meta.label != FALLOW).to_numpy() if cs in ("3", "4") else np.ones(len(meta), bool)
            sub_parts, sub_meta = [p[m] for p in parts], meta[m].copy()
            if cs == "3":                                              # merge Mustard + Lentil into one class (label 2)
                sub_meta["label"] = sub_meta.label.where(~sub_meta.label.isin((2, 3)), 2)
            for level in levels:
                X, y, g = build(sub_parts, sub_meta, level, kind)
                base = np.bincount(y).max() / len(y)
                for model in models:
                    for leaf in leaves:
                        r = [score(features.cv_predict(X, y, g, seed=s, trees=trees, leaf=leaf, model=model, proba=True), y)
                             for s in range(seeds)]
                        aucs = {c: np.mean([x["aucs"][c] for x in r]) for c in r[0]["aucs"]}
                        row = dict(run=dt.datetime.now().isoformat(timespec="seconds"), spec=name, kind=kind, model=model, leaf=leaf,
                                   classes=cs, level=level, seeds=seeds, n=len(y), n_features=X.shape[1],
                                   f1_mean=np.mean([x["f1"] for x in r]), f1_sd=np.std([x["f1"] for x in r]),
                                   bal_acc=np.mean([x["bal"] for x in r]), oa=np.mean([x["oa"] for x in r]), baseline_oa=base,
                                   auc_macro=np.mean([x["auc"] for x in r]), **{f"auc_{SHORT[c]}": v for c, v in aucs.items()})
                        rows.append(row)
                        print(f"{name:34s} {kind:7s} {model:3s} leaf{leaf:<3d} {cs}cl {level:5s} n={len(y):6d} F1 {row['f1_mean']:.3f}+/-{row['f1_sd']:.3f} "
                              f"bal {row['bal_acc']:.3f} OA {row['oa']:.3f}({base:.3f}) AUC {row['auc_macro']:.3f} "
                              + " ".join(f"{SHORT[c]}{v:.2f}" for c, v in aucs.items()))
    res = pd.DataFrame(rows)
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([pd.read_csv(p), res]).to_csv(p, index=False) if p.exists() else res.to_csv(p, index=False)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--leaves", default="10")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--kind", default="raw", choices=features.KINDS)
    ap.add_argument("--models", default="rf")
    ap.add_argument("--level", default="field", choices=["field", "pixel", "both"])
    ap.add_argument("--classes", default="4", help="comma list of 3 (Wheat, Mustard+Lentil, Sugarcane), 4 (no Fallow), 5 (all); 'both' = 4,5")
    ap.add_argument("--common", action="store_true")
    a = ap.parse_args()
    cl = ["4", "5"] if a.classes == "both" else a.classes.split(",")
    if any(c not in ("3", "4", "5") for c in cl):
        raise SystemExit("--classes must be a comma list of 3, 4, 5 (or 'both')")
    bad = [m for m in a.models.split(",") if m not in features.MODELS]
    if bad:
        raise SystemExit(f"unknown model(s) {bad}; choose from {features.MODELS}")
    run(a.specs, [int(x) for x in a.leaves.split(",")], a.seeds, a.kind, ["field", "pixel"] if a.level == "both" else [a.level],
        cl, a.models.split(","), a.common)
