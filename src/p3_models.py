"""Phase 3a: which features work? Random forest, spatial-block cross-validation, TRAIN fields only.

Usage: python p3_models.py [--seeds 1] SPEC [SPEC ...]
SPEC = a features folder, or two folders joined by '+' to fuse Sentinel-1 and NDVI, e.g.
       data/phase2/2021-10-01+data/phase2/2021-10-01_s2
For every SPEC it scores two class sets (all 5 classes; without Fallow) at two levels:
  pixel : every pixel is classified from its own features
  field : each field is classified from the MEAN of its pixels' features
Prints macro-F1 (mean and sd over seeds), balanced accuracy, overall accuracy and the always-guess-the-biggest-class baseline; saves data/phase3/ablation.csv.
The test split is never read.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import features
import study

FALLOW = 4


def load(spec):
    parts = [pd.read_csv(Path(p) / "features.csv", dtype={"chip": str}) for p in spec.split("+")]
    df = parts[0]
    for q in parts[1:]:
        cols = [c for c in q.columns if c.startswith(("VV_", "VH_", "NDVI_"))]
        if set(cols) & set(df.columns):
            raise SystemExit("Cannot fuse two tables of the same sensor")
        df = df.merge(q[["pid"] + cols], on="pid", how="inner", validate="one_to_one")
    df = df[df.split == "train"]
    df, dead = features.drop_empty(df)
    return df, dead


def evaluate(df, level, seed, trees, leaf=1, X=None):
    """X = optional feature matrix aligned with df rows (default: Sentinel-1/NDVI features of df)."""
    if X is None:
        X, _ = features.make_X(df)
    if level == "pixel":
        y, g = df.label.to_numpy(), df.block.to_numpy()
    else:
        fm = pd.DataFrame(X).groupby(df.field.to_numpy()).mean()
        meta = df.groupby("field").agg(label=("label", "first"), block=("block", "first")).loc[fm.index]
        X, y, g = fm.to_numpy(), meta.label.to_numpy(), meta.block.to_numpy()
    pred = features.cv_predict(X, y, g, seed=seed, trees=trees, leaf=leaf)
    base = np.bincount(y).max() / len(y)                      # accuracy of always guessing the biggest class
    return dict(f1=f1_score(y, pred, average="macro"), oa=accuracy_score(y, pred),
                bal=balanced_accuracy_score(y, pred), base=base, n=len(y))


def run(specs, seeds=1, trees=100, leaf=1, out_dir="data/phase3"):
    rows = []
    for spec in specs:
        df, dead = load(spec)
        name = "+".join(Path(p).name for p in spec.split("+"))
        for cs, d in [("5 classes", df), ("no fallow", df[df.label != FALLOW].reset_index(drop=True))]:
            for level in ["pixel", "field"]:
                r = [evaluate(d, level, s, trees, leaf) for s in range(seeds)]
                row = dict(features=name, classes=cs, level=level, n=r[0]["n"], f1_mean=np.mean([x["f1"] for x in r]),
                           f1_sd=np.std([x["f1"] for x in r]), oa=np.mean([x["oa"] for x in r]),
                           bal_acc=np.mean([x["bal"] for x in r]), baseline_oa=r[0]["base"], leaf=leaf, dead_px=dead)
                rows.append(row)
                print(f"{name:30s} {cs:10s} {level:5s} n={row['n']:6d} macro-F1 {row['f1_mean']:.3f}+/-{row['f1_sd']:.3f}  "
                      f"bal-acc {row['bal_acc']:.3f}  OA {row['oa']:.3f} (always-biggest-class {row['baseline_oa']:.3f})")
    res = pd.DataFrame(rows)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    res.to_csv(Path(out_dir) / "ablation.csv", index=False)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--leaf", type=int, default=1, help="random forest min_samples_leaf (larger = smoother, safer on noisy labels)")
    a = ap.parse_args()
    run(a.specs, a.seeds, leaf=a.leaf)
