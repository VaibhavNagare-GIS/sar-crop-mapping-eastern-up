"""Phase 2c: look at the data before modelling more.

Usage: python p2_diagnose.py <features folder> [<folder> ...]     (needs matplotlib)
Writes <folder>/signatures.png. Prints: class medians over time, confusion matrix (pixel level, spatial-block CV),
and pixel-level vs field-mean macro-F1 (does averaging pixels inside a field help?). Train fields only.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import features
import study


def diagnose(folder):
    df = pd.read_csv(Path(folder) / "features.csv", dtype={"chip": str})
    if not (df.split == "train").all():
        raise SystemExit(f"{folder} contains test fields; diagnostics must use train fields only.")
    df, dead = features.drop_empty(df)
    ser = features.series(df)
    X = np.hstack(list(ser.values()))
    start = Path(folder).name[:10]
    x = [b[0][5:] for b in study.bins(start)]
    crops = sorted(df.crop.unique())
    fig, axs = plt.subplots(1, len(ser), figsize=(5.5 * len(ser), 4), squeeze=False)
    print(f"\n=== {Path(folder).name}: {len(df)} px, {df.field.nunique()} fields, {dead} dead px")
    for ax, (name, arr) in zip(axs[0], ser.items()):
        rows = []
        for c in crops:
            a = arr[(df.crop == c).to_numpy()]
            lo, med, hi = np.percentile(a, [25, 50, 75], axis=0)
            ax.plot(x, med, label=c)
            ax.fill_between(x, lo, hi, alpha=.12)
            rows.append(pd.Series(med[::3].round(1), index=x[::3], name=c))
        ax.set_title(f"{name} median and IQR"); ax.tick_params(axis="x", rotation=60); ax.legend(fontsize=7)
        print(f"{name} class medians at every 3rd window:\n{pd.DataFrame(rows).to_string()}")
    fig.tight_layout(); fig.savefig(Path(folder) / "signatures.png", dpi=110); plt.close(fig)
    pred = features.cv_predict(X, df.label, df.block)
    labels = sorted(df.label.unique())
    print("pixel macro-F1:", round(f1_score(df.label, pred, average="macro"), 3),
          " per-class F1:", dict(zip([study.CLASSES[l] for l in labels], f1_score(df.label, pred, average=None, labels=labels).round(2))))
    cm = pd.crosstab(df.label.map(study.CLASSES), pd.Series(pred, index=df.index).map(study.CLASSES), normalize="index").round(2)
    print("confusion (rows = truth, share of pixels):\n", cm.to_string())
    fm = pd.DataFrame(X).groupby(df.field.to_numpy()).mean()
    meta = df.groupby("field").agg(label=("label", "first"), block=("block", "first")).loc[fm.index]
    fp = features.cv_predict(fm.to_numpy(), meta.label, meta.block)
    print("field-mean features, field macro-F1:", round(f1_score(meta.label, fp, average="macro"), 3),
          f"({len(meta)} fields)")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        diagnose(d)
