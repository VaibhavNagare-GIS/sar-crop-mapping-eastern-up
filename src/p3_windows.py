"""Phase 3g: which time windows carry the signal? Label-season control and earliest-reliable-date analysis. TRAIN fields only.
Usage: python p3_windows.py SPEC [--ranges 0:20,0:10,10:20,...] [--classes 3,4] [--seeds 5] [--leaf 10]
A range a:b keeps windows a..b-1; every other window is set to missing and gap-filled with the nearest kept value (no future
information leaks into an early cut-off). RAW features only (the multi-temporal filter looks at the whole season, so it must not be used here).
Window i of a folder named <start> covers <start> + 12*i days. Appends to data/phase3/windows.csv."""
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

import features
import p3_feat as pf
import p3_sweep as sw
import study

DEFAULT = "0:20," + ",".join(f"0:{k}" for k in (4, 6, 8, 10, 12, 14, 16, 18)) + ",10:20,0:10"


def mask_windows(df, a, b):
    out = df.copy()
    for cols in features.col_groups(out):
        for c in cols:
            if not a <= int(c.rsplit("_", 1)[1]) < b:
                out[c] = np.nan
    return out


def dates(start, a, b):
    s = dt.date.fromisoformat(start)
    return f"{s + dt.timedelta(days=study.BIN_DAYS * a)} to {s + dt.timedelta(days=study.BIN_DAYS * b - 1)}"


def run(parts, meta, ranges, classes=("3", "4"), seeds=5, leaf=10, trees=100, spec="", start=None, out=None):
    g = meta.field.to_numpy()
    fm = meta.groupby("field").agg(label=("label", "first"), block=("block", "first"))
    y, grp = fm.label.to_numpy().astype(int), fm.block.to_numpy()
    means = [p.groupby(g).mean() for p in parts]
    rows = []
    for a, b in ranges:
        X = np.hstack([features.make_X(mask_windows(m, a, b), "raw")[0] for m in means])
        for cs in classes:
            r = pf.evaluate(X, y, grp, cs, seeds, leaf, trees)
            span = dates(start, a, b) if start else ""
            rows.append(dict(run=dt.datetime.now().isoformat(timespec="seconds"), spec=spec, windows=f"{a}:{b}", dates=span, classes=cs,
                             seeds=seeds, leaf=leaf, n_fields=r["n"], f1=r["f1"], bal=r["bal"], auc=r["auc"],
                             auc_by_class=" ".join(f"{k}:{v:.2f}" for k, v in r["aucs"].items())))
            print(f"windows {a:2d}:{b:2d} {span:25s} {cs}cl F1 {r['f1']:.3f} bal {r['bal']:.3f} AUC {r['auc']:.3f}  per class {rows[-1]['auc_by_class']}")
    res = pd.DataFrame(rows)
    if out:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([pd.read_csv(p), res]).to_csv(p, index=False) if p.exists() else res.to_csv(p, index=False)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--ranges", default=DEFAULT)
    ap.add_argument("--classes", default="3,4")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--leaf", type=int, default=10)
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    rg = [tuple(int(v) for v in r.split(":")) for r in a.ranges.split(",")]
    if any(not 0 <= x < y <= study.N_BINS for x, y in rg):
        raise SystemExit(f"each range must satisfy 0 <= a < b <= {study.N_BINS}")
    parts, meta = sw.load(a.spec)
    start = Path(a.spec.split("+")[0]).name[:10]
    run(parts, meta, rg, a.classes.split(","), a.seeds, a.leaf, spec=a.spec, start=start, out=Path(a.data) / "phase3" / "windows.csv")
