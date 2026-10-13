"""Phase 6e: does labelled data from another region help in the study block? Block-wise CV on the study TRAIN pixels; the extra pixels
are added to every training fold and never scored.
python src/p6_extra_train.py [--leaf 20] [--seeds 2]
Study features: data/phase2/dense_asc_mt2 + dense_desc_mt2. Extra features: the same folders with the _far suffix (all fields are "external").
Chips of the extra set whose ascending slot was filled by a descending orbit are dropped, so each feature block means the same thing.
Variants: base, far (weight 1) and far_w0.3 (weight 0.3). Rule: adopted only if macro-F1 rises by 0.02 or more and AUC does not fall."""
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import features
import p3_sweep as sw
import p4_final as p4


def cv_with_extra(X, y, g, Xe=None, ye=None, we=None, seed=0, trees=100, leaf=20):
    """Out-of-fold probabilities on (X, y, g); the extra rows (Xe, ye, we) join every training fold."""
    classes = np.unique(y)
    P = np.zeros((len(y), len(classes)))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
        Xt, yt = (X[tr], y[tr]) if Xe is None else (np.vstack([X[tr], Xe]), np.concatenate([y[tr], ye]))
        w = None if Xe is None else np.concatenate([np.ones(len(tr)), we])
        m = features.make_model("rf", leaf, trees, seed).fit(Xt, yt, sample_weight=w)
        P[np.ix_(te, np.searchsorted(classes, m.classes_))] = m.predict_proba(X[te])
    return P


def run(X, y, g, Xe, ye, weights=(1.0, 0.3), seeds=2, trees=100, leaf=20):
    out = {}
    cases = [("base", None)] + [(f"far_w{w:g}" if w != 1 else "far", w) for w in weights]
    for name, w in cases:
        r = [sw.score(cv_with_extra(X, y, g, None if w is None else Xe, ye, None if w is None else np.full(len(ye), w), s, trees, leaf), y) for s in range(seeds)]
        out[name] = {k: float(np.mean([x[k] for x in r])) for k in ("f1", "bal", "oa", "auc")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaf", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    d = Path(a.data) / "phase2"
    parts, meta = sw.load(f"{d / 'dense_asc_mt2'}+{d / 'dense_desc_mt2'}")
    parts, meta = p4.prep(parts, meta, "3")
    X, y, g = sw.build(parts, meta, "pixel", "raw")
    fp, fm = sw.load(f"{d / 'dense_asc_mt2_far'}+{d / 'dense_desc_mt2_far'}", splits=("external",))
    orb = pd.read_csv(d / "dense_asc_mt2_far" / "features.csv", usecols=["pid", "orbit"]).set_index("pid").orbit
    ok = orb.reindex(fm.index).str.startswith("ASC").to_numpy()
    print(f"extra set: {len(fm)} pixels, {int((~ok).sum())} dropped because their ascending slot holds a descending orbit")
    fp, fm = [p[ok] for p in fp], fm[ok]
    fp, fm = p4.prep(fp, fm, "3")
    Xe, ye, _ = sw.build(fp, fm, "pixel", "raw")
    print(f"study train pixels {len(y)} {np.bincount(y.astype(int))[[1, 2, 6]].tolist()}, extra pixels {len(ye)} {np.bincount(ye.astype(int), minlength=7)[[1, 2, 6]].tolist()} (Wheat, Mustard+Lentil, Sugarcane)")
    res = run(X, y, g, Xe, ye, seeds=a.seeds, leaf=a.leaf)
    base = res["base"]
    rows = []
    for k, r in res.items():
        verdict = "" if k == "base" else ("ADOPT" if r["f1"] - base["f1"] >= 0.02 and r["auc"] >= base["auc"] else "no")
        print(f"{k:9s} F1 {r['f1']:.3f} bal {r['bal']:.3f} OA {r['oa']:.3f} AUC {r['auc']:.3f} {verdict}")
        rows.append(dict(run=dt.datetime.now().isoformat(timespec="seconds"), variant=k, leaf=a.leaf, seeds=a.seeds, verdict=verdict, **r))
    out = Path(a.data) / "phase6" / "extra_train.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    pd.concat([pd.read_csv(out), df]).to_csv(out, index=False) if out.exists() else df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
