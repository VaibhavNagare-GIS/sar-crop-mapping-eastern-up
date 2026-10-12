"""Phase 6c: do neighbourhood (context and texture) features help? TRAIN fields only, block-wise CV on pixels.
python src/p6_context.py SPEC [--asc dense_asc_mt2 --desc dense_desc_mt2] [--leaf 20] [--classes 3]
Added features, per orbit and polarisation, from the time-mean backscatter map of the chip (16 in total):
local mean (5 x 5 window), local standard deviation (3 x 3 and 5 x 5) and local entropy (disk of radius 2, 8-bit scaled).
Rule: keep only if macro-F1 rises by 0.02 or more over the base features and AUC does not fall."""
import argparse
import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from skimage.filters.rank import entropy
from skimage.morphology import disk

import features
import p3_sweep as sw
import p4_final as p4
import p5_chip_features as cf

LO, HI = -30.0, 5.0     # dB range mapped to 8 bits for the entropy filter


def context_maps(stack):
    """(bands, h, w) stack of asc VV, asc VH, desc VV, desc VH windows -> (16, h, w) context and texture maps."""
    n = stack.shape[0] // 4
    maps = []
    for k in range(4):
        with np.errstate(all="ignore"):
            m = np.nanmean(stack[k * n:(k + 1) * n], axis=0)
        m = np.where(np.isfinite(m), m, np.nanmean(m) if np.isfinite(m).any() else 0.0)
        mean5 = uniform_filter(m, 5)
        std3 = np.sqrt(np.maximum(uniform_filter(m * m, 3) - uniform_filter(m, 3) ** 2, 0))
        std5 = np.sqrt(np.maximum(uniform_filter(m * m, 5) - mean5 ** 2, 0))
        q = np.clip((m - LO) / (HI - LO) * 255, 0, 255).astype(np.uint8)
        maps += [mean5, std3, std5, entropy(q, disk(2)).astype("float64")]
    return np.stack(maps)


def run(parts, meta, pix, load_stack, cs="3", leaf=20, trees=100, seeds=2, log=None):
    log = log or (lambda *_: None)
    parts, meta = p4.prep(parts, meta, cs)
    X, y, g = sw.build(parts, meta, "pixel", "raw")
    px = pix.loc[meta.index]
    ctx = np.zeros((len(meta), 16))
    rows, cols = px.row.to_numpy(), px.col.to_numpy()
    t0, groups = time.time(), px.groupby("chip").indices
    for i, (chip, idx) in enumerate(groups.items(), 1):
        ctx[idx] = context_maps(load_stack(chip))[:, rows[idx], cols[idx]].T
        if i % 25 == 0 or i == len(groups):
            log(f"context features: {i}/{len(groups)} chips, {time.time() - t0:.0f}s")
    out = {}
    for name, M in (("base", X), ("base+context", np.hstack([X, ctx]))):
        r = [sw.score(features.cv_predict(M, y, g, seed=s, trees=trees, leaf=leaf, proba=True), y) for s in range(seeds)]
        out[name] = {k: float(np.mean([x[k] for x in r])) for k in ("f1", "bal", "oa", "auc")}
        out[name]["n_features"] = M.shape[1]
        log(f"{name}: cross-validation done")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--asc", default="dense_asc_mt2")
    ap.add_argument("--desc", default="dense_desc_mt2")
    ap.add_argument("--leaf", type=int, default=20)
    ap.add_argument("--classes", default="3")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    data = Path(a.data)
    parts, meta = sw.load(a.spec)
    pix = pd.read_csv(data / "phase1" / "pixels.csv", dtype={"chip": str})
    load = lambda chip: np.concatenate([cf.dequantise(np.load(data / "phase5" / n / f"{chip}.npy")) for n in (a.asc, a.desc)])
    res = run(parts, meta, pix, load, a.classes, a.leaf, log=print)
    b, c = res["base"], res["base+context"]
    verdict = "ADOPT" if c["f1"] - b["f1"] >= 0.02 and c["auc"] >= b["auc"] else "no"
    for k, r in res.items():
        print(f"{k:13s} F1 {r['f1']:.3f} bal {r['bal']:.3f} OA {r['oa']:.3f} AUC {r['auc']:.3f} features {r['n_features']}")
    print(f"change in F1 {c['f1'] - b['f1']:+.3f}, AUC {c['auc'] - b['auc']:+.3f} -> {verdict}")
    out = data / "phase6" / "context.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([dict(run=dt.datetime.now().isoformat(timespec="seconds"), spec=a.spec, classes=a.classes, leaf=a.leaf, variant=k, verdict=verdict if k != "base" else "", **r) for k, r in res.items()])
    pd.concat([pd.read_csv(out), df]).to_csv(out, index=False) if out.exists() else df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
