"""Phase 6a: does grouping pixels into objects help? TRAIN fields only, out-of-fold pixel probabilities.
python src/p6_objects.py SPEC [--asc dense_asc_mt2 --desc dense_desc_mt2] [--leaf 20] [--sizes 15,30,60] [--compactness 0.5,2]
SPEC = dense feature folders (data/phase2/dense_*); the stacks come from data/phase5/.
Variants: pixel (no objects), field (true field ids: an upper bound, since a real map has no field boundaries),
slic<size>_c<compactness> (superpixels of about <size> pixels built from the dense Sentinel-1 stack).
Rule: objects are adopted only if macro-F1 rises by 0.02 or more over the pixel result and AUC does not fall."""
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.segmentation import slic

import features
import p3_sweep as sw
import p4_final as p4
import p5_chip_features as cf


def segments(stack, target_px, compactness, comps=5):
    """Superpixels of a (bands, h, w) stack. Bands with no value anywhere on the chip (a missing window) are dropped, remaining gaps
    are filled with the band mean, bands are standardised and reduced to `comps` principal components."""
    b, h, w = stack.shape
    x = stack.reshape(b, -1).T.astype("float64")
    x = x[:, np.isfinite(x).any(axis=0)]
    if x.shape[1] == 0:
        raise SystemExit("segments: the stack has no valid band")
    mu = np.nanmean(x, axis=0)
    x = np.where(np.isnan(x), mu, x)
    x = (x - mu) / (x.std(axis=0) + 1e-9)
    u, _, _ = np.linalg.svd(x, full_matrices=False)
    z = u[:, :comps] * np.sqrt(len(x))
    return slic(z.reshape(h, w, -1), n_segments=max(1, h * w // target_px), compactness=compactness, channel_axis=-1, start_label=0, convert2lab=False)


def vote(P, obj):
    return pd.DataFrame(P).groupby(np.asarray(obj)).transform("mean").to_numpy()


def run(parts, meta, pix, load_stack, cs="3", leaf=20, trees=100, sizes=(15, 30, 60), compactness=(0.5, 2.0), seed=0):
    parts, meta = p4.prep(parts, meta, cs)
    X, y, g = sw.build(parts, meta, "pixel", "raw")
    P = features.cv_predict(X, y, g, seed=seed, trees=trees, leaf=leaf, proba=True)
    res = {"pixel": sw.score(P, y), "field": sw.score(vote(P, meta.field), y)}
    px = pix.loc[meta.index]
    rows, cols = px.row.to_numpy(), px.col.to_numpy()
    variants = [(s, c) for s in sizes for c in compactness]
    obj = {v: np.zeros(len(meta), dtype=np.int64) for v in variants}
    offset = 0
    for chip, idx in px.groupby("chip").indices.items():
        st = load_stack(chip)
        top = 0
        for v in variants:
            seg = segments(st, *v)
            obj[v][idx] = seg[rows[idx], cols[idx]] + offset
            top = max(top, seg.max() + 1)
        offset += top
    for (s, c), o in obj.items():
        res[f"slic{s}_c{c}"] = sw.score(vote(P, o), y)
    return res


def smoke(data, a):
    """Quick real-data check before the long run: chips with a missing window are included on purpose."""
    pix = pd.read_csv(data / "phase1" / "pixels.csv", dtype={"chip": str})
    chips = list(dict.fromkeys(["142c9", "78eaa", *pix.chip.unique()[:1]]))
    for chip in chips:
        st = np.concatenate([cf.dequantise(np.load(data / "phase5" / n / f"{chip}.npy")) for n in (a.asc, a.desc)])
        gap = int((~np.isfinite(st).any(axis=(1, 2))).sum())
        for size in [int(v) for v in a.sizes.split(",")]:
            seg = segments(st, size, float(a.compactness.split(",")[0]))
            print(f"chip {chip}: {gap} empty bands, target {size} px -> {seg.max() + 1} segments, mean {st.shape[1] * st.shape[2] / (seg.max() + 1):.0f} px")
    print("smoke test passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--asc", default="dense_asc_mt2")
    ap.add_argument("--desc", default="dense_desc_mt2")
    ap.add_argument("--leaf", type=int, default=20)
    ap.add_argument("--sizes", default="15,30,60")
    ap.add_argument("--compactness", default="0.5,2")
    ap.add_argument("--classes", default="3")
    ap.add_argument("--data", default="data")
    ap.add_argument("--smoke", action="store_true", help="only segment the first 3 chips of each size and report segment sizes")
    a = ap.parse_args()
    data = Path(a.data)
    if a.smoke:
        smoke(data, a)
        return
    parts, meta = sw.load(a.spec)
    pix = pd.read_csv(data / "phase1" / "pixels.csv", dtype={"chip": str})
    load = lambda chip: np.concatenate([cf.dequantise(np.load(data / "phase5" / n / f"{chip}.npy")) for n in (a.asc, a.desc)])
    res = run(parts, meta, pix, load, a.classes, a.leaf, sizes=[int(v) for v in a.sizes.split(",")], compactness=[float(v) for v in a.compactness.split(",")])
    base = res["pixel"]
    rows = []
    for k, r in res.items():
        verdict = "" if k in ("pixel", "field") else ("ADOPT" if r["f1"] - base["f1"] >= 0.02 and r["auc"] >= base["auc"] else "no")
        rows.append(dict(run=dt.datetime.now().isoformat(timespec="seconds"), spec=a.spec, classes=a.classes, leaf=a.leaf, variant=k, f1=r["f1"], bal=r["bal"], oa=r["oa"], auc=r["auc"], verdict=verdict))
        print(f"{k:14s} F1 {r['f1']:.3f} bal {r['bal']:.3f} OA {r['oa']:.3f} AUC {r['auc']:.3f} {verdict}")
    out = data / "phase6" / "objects.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    pd.concat([pd.read_csv(out), df]).to_csv(out, index=False) if out.exists() else df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
