"""Phase 6a: does grouping pixels into objects help? TRAIN fields only, out-of-fold pixel probabilities.
python src/p6_objects.py SPEC [--asc dense_asc_mt2 --desc dense_desc_mt2] [--leaf 20] [--sizes 15,30,60] [--compactness 0.5,2]
SPEC = dense feature folders (data/phase2/dense_*); the stacks come from data/phase5/.
Variants: pixel (no objects), field (true field ids: an upper bound, since a real map has no field boundaries),
slic<size>_c<compactness> (superpixels of about <size> pixels built from the dense Sentinel-1 stack).
Rule: objects are adopted only if macro-F1 rises by 0.02 or more over the pixel result and AUC does not fall."""
import argparse
import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.segmentation import slic

import features
import p3_sweep as sw
import p4_final as p4
import p5_chip_features as cf


def pc_image(stack, comps=5):
    """(bands, h, w) stack -> (h, w, comps) principal components. Bands with no value anywhere on the chip (a missing window) are dropped,
    other gaps are filled with the band mean, bands are standardised. The components come from the band correlation matrix, which is
    far cheaper than a decomposition of the full pixel matrix."""
    b, h, w = stack.shape
    x = stack.reshape(b, -1).T.astype("float64")
    x = x[:, np.isfinite(x).any(axis=0)]
    if x.shape[1] == 0:
        raise SystemExit("pc_image: the stack has no valid band")
    mu = np.nanmean(x, axis=0)
    x = np.where(np.isnan(x), mu, x)
    x = (x - mu) / (x.std(axis=0) + 1e-9)
    _, vec = np.linalg.eigh(x.T @ x / len(x))
    z = x @ vec[:, ::-1][:, :comps]
    return (z / (z.std(axis=0) + 1e-9)).reshape(h, w, -1)


def slic_labels(z, target_px, compactness):
    h, w = z.shape[:2]
    return slic(z, n_segments=max(1, h * w // target_px), compactness=compactness, channel_axis=-1, start_label=0, convert2lab=False)


def segments(stack, target_px, compactness, comps=5):
    return slic_labels(pc_image(stack, comps), target_px, compactness)


def px_chips(meta, pix):
    return pix.loc[meta.index].chip.nunique()


def vote(P, obj):
    return pd.DataFrame(P).groupby(np.asarray(obj)).transform("mean").to_numpy()


def run(parts, meta, pix, load_stack, cs="3", leaf=20, trees=100, sizes=(15, 30, 60), compactness=(0.5, 2.0), seed=0, log=None):
    log = log or (lambda *_: None)
    t0 = time.time()
    parts, meta = p4.prep(parts, meta, cs)
    X, y, g = sw.build(parts, meta, "pixel", "raw")
    P = features.cv_predict(X, y, g, seed=seed, trees=trees, leaf=leaf, proba=True)
    res = {"pixel": sw.score(P, y), "field": sw.score(vote(P, meta.field), y)}
    log(f"out-of-fold pixel probabilities done in {time.time() - t0:.0f}s; segmenting {px_chips(meta, pix)} chips")
    px = pix.loc[meta.index]
    rows, cols = px.row.to_numpy(), px.col.to_numpy()
    variants = [(s, c) for s in sizes for c in compactness]
    obj = {v: np.zeros(len(meta), dtype=np.int64) for v in variants}
    offset = 0
    t1, groups = time.time(), px.groupby("chip").indices
    for i, (chip, idx) in enumerate(groups.items(), 1):
        z = pc_image(load_stack(chip))
        top = 0
        for v in variants:
            seg = slic_labels(z, *v)
            obj[v][idx] = seg[rows[idx], cols[idx]] + offset
            top = max(top, seg.max() + 1)
        offset += top
        if i % 25 == 0 or i == len(groups):
            log(f"{i}/{len(groups)} chips, {time.time() - t1:.0f}s, about {(time.time() - t1) / i * (len(groups) - i) / 60:.1f} min left")
    for (s, c), o in obj.items():
        res[f"slic{s}_c{c}"] = sw.score(vote(P, o), y)
    return res


def smoke(data, a):
    """Quick real-data check before the long run: chips with a missing window are included on purpose, and the segmentation time is projected."""
    pix = pd.read_csv(data / "phase1" / "pixels.csv", dtype={"chip": str})
    chips = list(dict.fromkeys(["142c9", "78eaa", *pix.chip.unique()[:1]]))
    sizes, comp = [int(v) for v in a.sizes.split(",")], [float(v) for v in a.compactness.split(",")]
    t0 = time.time()
    for chip in chips:
        st = np.concatenate([cf.dequantise(np.load(data / "phase5" / n / f"{chip}.npy")) for n in (a.asc, a.desc)])
        gap = int((~np.isfinite(st).any(axis=(1, 2))).sum())
        z = pc_image(st)
        for size in sizes:
            for c in comp:
                seg = slic_labels(z, size, c)
                print(f"chip {chip}: {gap} empty bands, target {size} px, compactness {c} -> {seg.max() + 1} segments")
    per = (time.time() - t0) / len(chips)
    n = pix[pix.split == "train"].chip.nunique()
    print(f"smoke test passed: {per:.1f} s per chip, about {per * n / 60:.1f} min for the {n} chips with training pixels (cross-validation of the pixel model comes on top)")


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
    res = run(parts, meta, pix, load, a.classes, a.leaf, log=print, sizes=[int(v) for v in a.sizes.split(",")], compactness=[float(v) for v in a.compactness.split(",")])
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
