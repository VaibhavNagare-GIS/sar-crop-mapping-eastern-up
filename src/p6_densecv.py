"""Phase 6d: map-level cross-validation on TRAIN chips. It repeats what the final map does: a model trained on some chips predicts every
pixel of the held-out chips, and the prediction is then regularised. Scores are taken at the labelled pixels only.
python src/p6_densecv.py [--asc dense_asc_mt2 --desc dense_desc_mt2] [--max-chips N]
Variants: pixel, smooth3/5/7 (mean filter of the class-probability maps), grid8 (8 x 8 pixel blocks), seg<size>_c<compactness>
(probabilities averaged over superpixels, all pixels of the object), objrf<size>_c<compactness> (a Random Forest trained on superpixel
means), and field_true (probabilities averaged over the labelled pixels of each true field: an upper bound, boundaries are unknown on a map).
Superpixels come from the dataset's own 12-band Sentinel-2 snapshot (--objects snapshot, used for geometry only; the classifier sees
Sentinel-1 only) or from the Sentinel-1 stack (--objects s1). Each segmentation prints the share of image variance it explains beyond square tiles: about 0 means the
superpixels are a plain grid and carry no information about field boundaries.
Folds are whole chips grouped by their 0.1 degree block. Rule: a variant is adopted only if macro-F1 rises by 0.02 or more over
'pixel' and AUC does not fall. --max-chips gives a quick run and a projected total time."""
import argparse
import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from sklearn.model_selection import GroupKFold

import features
import nnprep
import p3_sweep as sw
import p4_final as p4
import p5_chip_features as cf
import p6_objects as ob

CLASSES = np.array([1, 2, 6])      # Wheat, Mustard+Lentil, Sugarcane
SIZES, COMPACT, GRID, SMOOTH = (25, 60), (0.1, 0.5), 8, (3, 5, 7)
MIN_LABELLED, MIN_PURITY = 5, 0.7  # an object is a training sample if it has this many labelled pixels and this label share


def dense_features(stack):
    """(bands, h, w) stack of asc then desc windows -> (h*w, F), the same features as the training table."""
    blocks = []
    for s in range(stack.shape[0] // 40):
        df = pd.DataFrame(stack[s * 40:(s + 1) * 40].reshape(40, -1).T, columns=cf.NAMES)
        blocks.append(features.make_X(df, "raw")[0])
    return np.nan_to_num(np.hstack(blocks))


def means_by(values, seg, n):
    cnt = np.bincount(seg, minlength=n).astype(float)
    return np.stack([np.bincount(seg, weights=values[:, j], minlength=n) for j in range(values.shape[1])], 1) / np.maximum(cnt, 1)[:, None]


def seg_gain(z, seg, tile):
    """Variance of the image z explained by the segmentation, relative to square tiles of side `tile`: 0 means no better than a grid,
    larger means the segments follow the content."""
    h, w = z.shape[:2]
    z2 = z.reshape(h * w, -1)
    g = ((np.arange(h)[:, None] // tile) * (w // tile + 1) + (np.arange(w)[None, :] // tile)).ravel()
    sse = lambda lab: float(((z2 - means_by(z2, lab, lab.max() + 1)[lab]) ** 2).sum())
    return 1 - sse(seg) / sse(g)


def prepare(X, z, h, w, flat, yi, sizes, compactness):
    """Superpixels of a chip from the image z, and for each: mean features and the label of its labelled pixels (flat = row * w + col)."""
    out = {}
    for s in sizes:
        for c in compactness:
            seg = ob.slic_labels(z, s, c).ravel()
            n = seg.max() + 1
            cnt = np.zeros((n, len(CLASSES)))
            np.add.at(cnt, (seg[flat], yi), 1)
            nl = cnt.sum(1)
            out[(s, c)] = dict(seg=seg, F=means_by(X, seg, n), nl=nl, lab=cnt.argmax(1), purity=cnt.max(1) / np.maximum(nl, 1),
                               gain=seg_gain(z, seg, max(2, int(round(s ** 0.5)))))
    return out


def run(parts, meta, pix, load_stack, load_z=None, leaf_pixel=20, leaf_obj=10, trees=100, sizes=SIZES, compactness=COMPACT, seed=0, n_splits=5, log=None):
    log = log or (lambda *_: None)
    parts, meta = p4.prep(parts, meta, "3")
    Xp, y, _ = sw.build(parts, meta, "pixel", "raw")
    yi = np.searchsorted(CLASSES, y)
    px = pix.loc[meta.index]
    chip_of, field_of = px.chip.to_numpy(), meta.field.to_numpy()
    chips, block = nnprep.chip_blocks(chip_of, px.block.to_numpy())
    n_splits = min(n_splits, len(set(block)))
    if n_splits < 2:
        raise SystemExit("need labelled chips from at least two spatial blocks")
    rows_of = {c: np.flatnonzero(chip_of == c) for c in chips}
    h, w = load_stack(chips[0]).shape[1:]
    flat = px.row.to_numpy() * w + px.col.to_numpy()

    t0, cache = time.time(), {}
    for i, c in enumerate(chips, 1):
        st = load_stack(c)
        cache[c] = prepare(dense_features(st), load_z(c) if load_z else ob.pc_image(st), h, w, flat[rows_of[c]], yi[rows_of[c]], sizes, compactness)
        if i % 25 == 0 or i == len(chips):
            log(f"objects: {i}/{len(chips)} chips, {time.time() - t0:.0f}s, about {(time.time() - t0) / i * (len(chips) - i) / 60:.1f} min left")

    combos = [(s, c) for s in sizes for c in compactness]
    gains = {k: float(np.mean([cache[ch][k]["gain"] for ch in chips])) for k in combos}
    log("variance explained by the superpixels beyond square tiles (0 = a plain grid): " + ", ".join(f"{s}px c{c}: {gains[(s, c)]:.2f}" for s, c in combos))
    for k, v in gains.items():
        if v < 0.05:
            log(f"WARNING: superpixels {k[0]}px c{k[1]} do not follow the image (gain {v:.2f}); they behave like a grid")
    keys = ["pixel", *[f"smooth{k}" for k in SMOOTH], f"grid{GRID}", *[f"seg{s}_c{c}" for s, c in combos], *[f"objrf{s}_c{c}" for s, c in combos]]
    P = {k: np.zeros((len(y), len(CLASSES))) for k in keys}
    t1, done = time.time(), 0
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(chips, groups=block)):
        trc, tec = chips[tr], chips[te]
        m = np.isin(chip_of, trc)
        pm = features.make_model("rf", leaf_pixel, trees, seed).fit(Xp[m], yi[m])
        om = {}
        for k in combos:
            ok = {c: (cache[c][k]["nl"] >= MIN_LABELLED) & (cache[c][k]["purity"] >= MIN_PURITY) for c in trc}
            om[k] = features.make_model("rf", leaf_obj, trees, seed).fit(np.vstack([cache[c][k]["F"][ok[c]] for c in trc]), np.concatenate([cache[c][k]["lab"][ok[c]] for c in trc]))
        for c in tec:
            idx = rows_of[c]
            X = dense_features(load_stack(c))
            pr = pm.predict_proba(X).reshape(h, w, -1)
            maps = {"pixel": pr}
            for k in SMOOTH:
                maps[f"smooth{k}"] = uniform_filter(pr, size=(k, k, 1), mode="nearest")
            g = (np.arange(h)[:, None] // GRID) * (w // GRID + 1) + (np.arange(w)[None, :] // GRID)
            gm = means_by(pr.reshape(h * w, -1), g.ravel(), g.max() + 1)
            maps[f"grid{GRID}"] = gm[g.ravel()].reshape(h, w, -1)
            for k in combos:
                o = cache[c][k]
                maps[f"seg{k[0]}_c{k[1]}"] = means_by(pr.reshape(h * w, -1), o["seg"], len(o["nl"]))[o["seg"]].reshape(h, w, -1)
                maps[f"objrf{k[0]}_c{k[1]}"] = om[k].predict_proba(o["F"])[o["seg"]].reshape(h, w, -1)
            for k in keys:
                P[k][idx] = maps[k].reshape(h * w, -1)[flat[idx]]
            done += 1
            if done % 10 == 0 or done == len(chips):
                log(f"fold {fold + 1}/{n_splits}: {done}/{len(chips)} chips predicted, {time.time() - t1:.0f}s")
    out = {k: sw.score(v, yi) for k, v in P.items()}
    out["field_true"] = sw.score(pd.DataFrame(P["pixel"]).groupby(field_of).transform("mean").to_numpy(), yi)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asc", default="dense_asc_mt2")
    ap.add_argument("--desc", default="dense_desc_mt2")
    ap.add_argument("--objects", choices=["snapshot", "s1"], default="snapshot")
    ap.add_argument("--max-chips", type=int, default=0)
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    data = Path(a.data)
    parts, meta = sw.load("+".join(str(data / "phase2" / n) for n in (a.asc, a.desc)))
    pix = pd.read_csv(data / "phase1" / "pixels.csv", dtype={"chip": str})
    if a.max_chips:
        chips = np.random.default_rng(0).permutation(pix.loc[meta.index].chip.unique())[:a.max_chips]
        keep = pix.loc[meta.index].chip.isin(chips).to_numpy()
        parts, meta = [p[keep] for p in parts], meta[keep]
    load = lambda chip: np.concatenate([cf.dequantise(np.load(data / "phase5" / n / f"{chip}.npy")) for n in (a.asc, a.desc)])
    load_z = None
    if a.objects == "snapshot":
        import p3_labelcheck as lc
        load_z = lambda chip: ob.pc_image(lc.read_snapshot(data, chip)[1])
    t0 = time.time()
    res = run(parts, meta, pix, load, load_z, log=print)
    base, inc = res["pixel"], res["smooth5"]
    rows = []
    print("verdicts: first against the pixel model, then against smooth5 (the simplest regulariser)")
    for k, r in res.items():
        rule = lambda ref: "yes" if r["f1"] - ref["f1"] >= 0.02 and r["auc"] >= ref["auc"] else "no"
        verdict = "" if k in ("pixel", "smooth5", "field_true") else f"{rule(base)} / {rule(inc)}"
        print(f"{k:16s} F1 {r['f1']:.3f} bal {r['bal']:.3f} OA {r['oa']:.3f} AUC {r['auc']:.3f} {verdict}")
        rows.append(dict(run=dt.datetime.now().isoformat(timespec="seconds"), asc=a.asc, desc=a.desc, chips=a.max_chips or "all", variant=k, f1=r["f1"], bal=r["bal"], oa=r["oa"], auc=r["auc"], verdict=verdict))
    n_all = pix.loc[meta.index].chip.nunique() if not a.max_chips else a.max_chips
    if a.max_chips:
        print(f"quick run on {a.max_chips} chips took {(time.time() - t0) / 60:.1f} min; the full run is roughly {(time.time() - t0) / a.max_chips * 367 / 60:.0f} min")
    out = data / "phase6" / "densecv.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    pd.concat([pd.read_csv(out), df]).to_csv(out, index=False) if out.exists() else df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
