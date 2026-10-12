"""Phase 6d: map-level cross-validation on TRAIN chips. It repeats what the final map does: a model trained on some chips predicts every
pixel of the held-out chips, and the prediction is then regularised. Scores are taken at the labelled pixels only.
python src/p6_densecv.py [--asc dense_asc_mt2 --desc dense_desc_mt2] [--max-chips N]
Variants: pixel, smooth3/5/7 (mean filter of the class-probability maps), grid8 (8 x 8 pixel blocks), slic60/slic120 (probabilities
averaged over superpixels, all pixels of the object), objrf60/objrf120 (a Random Forest trained on superpixel means), and
field_true (probabilities averaged over the labelled pixels of each true field: an upper bound, boundaries are unknown on a map).
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
import p3_sweep as sw
import p4_final as p4
import p5_chip_features as cf
import p6_objects as ob

CLASSES = np.array([1, 2, 6])      # Wheat, Mustard+Lentil, Sugarcane
SIZES, GRID, SMOOTH = (60, 120), 8, (3, 5, 7)
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


def prepare(stack, flat, yi, sizes, compactness):
    """Superpixels of a chip and, for each, mean features plus the label of its labelled pixels (flat = row * w + col of labelled pixels)."""
    X, z, out = dense_features(stack), ob.pc_image(stack), {}
    for s in sizes:
        seg = ob.slic_labels(z, s, compactness).ravel()
        n = seg.max() + 1
        cnt = np.zeros((n, len(CLASSES)))
        np.add.at(cnt, (seg[flat], yi), 1)
        nl = cnt.sum(1)
        out[s] = dict(seg=seg, F=means_by(X, seg, n), nl=nl, lab=cnt.argmax(1), purity=cnt.max(1) / np.maximum(nl, 1))
    return out


def run(parts, meta, pix, load_stack, leaf_pixel=20, leaf_obj=10, trees=100, sizes=SIZES, compactness=1.0, seed=0, n_splits=5, log=None):
    log = log or (lambda *_: None)
    parts, meta = p4.prep(parts, meta, "3")
    Xp, y, _ = sw.build(parts, meta, "pixel", "raw")
    yi = np.searchsorted(CLASSES, y)
    px = pix.loc[meta.index]
    chip_of, field_of = px.chip.to_numpy(), meta.field.to_numpy()
    chips = np.array(sorted(set(chip_of)))
    block = px.groupby("chip").block.agg(lambda s: s.mode().iloc[0]).loc[chips].to_numpy()
    n_splits = min(n_splits, len(set(block)))
    if n_splits < 2:
        raise SystemExit("need labelled chips from at least two spatial blocks")
    rows_of = {c: np.flatnonzero(chip_of == c) for c in chips}
    h, w = load_stack(chips[0]).shape[1:]
    flat = px.row.to_numpy() * w + px.col.to_numpy()

    t0, cache = time.time(), {}
    for i, c in enumerate(chips, 1):
        cache[c] = prepare(load_stack(c), flat[rows_of[c]], yi[rows_of[c]], sizes, compactness)
        if i % 25 == 0 or i == len(chips):
            log(f"objects: {i}/{len(chips)} chips, {time.time() - t0:.0f}s, about {(time.time() - t0) / i * (len(chips) - i) / 60:.1f} min left")

    keys = ["pixel", *[f"smooth{k}" for k in SMOOTH], f"grid{GRID}", *[f"slic{s}" for s in sizes], *[f"objrf{s}" for s in sizes]]
    P = {k: np.zeros((len(y), len(CLASSES))) for k in keys}
    t1, done = time.time(), 0
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(chips, groups=block)):
        trc, tec = chips[tr], chips[te]
        m = np.isin(chip_of, trc)
        pm = features.make_model("rf", leaf_pixel, trees, seed).fit(Xp[m], yi[m])
        om = {}
        for s in sizes:
            F = np.vstack([cache[c][s]["F"][ok] for c in trc for ok in [(cache[c][s]["nl"] >= MIN_LABELLED) & (cache[c][s]["purity"] >= MIN_PURITY)]])
            L = np.concatenate([cache[c][s]["lab"][ok] for c in trc for ok in [(cache[c][s]["nl"] >= MIN_LABELLED) & (cache[c][s]["purity"] >= MIN_PURITY)]])
            om[s] = features.make_model("rf", leaf_obj, trees, seed).fit(F, L)
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
            for s in sizes:
                o = cache[c][s]
                maps[f"slic{s}"] = means_by(pr.reshape(h * w, -1), o["seg"], len(o["nl"]))[o["seg"]].reshape(h, w, -1)
                maps[f"objrf{s}"] = om[s].predict_proba(o["F"])[o["seg"]].reshape(h, w, -1)
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
    t0 = time.time()
    res = run(parts, meta, pix, load, log=print)
    base = res["pixel"]
    rows = []
    for k, r in res.items():
        verdict = "" if k in ("pixel", "field_true") else ("ADOPT" if r["f1"] - base["f1"] >= 0.02 and r["auc"] >= base["auc"] else "no")
        print(f"{k:11s} F1 {r['f1']:.3f} bal {r['bal']:.3f} OA {r['oa']:.3f} AUC {r['auc']:.3f} {verdict}")
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
