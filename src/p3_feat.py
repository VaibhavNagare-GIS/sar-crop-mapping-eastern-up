"""Phase 3f: SAR feature-upgrade ladder at FIELD level. TRAIN fields only (the test split is never read).
Usage: python p3_feat.py SPEC [--angle asc.csv[,desc.csv]] [--embed data/phase2/embed.csv] [--classes 3,4] [--seeds 5] [--leaf 10]
SPEC = feature folders joined by '+' (as in p3_sweep). --angle = one angle file per Sentinel-1 folder, in order.
Rungs:  base   per-field mean of dB values (what p3_sweep does; must reproduce its numbers)
        lin    average in linear power, then back to dB (the physically correct average)
        inner  lin, using only pixels whose 4 neighbours belong to the same field (fields with none keep all pixels)
        std    inner + within-field spread (time-mean of the pixel std, VV and VH)
        angle  std + mean incidence angle as a feature                    (needs --angle)
        norm   std with dB shifted to a 39 degree reference by the cos^2 law (needs --angle)
        emb    Google embeddings alone, and fused with the best SAR rung   (needs --embed; NOT SAR-only, benchmark)
Rule written in advance: a rung is KEPT if macro-F1 rises by >= 0.02 over the last kept rung and AUC does not fall.
The rule is applied to the FIRST class scheme listed (default 3 = the map product). Appends to data/phase3/feat.csv."""
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

import features
import p3_sweep as sw

REF = 39.0   # degrees, middle of the IW incidence-angle range (only used for the 'norm' rung)
RUNGS = {"base": dict(lin=0, inner=0, std=0, ang=""), "lin": dict(lin=1, inner=0, std=0, ang=""),
         "inner": dict(lin=1, inner=1, std=0, ang=""), "std": dict(lin=1, inner=1, std=1, ang=""),
         "angle": dict(lin=1, inner=1, std=1, ang="add"), "norm": dict(lin=1, inner=1, std=1, ang="norm")}


def s1_cols(df):
    return [c for c in df.columns if c.startswith(("VV_", "VH_"))]


def inner_mask(pix):
    """True where all 4 neighbours (same chip) carry the same field id. pix needs chip,row,col,field."""
    key = dict(zip(zip(pix.chip, pix.row, pix.col), pix.field))
    nb = ((1, 0), (-1, 0), (0, 1), (0, -1))
    ok = [all(key.get((c, r + dr, k + dc)) == f for dr, dc in nb) for c, r, k, f in zip(pix.chip, pix.row, pix.col, pix.field)]
    return pd.Series(ok, index=pix.index)


def field_X(parts, meta, v, inner=None, angles=()):
    """Field-level matrix for one rung setting v. Returns X (fields x features) in the sorted-field order of meta.groupby('field')."""
    g = meta.field
    keep = pd.Series(True, index=meta.index)
    if v["inner"]:
        inn = inner.reindex(meta.index).fillna(False)
        keep = inn | ~inn.groupby(g).transform("any")          # fields with no inner pixel keep all pixels
    blocks, ai = [], 0
    for p in parts:
        cols = s1_cols(p)
        p = p.copy()
        ang = None
        if cols:
            if v["ang"]:
                if ai >= len(angles):
                    raise SystemExit("this rung needs --angle with one file per Sentinel-1 folder")
                ang = angles[ai].reindex(p.index)
                if ang.isna().any():
                    raise SystemExit(f"{int(ang.isna().sum())} pixels have no incidence angle; rerun p2_extra.py angle")
                ai += 1
            if v["ang"] == "norm":
                p[cols] = p[cols].to_numpy() + (20 * np.log10(np.cos(np.radians(REF)) / np.cos(np.radians(ang.to_numpy()))))[:, None]
        pk, gk = p[keep], g[keep]
        m = pk.groupby(gk).mean()
        if cols and v["lin"]:
            m[cols] = 10 * np.log10((10 ** (pk[cols] / 10)).groupby(gk).mean())
        blocks.append(features.make_X(m, "raw")[0])
        if cols and v["std"]:
            vv = [c for c in cols if c.startswith("VV_")]
            vh = [c for c in cols if c.startswith("VH_")]
            blocks.append(np.column_stack([pk[vv].groupby(gk).std(ddof=0).mean(axis=1), pk[vh].groupby(gk).std(ddof=0).mean(axis=1)]))
        if cols and v["ang"] == "add":
            blocks.append(ang[keep].groupby(gk).mean().to_numpy()[:, None])
    return np.hstack(blocks)


def evaluate(X, y, g, cs, seeds, leaf, trees):
    k = y != sw.FALLOW
    X, y, g = X[k], y[k], g[k]
    if cs == "3":
        y = np.where(np.isin(y, (2, 3)), 2, y)
    r = [sw.score(features.cv_predict(X, y, g, seed=s, trees=trees, leaf=leaf, proba=True), y) for s in range(seeds)]
    return dict(f1=np.mean([x["f1"] for x in r]), auc=np.mean([x["auc"] for x in r]), bal=np.mean([x["bal"] for x in r]), n=len(y),
                aucs={c: float(np.mean([x["aucs"][c] for x in r])) for c in r[0]["aucs"]})


def run(parts, meta, pix, angles=(), embed=None, classes=("3", "4"), seeds=5, leaf=10, trees=100, spec="", out=None):
    fm = meta.groupby("field").agg(label=("label", "first"), block=("block", "first"))
    y, g = fm.label.to_numpy().astype(int), fm.block.to_numpy()
    inner = inner_mask(pix)
    print(f"inner pixels: {inner.reindex(meta.index).mean():.1%} of labelled pixels")
    Xs = {}
    for name, v in RUNGS.items():
        if v["ang"] and not len(angles):
            continue
        X = field_X(parts, meta, v, inner, angles)
        assert len(X) == len(fm)
        Xs[name] = X
    if embed is not None:
        E = embed.reindex(meta.index)
        if E.isna().any().any():
            raise SystemExit("embedding table has missing pixels; rerun p2_extra.py embed")
        Xe = E.groupby(meta.field).mean().to_numpy()
    rows, kept = [], {}
    def add(name, X, decide):
        for cs in classes:
            r = evaluate(X, y, g, cs, seeds, leaf, trees)
            row = dict(run=dt.datetime.now().isoformat(timespec="seconds"), spec=spec, rung=name, classes=cs, seeds=seeds, leaf=leaf,
                       n_fields=r["n"], n_features=X.shape[1], f1=r["f1"], bal=r["bal"], auc=r["auc"], verdict="")
            if decide and cs == classes[0]:
                ref = kept.get(cs)
                row["verdict"] = "reference" if ref is None else ("KEEP" if r["f1"] - ref["f1"] >= 0.02 and r["auc"] >= ref["auc"] else "drop")
                if ref is None or row["verdict"] == "KEEP":
                    kept[cs] = dict(f1=r["f1"], auc=r["auc"], name=name, X=X)
            rows.append(row)
            print(f"{name:6s} {cs}cl F1 {r['f1']:.3f} bal {r['bal']:.3f} AUC {r['auc']:.3f} feats {X.shape[1]:4d} {row['verdict']}")
    for name, X in Xs.items():
        add(name, X, True)
    if embed is not None:
        add("emb", Xe, False)
        best = max((r for r in rows if r["classes"] == classes[0] and r["rung"] in Xs), key=lambda r: r["f1"])["rung"]
        add(f"{best}+emb", np.hstack([Xs[best], Xe]), False)
    res = pd.DataFrame(rows)
    if out:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([pd.read_csv(p), res]).to_csv(p, index=False) if p.exists() else res.to_csv(p, index=False)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--angle", default="")
    ap.add_argument("--embed", default="")
    ap.add_argument("--classes", default="3,4")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--leaf", type=int, default=10)
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    parts, meta = sw.load(a.spec)
    pix = pd.read_csv(Path(a.data) / "phase1" / "pixels.csv", dtype={"chip": str})
    angles = [pd.read_csv(f, index_col="pid").ANG for f in a.angle.split(",") if f]
    n_s1 = sum(bool(s1_cols(p)) for p in parts)
    if angles and len(angles) != n_s1:
        raise SystemExit(f"--angle needs {n_s1} file(s), got {len(angles)}")
    emb = pd.read_csv(a.embed, index_col="pid") if a.embed else None
    run(parts, meta, pix, angles, emb, a.classes.split(","), a.seeds, a.leaf, spec=a.spec, out=Path(a.data) / "phase3" / "feat.csv")
