"""Phase 3b-2: are the labels learnable and aligned? Uses the dataset's own Sentinel-2 snapshot (train fields only).

Usage: python p3_labelcheck.py [data_dir]
 A  classify the 12-band snapshot at labelled pixels (spatial-block CV): same score table as p3_models.py
 B  alignment test: within-field NDVI variance when the label map is shifted by up to 3 pixels (minimum should be at 0,0)
 C  acquisition-date guess: Spearman correlation of snapshot NDVI with each 12-day NDVI window of the Earth Engine
    table (needs data/phase2/<start>_s2/features.csv; skipped if missing)
"""
import re
import sys
from pathlib import Path

import env_fix  # must come before rasterio
import numpy as np
import pandas as pd
import rasterio

import p0_inventory as inv
import p3_models as p3
import study

SHIFT = 3
BAND = re.compile(r"_(B\d+A?)_10m\.tif$")


def read_snapshot(data, chip):
    """-> (bands list, array (bands, H, W) float32) for one chip, bands sorted by name."""
    files = sorted(Path(data, "source").glob(f"*_{chip}/*_10m.tif"))
    found = {BAND.search(f.name).group(1): f for f in files if BAND.search(f.name)}
    if not found:
        raise SystemExit(f"No source bands for chip {chip}; run p3_snapshot_download.py")
    arrs = []
    for b in sorted(found):
        with rasterio.open(found[b]) as s:
            arrs.append(s.read(1).astype("float32"))
    return sorted(found), np.stack(arrs)


def ndvi(arr, bands):
    nir, red = arr[bands.index("B08")], arr[bands.index("B04")]
    den = nir + red
    return np.where(den > 0, (nir - red) / np.where(den > 0, den, 1), np.nan)


def shift(a, dy, dx, m=SHIFT):
    """out[y, x] = a[y - dy, x - dx], zeros where undefined."""
    h, w = a.shape
    p = np.pad(a, ((m, m), (m, m)))
    return p[m - dy:m - dy + h, m - dx:m - dx + w]


def within_var(fid, v):
    """Pooled within-field variance parts: (sum of squares around each field mean, degrees of freedom = sum of n-1).
    Every field fragment counts, so shifting the labels cannot hide mixed pixels by creating tiny fragments."""
    ok = (fid > 0) & np.isfinite(v)
    if not ok.any():
        return 0.0, 0
    u, inv_ = np.unique(fid[ok], return_inverse=True)
    n = np.bincount(inv_)
    s, s2 = np.bincount(inv_, v[ok]), np.bincount(inv_, v[ok] ** 2)
    ss = np.maximum(s2 - s ** 2 / n, 0)
    return float(ss.sum()), int((n - 1).sum())


def alignment(chip_arrays):
    """chip_arrays: iterable of (field_ids, ndvi). -> DataFrame of pooled within-field variance by (dy, dx), relative to (0, 0)."""
    acc = {(dy, dx): [0.0, 0] for dy in range(-SHIFT, SHIFT + 1) for dx in range(-SHIFT, SHIFT + 1)}
    for fid, v in chip_arrays:
        for (dy, dx), a in acc.items():
            sv, n = within_var(shift(fid, dy, dx), v)
            a[0] += sv
            a[1] += n
    grid = pd.DataFrame({dy: {dx: acc[(dy, dx)][0] / acc[(dy, dx)][1] for dx in range(-SHIFT, SHIFT + 1)}
                         for dy in range(-SHIFT, SHIFT + 1)})            # columns dy, rows dx
    return grid / grid.loc[0, 0]


def date_guess(snap_ndvi, s2, sample=20000, seed=0):
    """Spearman correlation between snapshot NDVI and each NDVI window. snap_ndvi, s2 are indexed by pid."""
    pid = snap_ndvi.dropna().index.intersection(s2.index)
    pid = pid.to_series().sample(min(sample, len(pid)), random_state=seed).index
    out = {c: snap_ndvi[pid].corr(s2.loc[pid, c], method="spearman") for c in features_cols(s2)}
    return pd.Series(out)


def features_cols(df):
    return [c for c in df.columns if c.startswith("NDVI_")]


def main(data="data"):
    env_fix.check()
    data = Path(data)
    pix = pd.read_csv(data / "phase1" / "pixels.csv", dtype={"chip": str})
    pix["pid"] = pix.index
    files = inv.find_chips(data / "train_labels")
    rows, align_in, bands = [], [], None
    for chip, g in pix.groupby("chip"):
        b, arr = read_snapshot(data, chip)
        bands = bands or b
        if b != bands:
            raise SystemExit(f"Chip {chip} has bands {b}, expected {bands}")
        v = ndvi(arr, bands)
        with rasterio.open(files[chip][0]) as src:
            f = inv.as_int(src.read(1), "field_ids")
        if f.shape != v.shape:
            raise SystemExit(f"{chip}: snapshot and labels differ in shape {v.shape} vs {f.shape}")
        align_in.append((f, v))
        d = pd.DataFrame(arr[:, g.row.to_numpy(), g.col.to_numpy()].T, columns=bands, index=g.pid)
        d["NDVI"] = v[g.row.to_numpy(), g.col.to_numpy()]
        rows.append(d)
    snap = pd.concat(rows).sort_index()
    if len(snap) != len(pix):
        raise SystemExit("Snapshot table does not cover all pixels")
    print(f"snapshot bands: {bands}; value range {np.nanmin(snap[bands].to_numpy()):.0f} to {np.nanmax(snap[bands].to_numpy()):.0f}")
    (data / "phase3").mkdir(parents=True, exist_ok=True)
    snap.to_csv(data / "phase3" / "snapshot.csv")
    # A: learnable?
    df = pix.join(snap, on="pid")
    df = df[df.split == "train"].reset_index(drop=True)
    print("\nA. Snapshot classification, spatial-block CV, train fields only")
    for cs, d in [("5 classes", df), ("no fallow", df[df.label != p3.FALLOW].reset_index(drop=True))]:
        for level in ["pixel", "field"]:
            r = p3.evaluate(d, level, 0, 100, X=d[bands].to_numpy())
            print(f"  {cs:10s} {level:5s} n={r['n']:6d} macro-F1 {r['f1']:.3f}  bal-acc {r['bal']:.3f}  OA {r['oa']:.3f} (always-biggest-class {r['base']:.3f})")
    # B: aligned?
    g = alignment(align_in)
    print("\nB. Within-field NDVI variance when labels are shifted (relative to no shift; rows dx, columns dy)")
    print(g.round(3).to_string())
    dx, dy = np.unravel_index(np.nanargmin(g.to_numpy()), g.shape)
    print(f"minimum at dx={g.index[dx]}, dy={g.columns[dy]} -> {'aligned' if (g.index[dx], g.columns[dy]) == (0, 0) else 'SHIFTED, labels may be offset'}")
    # C: which date?
    p = data / "phase2" / f"{study.SEASON_START}_s2" / "features.csv"
    if p.exists():
        s2 = pd.read_csv(p).set_index("pid")
        c = date_guess(snap.NDVI, s2)
        wins = study.bins()
        print("\nC. Snapshot NDVI vs each 12-day NDVI window (Spearman; highest = likely acquisition window)")
        for (a, _), (col, val) in zip(wins, c.items()):
            print(f"  {a}  {val:+.3f}{'  <== max' if col == c.idxmax() else ''}")
    else:
        print("\nC. skipped (full Sentinel-2 table not finished yet). Re-run this script later.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
