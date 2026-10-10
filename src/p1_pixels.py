"""Phase 1a: labelled-pixel table for the study block, with a leak-proof train/test split.

Usage:  python p1_pixels.py [data_dir]        (needs data/inventory from p0_inventory.py)
Writes data/phase1/pixels.csv and fields_study.csv.
Split rule: whole FIELDS, and whole ~11 km spatial blocks, go to either train or test. Never both.
"""
import sys
from pathlib import Path

import env_fix  # must come before rasterio
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from rasterio.warp import transform as warp_transform
from sklearn.model_selection import StratifiedGroupKFold

import p0_inventory as inv
import study

MIN_TEST_FIELDS = 15  # my rule of thumb: fewer test fields per class gives very wide intervals


def build(data_dir="data", out_dir=None, cells=None, classes=None, seed=42, test_frac=0.2, block_deg=0.1):
    env_fix.check()
    cells = cells or study.BLOCK_CELLS
    classes = classes or study.CLASSES
    data_dir = Path(data_dir)
    out = Path(out_dir) if out_dir else data_dir / "phase1"
    out.mkdir(parents=True, exist_ok=True)
    chips = pd.read_csv(data_dir / "inventory" / "chips.csv", dtype={"chip": str, "cell": str})
    fields = pd.read_csv(data_dir / "inventory" / "fields.csv")
    sel = chips[chips.cell.isin(cells)]
    if sel.empty:
        raise SystemExit(f"No chips in cells {cells}")
    files = inv.find_chips(data_dir / "train_labels")
    miss = set(sel.chip) - set(files)
    if miss:
        raise SystemExit(f"Chips without label files: {sorted(miss)[:5]}")
    parts = []
    for cid in sel.chip:
        f, l = files[cid]
        with rasterio.open(f) as a, rasterio.open(l) as b:
            if a.shape != b.shape or a.transform != b.transform or a.crs != b.crs:
                raise SystemExit(f"{cid}: rasters not aligned")
            fid, lb, tr, crs = a.read(1), b.read(1), a.transform, a.crs
        fid, lb = inv.as_int(fid, "field_ids"), inv.as_int(lb, "raster_labels")
        r, c = np.nonzero((fid > 0) & (lb > 0))
        if not len(r):
            continue
        xs, ys = xy(tr, r, c, offset="center")
        lon, lat = warp_transform(crs, "EPSG:4326", list(np.asarray(xs)), list(np.asarray(ys)))
        parts.append(pd.DataFrame(dict(chip=cid, row=r, col=c, lon=lon, lat=lat, field=fid[r, c], label=lb[r, c])))
    if not parts:
        raise SystemExit("No labelled pixels in the selected chips")
    pix = pd.concat(parts, ignore_index=True)
    n0 = len(pix)
    ok_fields = set(fields.field[fields.label.isin(classes)])   # drops conflicted fields and excluded classes
    pix = pix[pix.field.isin(ok_fields)].reset_index(drop=True)
    print(f"pixels in block: {n0}, kept (study classes, consistent fields): {len(pix)}, dropped: {n0 - len(pix)}")
    lab = fields.set_index("field").label
    if (pix.label != pix.field.map(lab)).any():
        raise SystemExit("Pixel labels disagree with fields.csv. Re-run p0_inventory.py")
    ft = pix.groupby("field").agg(label=("label", "first"), npix=("label", "size"),
                                  lon=("lon", "mean"), lat=("lat", "mean"))
    ft["block"] = (np.floor(ft.lon / block_deg).astype(int).astype(str) + "_"
                   + np.floor(ft.lat / block_deg).astype(int).astype(str))
    ft["crop"] = ft.label.map(classes)
    sg = StratifiedGroupKFold(n_splits=round(1 / test_frac), shuffle=True, random_state=seed)
    _, te = next(sg.split(ft, ft.label, ft.block))
    ft["split"] = "train"
    ft.iloc[te, ft.columns.get_loc("split")] = "test"
    tr_b, te_b = set(ft.block[ft.split == "train"]), set(ft.block[ft.split == "test"])
    if tr_b & te_b:
        raise SystemExit("Spatial block in both train and test (bug)")
    cnt = ft.pivot_table(index="crop", columns="split", values="npix", aggfunc="size", fill_value=0)
    if set(cnt.columns) != {"train", "test"} or (cnt == 0).any().any():
        raise SystemExit(f"A class is missing from train or test:\n{cnt}")
    pix["split"] = pix.field.map(ft.split)
    pix["block"] = pix.field.map(ft.block)
    pix["crop"] = pix.label.map(classes)
    pix.to_csv(out / "pixels.csv", index=False)
    ft.reset_index().to_csv(out / "fields_study.csv", index=False)
    print(f"\nfields per class and split (blocks: {len(tr_b)} train, {len(te_b)} test):\n{cnt.to_string()}")
    px = pix.pivot_table(index="crop", columns="split", values="field", aggfunc="size", fill_value=0)
    print(f"\npixels per class and split:\n{px.to_string()}")
    small = cnt.index[cnt["test"] < MIN_TEST_FIELDS].tolist()
    if small:
        print(f"\nWARNING: fewer than {MIN_TEST_FIELDS} test fields for {small}: their accuracy will have wide intervals.")
    return pix, ft


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "data")
