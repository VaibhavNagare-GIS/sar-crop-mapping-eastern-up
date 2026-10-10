"""Phase 0b: inventory of the AgriFieldNet train labels.

Usage:  python p0_inventory.py [data_dir] [out_dir] [cell_deg] [min_fields]
        defaults: data  data/inventory  0.5  30
Writes fields.csv, chips.csv, cells.csv, classes.csv and prints the gate summary.
Every inconsistency is printed or raises; nothing is dropped without a message.
"""
import json
import sys
from pathlib import Path

import env_fix  # must come before rasterio
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds

# Verified against the dataset Documentation.pdf (0 = pixel not labelled).
CLASSES = {1: "Wheat", 2: "Mustard", 3: "Lentil", 4: "No crop/Fallow", 5: "Green pea",
           6: "Sugarcane", 8: "Garlic", 9: "Maize", 13: "Gram", 14: "Coriander",
           15: "Potato", 16: "Bersem", 36: "Rice"}


def find_chips(root):
    """Flat layout: <prefix>_<id>.tif (crop labels) and <prefix>_<id>_field_ids.tif (field ids)."""
    root = Path(root)
    chips = {}
    for p in sorted(root.glob("*.tif")):
        is_f = p.name.endswith("_field_ids.tif")
        stem = p.name[: -len("_field_ids.tif")] if is_f else p.name[: -len(".tif")]
        cid = stem.rsplit("_", 1)[-1]
        slot = chips.setdefault(cid, {})
        if ("f" if is_f else "l") in slot:
            raise SystemExit(f"Duplicate file for chip {cid}: {p}")
        slot["f" if is_f else "l"] = p
    if not chips:
        raise SystemExit(f"No .tif files under {root}. Run p0_download.py first.")
    for cid, slot in chips.items():
        if set(slot) != {"f", "l"}:
            raise SystemExit(f"Chip {cid} is missing its {'field_ids' if 'f' not in slot else 'label'} file")
    return {cid: (slot["f"], slot["l"]) for cid, slot in chips.items()}


def as_int(a, name):
    if np.issubdtype(a.dtype, np.floating):
        if not np.all(np.isfinite(a)) or not np.all(a == np.floor(a)):
            raise SystemExit(f"{name} has NaN or non-integer values")
    return a.astype(np.int64)


def read_chip(cid, f, lab):
    with rasterio.open(f) as a, rasterio.open(lab) as b:
        if a.count != 1 or b.count != 1:
            raise SystemExit(f"{cid}: expected single-band rasters")
        if a.shape != b.shape or a.transform != b.transform or a.crs != b.crs:
            raise SystemExit(f"{cid}: field_ids and raster_labels are not aligned")
        if a.crs is None:
            raise SystemExit(f"{cid}: no CRS")
        fid, lb = as_int(a.read(1), "field_ids"), as_int(b.read(1), "raster_labels")
        w, s, e, n = transform_bounds(a.crs, "EPSG:4326", *a.bounds)
        px_m2 = abs(a.transform.a * a.transform.e) if a.crs.is_projected else np.nan
        nodata = (a.nodata, b.nodata)
    return fid, lb, (w, s, e, n), px_m2, nodata


def stac_dates(src_root, wanted):
    """chip id -> 'YYYY-MM-DD'. Scans every .json under source/, finds the chip id among the
    path/id tokens (matched against known label chip ids), reads properties.datetime."""
    import re
    wanted = set(wanted)
    out = {c: None for c in wanted}
    src_root = Path(src_root)
    if not src_root.exists():
        print(f"WARNING: {src_root} not found, no Sentinel-2 dates")
        return out
    unmatched = bad = 0
    for p in src_root.rglob("*.json"):
        try:
            j = json.loads(p.read_text())
        except ValueError:
            bad += 1
            continue
        props = j.get("properties", {}) if isinstance(j, dict) else {}
        d = props.get("datetime") or props.get("start_datetime")
        if not d:
            continue
        tokens = set(re.split(r"[_./\\-]", str(p.relative_to(src_root)) + "_" + str(j.get("id", ""))))
        hit = tokens & wanted
        if len(hit) != 1:
            unmatched += 1
            continue
        out[hit.pop()] = str(d)[:10]
    if bad or unmatched:
        print(f"WARNING: {bad} unreadable json, {unmatched} json with a date but no unique chip id match")
    return out


def run(data_dir="data", out_dir=None, cell_deg=0.5, min_fields=30):
    env_fix.check()
    data_dir = Path(data_dir)
    out_dir = Path(out_dir) if out_dir else data_dir / "inventory"
    out_dir.mkdir(parents=True, exist_ok=True)
    chips = find_chips(data_dir / "train_labels")  # flat folder of tifs
    rows, chip_rows, nodatas, px_areas = [], [], set(), set()
    for cid, (f, lab) in chips.items():
        fid, lb, bb, px_m2, nd = read_chip(cid, f, lab)
        nodatas.add(nd)
        px_areas.add(px_m2)
        has_f, has_l = fid > 0, lb > 0
        no_label = int(np.sum(has_f & ~has_l))   # field id but label 0
        no_field = int(np.sum(~has_f & has_l))   # label but field id 0
        ok = has_f & has_l
        unknown = set(np.unique(lb[ok])) - set(CLASSES)
        if unknown:
            raise SystemExit(f"{cid}: unknown class codes {sorted(unknown)}")
        if ok.any():
            pairs, n = np.unique(np.stack([fid[ok], lb[ok]]), axis=1, return_counts=True)
            rows += [(cid, int(a), int(b), int(c), px_m2) for (a, b), c in zip(pairs.T, n)]
        chip_rows.append(dict(chip=cid, west=bb[0], south=bb[1], east=bb[2], north=bb[3],
                              lon=(bb[0] + bb[2]) / 2, lat=(bb[1] + bb[3]) / 2,
                              n_fields=len(np.unique(fid[ok])), n_labeled_px=int(ok.sum()),
                              px_field_no_label=no_label, px_label_no_field=no_field))
    if not rows:
        raise SystemExit("No labelled pixels found in any chip")
    pix = pd.DataFrame(rows, columns=["chip", "field", "label", "npix", "px_m2"])
    chips_df = pd.DataFrame(chip_rows)

    # a field must carry one label everywhere (it can span several chips)
    nl = pix.groupby("field")["label"].nunique()
    conflict = nl[nl > 1].index
    if len(conflict):
        print(f"WARNING: {len(conflict)} fields have conflicting labels and are DROPPED: {list(conflict[:10])}")
        pix[pix.field.isin(conflict)].to_csv(out_dir / "conflicting_fields.csv", index=False)
        pix = pix[~pix.field.isin(conflict)]

    pix["area_ha"] = pix.npix * pix.px_m2 / 1e4
    fields = (pix.groupby("field")
              .agg(label=("label", "first"), npix=("npix", "sum"), n_chips=("chip", "nunique"),
                   area_ha=("area_ha", "sum")).reset_index())
    fields["crop"] = fields.label.map(CLASSES)

    dates = stac_dates(data_dir / "source", chips_df.chip)
    chips_df["s2_date"] = chips_df.chip.map(dates)
    chips_df["cell"] = (np.floor(chips_df.lon / cell_deg).astype(int).astype(str) + "_"
                        + np.floor(chips_df.lat / cell_deg).astype(int).astype(str))

    # per-cell class counts (a field that touches two cells counts in both: tiny border effect)
    fc = pix.merge(chips_df[["chip", "cell"]], on="chip").merge(fields[["field", "crop"]], on="field")
    fc = fc.drop_duplicates(["cell", "field"])
    cells = fc.pivot_table(index="cell", columns="crop", values="field", aggfunc="nunique", fill_value=0)
    cells["n_fields"] = cells.sum(axis=1)
    cells["classes_ok"] = (cells.drop(columns="n_fields") >= min_fields).sum(axis=1)
    g = chips_df.groupby("cell")
    cells["n_chips"] = g.chip.count()
    for c, fn in [("west", "min"), ("south", "min"), ("east", "max"), ("north", "max")]:
        cells[c] = g[c].agg(fn)
    cells["s2_min"], cells["s2_max"] = g.s2_date.min(), g.s2_date.max()
    cells = cells.sort_values(["classes_ok", "n_fields"], ascending=False)

    fields.to_csv(out_dir / "fields.csv", index=False)
    chips_df.to_csv(out_dir / "chips.csv", index=False)
    cells.to_csv(out_dir / "cells.csv")
    cls = fields.groupby("crop").agg(fields=("field", "count"), pixels=("npix", "sum"),
                                     median_px=("npix", "median"), ha=("area_ha", "sum")).sort_values("fields", ascending=False)
    cls.to_csv(out_dir / "classes.csv")

    print(f"\nchips: {len(chips_df)}   fields: {len(fields)}   labelled pixels: {int(fields.npix.sum())}")
    print(f"fields spanning >1 chip: {int((fields.n_chips > 1).sum())}  (split by FIELD id, never by chip)")
    print(f"pixel area m2 values: {px_areas}   nodata (field_ids, labels): {nodatas}")
    print(f"pixels with field id but no label: {int(chips_df.px_field_no_label.sum())}, "
          f"label but no field id: {int(chips_df.px_label_no_field.sum())} (both excluded)")
    print("\nclass table:\n", cls.to_string())
    q = fields.npix.quantile([.1, .5, .9]).round(1).tolist()
    print(f"\nfield size in pixels, p10/p50/p90: {q}   share under 20 px (0.2 ha at 10 m): "
          f"{(fields.npix < 20).mean():.0%}")
    d = chips_df.s2_date.dropna()
    print(f"\nS2 dates found for {len(d)} of {len(chips_df)} chips")
    if len(d):
        print("S2 date min/max:", d.min(), d.max())
        print(pd.to_datetime(d).dt.to_period("M").value_counts().sort_index().to_string())
    else:
        print("NO DATES FOUND in the source files; the date parser needs to be adapted to the file layout.")
    show = ["n_chips", "n_fields", "classes_ok", "west", "south", "east", "north", "s2_min", "s2_max"]
    print(f"\ntop cells ({cell_deg} deg grid), classes_ok = classes with >= {min_fields} fields:\n",
          cells[show].head(10).round(3).to_string())
    return fields, chips_df, cells


if __name__ == "__main__":
    a = sys.argv[1:]
    run(a[0] if a else "data", a[1] if len(a) > 1 else None,
        float(a[2]) if len(a) > 2 else 0.5, int(a[3]) if len(a) > 3 else 30)
