"""Phase 8b: field polygons with crop, split, predicted class and confidence, as GeoJSON files for the web map.
python src/p8_geojson.py [--set _near|_far] [--out docs]
Study block (no --set): docs/data/fields_train.geojson and fields_test.geojson. With --set: fields_near.geojson or fields_far.geojson.
Every field of pixels.csv is written. pred = majority of the smoothed class pixels inside the field (the field vote that p7_eval.py scores),
conf = mean confidence of the winning class's pixels, target = the modelled class of the surveyed crop (mustard and lentil merged) or null for
crops the model has no class for (fallow and others); ok is 1 or 0 only where both pred and target exist, else null (not scored)."""
import argparse
import json
from pathlib import Path

import env_fix  # noqa: F401  (before rasterio)
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_geom

import p0_inventory as inv

TARGET = {1: 1, 2: 2, 3: 2, 6: 6}  # surveyed crop code -> modelled class
NAMES = {1: "Wheat", 2: "Mustard and lentil", 6: "Sugarcane"}


def chip_fields(fid_path, cls_path, conf_path, wanted):
    """Yield (field, MultiPolygon coordinates in lon/lat, pred class or None, confidence or None) for the wanted field ids of one chip."""
    with rasterio.open(fid_path) as s:
        fid, tr, crs = s.read(1).astype(np.int64), s.transform, s.crs
    with rasterio.open(cls_path) as s:
        cls = s.read(1)
    with rasterio.open(conf_path) as s:
        conf = s.read(1)
    geoms = {}
    for g, v in shapes(fid.astype(np.int32), mask=np.isin(fid, wanted), transform=tr):
        geoms.setdefault(int(v), []).append(transform_geom(crs, "EPSG:4326", g, precision=5)["coordinates"])
    for f, polys in geoms.items():
        m = (fid == f) & (cls > 0)
        pred = conf_v = None
        if m.any():
            vals, cnt = np.unique(cls[m], return_counts=True)
            pred = int(vals[cnt.argmax()])
            conf_v = round(float(conf[m & (cls == pred)].mean()) / 255, 2)
        yield f, polys, pred, conf_v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_", default="")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="docs")
    a = ap.parse_args()
    data = Path(a.data)
    pix = pd.read_csv(data / f"phase1{a.set_}" / "pixels.csv", dtype={"chip": str})
    ft = pix.groupby(["chip", "field"]).agg(crop=("crop", "first"), label=("label", "first"), split=("split", "first"), npix=("chip", "size"))
    files = inv.find_chips(data / "train_labels")
    mapdir = data / "phase7" / f"map{a.set_}"
    feats, skipped = {}, 0
    for chip, g in ft.groupby(level="chip"):
        cp, fp = mapdir / f"{chip}_class.tif", mapdir / f"{chip}_conf.tif"
        if not cp.exists():
            skipped += 1
            continue
        for f, polys, pred, conf in chip_fields(files[chip][0], cp, fp, g.index.get_level_values("field").to_numpy()):
            r = ft.loc[(chip, f)]
            target = TARGET.get(int(r.label))
            feats.setdefault(r.split, []).append({"type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates": polys}, "properties": {
                "id": int(f), "crop": str(r.crop), "target": NAMES.get(target), "pred": NAMES.get(pred), "conf": conf, "px": int(r.npix),
                "ok": None if pred is None or target is None else int(pred == target)}})
    out = Path(a.out, "data")
    out.mkdir(parents=True, exist_ok=True)
    for split, fs in feats.items():
        p = out / (f"fields_{split}.geojson" if split in ("train", "test") else f"fields{a.set_}.geojson")
        p.write_text(json.dumps({"type": "FeatureCollection", "features": fs}, separators=(",", ":")))
        print(f"{p}: {len(fs)} fields, {p.stat().st_size / 1e6:.1f} MB")
    if skipped:
        print(f"{skipped} chips had no map files and are not in the GeoJSON")


if __name__ == "__main__":
    main()
