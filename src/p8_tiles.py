"""Phase 8a: Web Mercator XYZ PNG tiles from the per-chip map rasters.
python src/p8_tiles.py [--set _near|_far] [--masked | --conf] [--zmin 8] [--zmax 14] [--out docs]
Reads data/phase7/map<set>/<chip>_class[_masked].tif (codes 1 Wheat, 2 Mustard/Lentil, 6 Sugarcane, 0 none) or <chip>_conf.tif (largest
class probability x 255, 0 none) and writes docs/tiles/<name>/<z>/<x>/<y>.png (transparent where 0) plus the layer entry in docs/data/meta.json.
Zoom 14 is the native resolution here (about 8.5 m per pixel at 26 degrees north); the web page overzooms beyond it with nearest-neighbour."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import env_fix  # noqa: F401  (before rasterio)
import mercantile
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds

CLASSES = {1: ("Wheat", (0, 158, 115)), 2: ("Mustard and lentil", (240, 228, 66)), 6: ("Sugarcane", (213, 94, 0))}  # Okabe-Ito
RAMP = [(237, 248, 177), (65, 182, 196), (37, 52, 148)]  # ColourBrewer YlGnBu: low to high confidence
CONF_LOW = 85  # three classes: the largest probability is at least 1/3 (85 of 255)


def palette(kind):
    pal = np.zeros((256, 3), np.uint8)
    if kind == "class":
        for k, (_, c) in CLASSES.items():
            pal[k] = c
    else:
        t = np.clip((np.arange(256) - CONF_LOW) / (255 - CONF_LOW), 0, 1) * (len(RAMP) - 1)
        for i in range(256):
            lo = min(int(t[i]), len(RAMP) - 2)
            f = t[i] - lo
            pal[i] = [round(RAMP[lo][j] * (1 - f) + RAMP[lo + 1][j] * f) for j in range(3)]
    return pal


def render(paths, tile, resampling):
    t = from_bounds(*mercantile.xy_bounds(tile), 256, 256)
    out = np.zeros((256, 256), np.uint8)
    for p in paths:
        buf = np.zeros((256, 256), np.uint8)
        with rasterio.open(p) as s:
            reproject(rasterio.band(s, 1), buf, dst_transform=t, dst_crs="EPSG:3857", resampling=resampling)
        out = np.where(buf > 0, buf, out)
    return out


def save_png(idx, path, pal):
    im = Image.fromarray(idx, "P")
    im.putpalette(pal.flatten().tolist())
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, transparency=0, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_", default="")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--masked", action="store_true", help="the WorldCover-masked class map (run p7_map.py --mask first)")
    g.add_argument("--conf", action="store_true", help="the confidence map instead of the class map")
    ap.add_argument("--zmin", type=int, default=8)
    ap.add_argument("--zmax", type=int, default=14)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="docs")
    a = ap.parse_args()
    kind = "conf" if a.conf else "class"
    name = kind + a.set_ + ("_masked" if a.masked else "")
    pattern = "*_conf.tif" if a.conf else "*_class_masked.tif" if a.masked else "*_class.tif"
    files = sorted(Path(a.data, "phase7", f"map{a.set_}").glob(pattern))
    if not files:
        raise SystemExit(f"No {pattern} files found. Run p7_map.py {'--mask ' if a.masked else ''}{'--set ' + a.set_ if a.set_ else ''} first.")
    boxes = []
    for p in files:
        with rasterio.open(p) as s:
            boxes.append(transform_bounds(s.crs, "EPSG:4326", *s.bounds))
    west, south, east, north = min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)
    tdir, pal = Path(a.out, "tiles", name), palette(kind)
    n = size = 0
    for z in range(a.zmin, a.zmax + 1):
        todo = defaultdict(list)
        for p, b in zip(files, boxes):
            for t in mercantile.tiles(*b, z):
                todo[t].append(p)
        rs = Resampling.nearest if z >= 13 else Resampling.mode if kind == "class" else Resampling.average
        for t, paths in todo.items():
            idx = render(paths, t, rs)
            if idx.any():
                f = tdir / str(z) / str(t.x) / f"{t.y}.png"
                save_png(idx, f, pal)
                n += 1
                size += f.stat().st_size
        print(f"z{z}: {len(todo)} candidate tiles, {n} written so far, {size / 1e6:.1f} MB")
    mp = Path(a.out, "data", "meta.json")
    mp.parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads(mp.read_text()) if mp.exists() else {}
    meta["classes"] = {str(k): dict(name=v[0], color="#%02x%02x%02x" % v[1]) for k, v in CLASSES.items()}
    meta["confidence_ramp"] = dict(low=round(CONF_LOW / 255, 2), colors=["#%02x%02x%02x" % c for c in RAMP])
    meta.setdefault("tiles", {})[name] = dict(bounds=[west, south, east, north], minzoom=a.zmin, maxzoom=a.zmax, tiles=n, mb=round(size / 1e6, 1))
    mp.write_text(json.dumps(meta, indent=1))
    print(f"wrote {n} tiles ({size / 1e6:.1f} MB) to {tdir} and the entry '{name}' in {mp}")


if __name__ == "__main__":
    main()
