"""Run: python tests/selftest_phase8.py -> 'SELFTEST OK'. Tiles and field GeoJSON on synthetic chips, including a fallow field the model has no class for."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

import env_fix  # noqa: F401
import rasterio
from PIL import Image
from rasterio.transform import from_origin

import p8_geojson as gj
import p8_tiles as tl

tmp = Path(tempfile.mkdtemp())
for d in ("train_labels", "phase7/map", "phase1"):
    (tmp / "data" / d).mkdir(parents=True)
rows = []
for k, (cid, x0) in enumerate((("aaa111", 500000), ("bbb222", 502560))):
    prof = dict(driver="GTiff", height=256, width=256, count=1, crs="EPSG:32644", transform=from_origin(x0, 2900000, 10, 10))
    fid, lab, cls, conf = (np.zeros((256, 256), np.int32), np.zeros((256, 256), np.uint8), np.zeros((256, 256), np.uint8), np.zeros((256, 256), np.uint8))
    for j, (r0, c0, label, pred) in enumerate(((20, 20, 1, 1), (100, 100, 3, 1), (180, 40, 6, 6), (20, 150, 4, 1))):  # lentil predicted wheat; field 4 is fallow
        f = 1000 * (k + 1) + j + 1
        fid[r0:r0 + 40, c0:c0 + 60], lab[r0:r0 + 40, c0:c0 + 60], cls[r0:r0 + 40, c0:c0 + 60], conf[r0:r0 + 40, c0:c0 + 60] = f, label, pred, 200
        rows += [dict(chip=cid, row=r, col=c0, field=f, label=label, crop={1: "Wheat", 3: "Lentil", 4: "No crop/Fallow", 6: "Sugarcane"}[label],
                      split="test" if k else "train", block="b") for r in range(r0, r0 + 40, 5)]
    for name, arr, dt in ((f"pfx_{cid}", lab, "uint8"), (f"pfx_{cid}_field_ids", fid, "int32")):
        with rasterio.open(tmp / "data" / "train_labels" / f"{name}.tif", "w", **{**prof, "dtype": dt}) as d:
            d.write(arr.astype(dt), 1)
    for name, arr in ((f"{cid}_class", cls), (f"{cid}_conf", conf), (f"{cid}_class_masked", np.where(cls == 6, 0, cls).astype(np.uint8))):
        with rasterio.open(tmp / "data" / "phase7" / "map" / f"{name}.tif", "w", **{**prof, "dtype": "uint8", "nodata": 0}) as d:
            d.write(arr, 1)
pd.DataFrame(rows).to_csv(tmp / "data" / "phase1" / "pixels.csv", index=False)

base = ["--data", str(tmp / "data"), "--out", str(tmp / "docs")]
for extra in ([], ["--masked"], ["--conf"]):
    sys.argv = ["p8_tiles.py", *base, "--zmin", "9", "--zmax", "14", *extra]
    tl.main()
meta = json.loads((tmp / "docs" / "data" / "meta.json").read_text())
assert set(meta["tiles"]) == {"class", "class_masked", "conf"} and meta["tiles"]["class"]["tiles"] > meta["tiles"]["class_masked"]["tiles"] - 1
png = next((tmp / "docs" / "tiles" / "class" / "14").rglob("*.png"))
im = Image.open(png)
assert im.mode == "P" and im.info["transparency"] == 0 and set(np.unique(np.array(im))) <= {0, 1, 2, 6}
cim = Image.open(next((tmp / "docs" / "tiles" / "conf" / "14").rglob("*.png")))
assert set(np.unique(np.array(cim))) <= {0, 200}

sys.argv = ["p8_geojson.py", *base]
gj.main()
for split, first in (("train", 1001), ("test", 2001)):
    fs = {f["properties"]["id"]: f["properties"] for f in json.loads((tmp / "docs" / "data" / f"fields_{split}.geojson").read_text())["features"]}
    assert len(fs) == 4 and fs[first]["ok"] == 1 and fs[first]["crop"] == "Wheat" and fs[first]["conf"] == 0.78
    assert fs[first + 1]["crop"] == "Lentil" and fs[first + 1]["target"] == "Mustard and lentil" and fs[first + 1]["pred"] == "Wheat" and fs[first + 1]["ok"] == 0
    assert fs[first + 3]["crop"] == "No crop/Fallow" and fs[first + 3]["target"] is None and fs[first + 3]["ok"] is None and fs[first + 3]["pred"] == "Wheat"
print("SELFTEST OK")
