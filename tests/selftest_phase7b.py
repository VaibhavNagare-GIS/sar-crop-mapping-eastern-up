"""Run: python tests/selftest_phase7b.py -> 'SELFTEST OK'. Final model, map files and evaluation on synthetic data."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

import p5_chip_features as cf
import p7_eval as ev
import p7_map as mp

rng = np.random.default_rng(0)
H = W = 32
# model and maps
means = {1: -12.0, 2: -15.0, 6: -9.0}
lab = np.kron(np.array([[1, 2], [6, 1]]), np.ones((16, 16), int))
stack = (np.vectorize(means.get)(lab)[None] + rng.normal(0, 3.0, (80, H, W))).astype("float32")
rows = [dict(chip="a", row=r, col=c, field=(r // 16) * 2 + c // 16, label=int(lab[r, c]), block=f"b{(r // 16) * 2 + c // 16}", split="train") for r in range(H) for c in range(W)]
pix = pd.DataFrame(rows)
parts = [pd.DataFrame(np.stack([stack[s * 40:(s + 1) * 40, r, c] for r, c in zip(pix.row, pix.col)]), columns=cf.NAMES) for s in (0, 1)]
meta = pix[["field", "label", "block", "split"]]
model, n = mp.train_final(parts, meta, 3, 30, 0)
raw, sm, valid = mp.predict_maps(model, stack)
assert n == H * W and raw.shape == (H, W, 3) and np.allclose(sm.sum(-1), 1, atol=1e-6)
assert valid.all()
cls, conf = mp.class_conf(sm, valid)
assert set(np.unique(cls)) <= {1, 2, 6} and cls.dtype == np.uint8 and conf.max() <= 255 and (cls == lab).mean() > 0.9
# pixels without any data: no crash, class 0, and the neighbours are not polluted
holes = stack.copy()
holes[:, :4, :4] = np.nan
raw2, sm2, v2 = mp.predict_maps(model, holes)
cls2, conf2 = mp.class_conf(sm2, v2)
assert (~v2[:4, :4]).all() and v2[4:, 4:].all() and (cls2[:4, :4] == 0).all() and (conf2[:4, :4] == 0).all() and np.isfinite(sm2).all()
assert (cls2[8:, 8:] == cls[8:, 8:]).mean() > 0.95
# fused features: Sentinel-1 plus NDVI
ndvi = (0.3 + 0.1 * np.vectorize({1: 1, 2: 2, 6: 3}.get)(lab))[None] + rng.normal(0, 0.05, (20, H, W))
Xf, okf = mp.dc.fused_features(stack, ndvi.astype("float32"))
assert Xf.shape[1] == mp.dc.dense_features(stack).shape[1] + 20 and okf.all()
with tempfile.TemporaryDirectory() as d:
    tif = Path(d) / "label.tif"
    with rasterio.open(tif, "w", driver="GTiff", height=H, width=W, count=1, dtype="uint8", crs="EPSG:32644", transform=from_origin(500000, 3000000, 10, 10)) as f:
        f.write(np.ones((1, H, W), "uint8"))
    mp.write_tif(Path(d) / "c.tif", cls, tif)
    with rasterio.open(Path(d) / "c.tif") as s:
        assert s.crs.to_epsg() == 32644 and s.transform == from_origin(500000, 3000000, 10, 10) and (s.read(1) == cls).all() and s.nodata == 0
    cfg = dict(leaf=20, trees=500, seed=42, smooth=5, classes="3", features="x")
    mp.record_config(Path(d) / "cfg.json", cfg)
    mp.record_config(Path(d) / "cfg.json", cfg)                       # same settings: fine
    try:
        mp.record_config(Path(d) / "cfg.json", dict(cfg, leaf=5))
        raise AssertionError("changed settings must be refused")
    except SystemExit:
        pass

# evaluation: macro scores only over classes that occur
n = 400
y = rng.choice([1, 2], n)
f = rng.integers(0, 40, n)
P = np.zeros((n, 3))
P[np.arange(n), np.searchsorted([1, 2, 6], y)] = 1                    # a perfect map of a region without Sugarcane
df = pd.DataFrame(np.hstack([P, P]), columns=[f"{p}{k}" for p in ("raw", "sm") for k in (1, 2, 6)])
df["label"], df["field"] = y, f
res, cms = ev.evaluate(df, boot=50)
assert res["classes_present"] == ["Wheat", "Mustard+Lentil"] and abs(res["pixel_smoothed"]["f1"] - 1) < 1e-9 and cms["pixel_raw"].shape == (3, 3)
assert res["pixel_raw"]["ci"]["f1"][0] <= 1.0 and res["n_fields"] == 40
# cropland mask and its coverage table
wc = np.full((4, 4), 40, dtype=np.uint8)
wc[0, :] = 50
c4 = np.arange(16, dtype=np.uint8).reshape(4, 4) % 3 + 1
assert (mp.apply_mask(c4, wc)[0] == 0).all() and (mp.apply_mask(c4, wc)[1:] == c4[1:]).all()
import p7_mask_check as mc
pxm = pd.DataFrame(dict(chip="a", row=[0, 1, 2, 3], col=[0, 1, 2, 3], crop=["Wheat", "Wheat", "Sugarcane", "Sugarcane"]))
tab, cnt = mc.coverage(pxm, lambda chip: wc)
assert tab.loc["Wheat", "Cropland"] == 50.0 and tab.loc["Wheat", "Built-up"] == 50.0 and tab.loc["Sugarcane", "Cropland"] == 100.0 and cnt["Wheat"] == 2
print("SELFTEST OK")
