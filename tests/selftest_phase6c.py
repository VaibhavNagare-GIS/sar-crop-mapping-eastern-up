"""Run: python tests/selftest_phase6c.py -> 'SELFTEST OK'. Map-level cross-validation on synthetic chips."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

import p5_chip_features as cf
import p6_densecv as dc

rng = np.random.default_rng(0)
H = W = 64
means = {1: -12.0, 2: -15.0, 6: -9.0}
layout = np.array([[1, 2], [6, 1]])
stacks, rows = {}, []
for ci in range(12):
    lab = np.kron(layout, np.ones((32, 32), int))
    st = (np.vectorize(means.get)(lab)[None] + rng.normal(0, 9.0, (80, H, W))).astype("float32")
    stacks[f"c{ci}"] = st
    for r in range(H):
        for c in range(W):
            rows.append(dict(chip=f"c{ci}", row=r, col=c, field=ci * 4 + (r // 32) * 2 + c // 32, label=int(lab[r, c]), block=f"b{ci}", split="train"))
pix = pd.DataFrame(rows)
parts = [pd.DataFrame(np.stack([stacks[ch][s * 40:(s + 1) * 40, r, k] for ch, r, k in zip(pix.chip, pix.row, pix.col)]), columns=cf.NAMES) for s in (0, 1)]
meta = pix[["field", "label", "block", "split"]]
msgs = []
res = dc.run(parts, meta, pix, lambda chip: stacks[chip], leaf_pixel=5, leaf_obj=5, trees=20, sizes=(60,), compactness=(0.5,), log=msgs.append)
assert {"pixel", "smooth3", "smooth5", "smooth7", "grid8", "seg60_c0.5", "objrf60_c0.5", "field_true"} <= set(res)
f = {k: round(v["f1"], 3) for k, v in res.items()}
assert all(res[k]["f1"] > res["pixel"]["f1"] for k in ("smooth5", "grid8", "seg60_c0.5", "objrf60_c0.5")), f      # regularising helps when pixels are noisy
assert all(f[k] > 0.5 for k in f) and res["field_true"]["f1"] >= res["pixel"]["f1"] and msgs, f
# a segmentation of pure noise is a plain grid and must be reported as such; fields with contrast must give a clear gain
from scipy.spatial import cKDTree
r2 = np.random.default_rng(1)
n = 96
noise = r2.normal(size=(n, n, 5))
fid = cKDTree(r2.uniform(0, n, (150, 2))).query(np.c_[np.mgrid[0:n, 0:n][0].ravel(), np.mgrid[0:n, 0:n][1].ravel()])[1].reshape(n, n)
fields = r2.normal(size=(150, 5))[fid] + 0.7 * r2.normal(size=(n, n, 5))
g_noise = dc.seg_gain(noise, dc.ob.slic_labels(noise, 64, 0.5).ravel(), 8)
g_fields = dc.seg_gain(fields, dc.ob.slic_labels(fields, 64, 0.5).ravel(), 8)
assert g_noise < 0.05 and g_fields > 0.2, (g_noise, g_fields)
assert any("variance explained" in m for m in msgs)
d = dc.dense_features(stacks["c0"])
assert d.shape == (H * W, 120) and np.isfinite(d).all()
print("SELFTEST OK")
