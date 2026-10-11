"""Run: python tests/selftest_phase6.py -> 'SELFTEST OK'. Synthetic chips; checks object voting and the class-weight tuning."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

import p5_chip_features as cf
import p6_objects as ob
import p6_weights as wt

rng = np.random.default_rng(0)
H = W = 48
means = {1: -12.0, 2: -15.0, 6: -9.0}
stacks, rows = {}, []
field_of = (np.arange(H)[:, None] // 12) * 4 + (np.arange(W)[None, :] // 12)      # 16 blocks of 12 x 12 pixels per chip
for ci, chip in enumerate(("a", "b")):
    lab = np.array([1, 2, 6, 1, 2, 6, 1, 2, 1, 6, 2, 1, 2, 1, 6, 2])[field_of]
    base = np.vectorize(means.get)(lab)
    st = (base[None] + rng.normal(0, 4.0, (80, H, W))).astype("float32")
    stacks[chip] = st
    for r in range(H):
        for c in range(W):
            f = ci * 16 + field_of[r, c]
            rows.append(dict(chip=chip, row=r, col=c, field=f, label=int(lab[r, c]), block=f"b{f}", split="train"))
pix = pd.DataFrame(rows)
parts = [pd.DataFrame(np.stack([stacks[ch][s * 40:(s + 1) * 40, r, k] for ch, r, k in zip(pix.chip, pix.row, pix.col)]), columns=cf.NAMES) for s in (0, 1)]
meta = pix[["field", "label", "block", "split"]]
res = ob.run(parts, meta, pix, lambda chip: stacks[chip], "3", leaf=5, trees=30, sizes=(144,), compactness=(1.0,))
assert {"pixel", "field", "slic144_c1.0"} <= set(res)
assert res["field"]["f1"] > res["pixel"]["f1"] and res["slic144_c1.0"]["f1"] > res["pixel"]["f1"], {k: v["f1"] for k, v in res.items()}

# class weights: probabilities biased against a minority class are repaired, and the held-out gain is positive
n = 3000
y = rng.choice([1, 2, 6], n, p=[0.7, 0.2, 0.1])
logit = np.eye(3)[np.searchsorted([1, 2, 6], y)] * 1.2 + rng.normal(0, 1, (n, 3)) + np.log([0.7, 0.2, 0.1])
P = np.exp(logit) / np.exp(logit).sum(1, keepdims=True)
g = rng.integers(0, 20, n)
w, _ = wt.tune(P, y, np.array([1, 2, 6]))
gain, _ = wt.holdout_gain(P, y, g, np.array([1, 2, 6]), seeds=2)
assert w[1] > 1 and w[2] > 1 and gain > 0
print("SELFTEST OK")
