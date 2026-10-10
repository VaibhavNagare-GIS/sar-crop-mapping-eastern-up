"""Run: python selftest_phase3e.py -> 'SELFTEST OK'. Synthetic data and a fake Earth Engine call only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))   # tests import the code in src/
import contextlib
import io
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

import features
import p2_extra
import p2_extract as pe
import p3_feat as pf
import p3_sweep as sw

rng = np.random.default_rng(0)
labs = [1, 2, 3, 4, 6]
curves = {1: -12 + 4 * np.sin(np.arange(20) / 3), 2: -14 + 4 * np.cos(np.arange(20) / 3), 3: np.full(20, -13.0),
          4: -12 + 2 * np.sin(np.arange(20) / 2), 6: np.linspace(-12, -9, 20)}
pr, rows = [], []
for f in range(60):
    r0, c0 = (f // 10) * 6, (f % 10) * 6                     # 5x5 fields with a 1-pixel gap
    for dr in range(5):
        for dc in range(5):
            rows.append(dict(chip="c1", row=r0 + dr, col=c0 + dc, field=f, label=labs[f % 5], block=f"b{f // 6}", split="train"))
pix = pd.DataFrame(rows)
vv = np.array([curves[l] for l in pix.label]) + rng.normal(0, 2, (len(pix), 20))
vh = vv - 6 + rng.normal(0, 1, (len(pix), 20))
data = pd.DataFrame(np.hstack([vv, vh]), columns=sum(features.band_cols(), []))
meta = pix[["field", "label", "block", "split"]]

# inner mask: a 5x5 field has a 3x3 inner core
inn = pf.inner_mask(pix)
assert (inn.groupby(pix.field).sum() == 9).all()
# a single isolated pixel is never inner
assert not pf.inner_mask(pd.DataFrame(dict(chip=["c"], row=[3], col=[3], field=[1]))).iloc[0]

# linear mean >= dB mean (Jensen), strictly for noisy data
Xb = pf.field_X([data], meta, pf.RUNGS["base"], inn)
Xl = pf.field_X([data], meta, pf.RUNGS["lin"], inn)
assert (Xl[:, 0] > Xb[:, 0]).all()

# angle normalisation does nothing at the reference angle and shifts dB otherwise
ang_ref = pd.Series(pf.REF, index=data.index)
assert np.allclose(pf.field_X([data], meta, pf.RUNGS["norm"], inn, [ang_ref]), pf.field_X([data], meta, pf.RUNGS["std"], inn))
assert not np.allclose(pf.field_X([data], meta, pf.RUNGS["norm"], inn, [ang_ref + 5]), pf.field_X([data], meta, pf.RUNGS["std"], inn))
try:
    pf.field_X([data], meta, pf.RUNGS["angle"], inn, [])
    raise AssertionError("missing angle must fail loudly")
except SystemExit:
    pass

# ladder runs end to end; base rung reproduces p3_sweep on the same data
ang = pd.Series(rng.uniform(31, 44, len(data)), index=data.index)
emb = pd.DataFrame(rng.normal(size=(len(data), 6)), index=data.index, columns=[f"A0{i}_2021" for i in range(6)])
with contextlib.redirect_stdout(io.StringIO()):
    res = pf.run([data], meta, pix, [ang], emb, ("3", "4"), seeds=2, leaf=5, trees=30)
assert set(res.rung) == {"base", "lin", "inner", "std", "angle", "norm", "emb", "base+emb"}, set(res.rung)
assert res[res.rung == "base"].verdict.iloc[0] == "reference" and set(res.verdict) <= {"", "reference", "KEEP", "drop"}
X, y, g = sw.build([data], meta, "field", "raw")
direct = pf.evaluate(X, y, g, "4", 2, 5, 30)["f1"]
assert abs(direct - res[(res.rung == "base") & (res.classes == "4")].f1.iloc[0]) < 1e-9

# p2_extra.sample: chunking, resume, loud failure
pp = pix.copy()
pp["pid"] = np.arange(len(pp))
orb = pd.DataFrame({"orbit": "ASC_56"}, index=["c1"])
calls = []
def fake_fetch(ee, img, names, sub, ts):
    calls.append(len(sub))
    return pd.DataFrame({n: sub.pid.to_numpy() * 1.0 for n in names}, index=pd.Index(sub.pid.to_numpy(), name="pid"))
pe.fetch = fake_fetch
with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
    r1 = p2_extra.sample(None, pp, orb, lambda o: None, ["A", "B"], d, 1, chunk=500)
    n1 = len(calls)
    r2 = p2_extra.sample(None, pp, orb, lambda o: None, ["A", "B"], d, 1, chunk=500)
    assert len(calls) == n1 and r1.equals(r2) and len(r1) == len(pp)           # second run reuses chunks
    try:
        p2_extra.sample(None, pp.iloc[:900], orb, lambda o: None, ["A", "B"], d, 1, chunk=500)
        raise AssertionError("stale chunks must fail loudly")
    except SystemExit:
        pass
# image builders only checked for plain Python errors (a MagicMock accepts any call, so this does NOT validate the Earth Engine API)
p2_extra.angle_image(MagicMock(), "DESC_92", "2021-10-01")
p2_extra.embed_image(MagicMock())
print("SELFTEST OK")
