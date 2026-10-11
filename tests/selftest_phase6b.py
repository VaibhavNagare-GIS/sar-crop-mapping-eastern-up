"""Run: python tests/selftest_phase6b.py -> 'SELFTEST OK'. Context features and the orbit summary on synthetic data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

import p1_orbit_summary as osu
import p6_context as cx

rng = np.random.default_rng(0)
st = rng.normal(-12, 3, (80, 40, 40)).astype("float32")
st[10:12] = np.nan                       # an empty window must not produce NaN features
m = cx.context_maps(st)
assert m.shape == (16, 40, 40) and np.isfinite(m).all()
flat = np.full((80, 40, 40), -12.0, dtype="float32")
assert np.allclose(cx.context_maps(flat)[1], 0, atol=1e-6)      # no texture on a flat image

c = pd.DataFrame({"ASC_56": [20, 20, 5], "ASC_158": [20, 0, 19], "DESC_92": [21, 21, 21]}, index=["a", "b", "c"])
per, combo = osu.summarise(c, 18)
assert per["ASC_56"] == 2 and per["DESC_92"] == 3 and combo["ASC_56+ASC_158+DESC_92"] == 1 and combo["ASC_56+DESC_92"] == 1
print("SELFTEST OK")
