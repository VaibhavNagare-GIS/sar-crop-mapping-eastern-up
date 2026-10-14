"""Run: python tests/selftest_phase7.py -> 'SELFTEST OK'. numpy parts of the neural-network scripts (no PyTorch needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

import nnprep as nn

# gaps: a window missing over the whole chip is interpolated in time, ends take the nearest window
a = np.zeros((6, 2, 2), dtype=np.float32)
for t, v in enumerate([1, 2, 3, 4, 5, 6]):
    a[t] = v
b = a.copy()
b[[1, 2, 5]] = np.nan                       # windows 1, 2 between 0 and 3; window 5 after the last valid one (4)
out = nn.fill_windows(b)
assert np.allclose(out[:, 0, 0], [1, 2, 3, 4, 5, 5]), out[:, 0, 0]
try:
    nn.fill_windows(np.full((3, 2, 2), np.nan))
    raise AssertionError("an all-missing series must fail")
except ValueError:
    pass

# chip array layout: asc VV, VH, VH-VV, desc VV, VH, VH-VV
st = np.zeros((80, 4, 4), dtype=np.float32)
st[0:20], st[20:40], st[40:60], st[60:80] = -10, -16, -11, -17
ca = nn.chip_array(st)
assert ca.shape == (6, 20, 4, 4) and np.allclose(ca[2], -6) and np.allclose(ca[3], -11) and np.allclose(ca[5], -6)

# normalisation and labels
m, s = nn.channel_stats([ca, ca + 1])
z = nn.normalise(ca, m, s)
assert z.shape == ca.shape and np.isfinite(z).all()
lab = nn.label_image(np.array([0, 1]), np.array([2, 3]), np.array([1, 2]), 4, 4)
assert lab[0, 2] == 1 and lab[1, 3] == 2 and (lab == -1).sum() == 14

# dequantise agrees with the one used for the stacks
q = np.array([[-1200, nn.NO]], dtype=np.int16)
d = nn.dequantise(q)
assert np.isclose(d[0, 0], -12.0) and np.isnan(d[0, 1])
try:
    import p5_chip_features as cf
    assert np.allclose(cf.dequantise(q), d, equal_nan=True)
except ImportError:
    pass                                      # rasterio is not installed on the GPU machine; the comparison runs on the laptop

# chips and their blocks
chips, blocks = nn.chip_blocks(np.array(["b", "a", "a", "b", "b"]), np.array(["x", "y", "y", "z", "z"]))
assert list(chips) == ["a", "b"] and list(blocks) == ["y", "z"]
print("SELFTEST OK")
