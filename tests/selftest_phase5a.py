"""Run: python selftest_phase5a.py -> 'SELFTEST OK'. Fake Earth Engine and a synthetic GeoTIFF; checks grid, fetch, split-on-memory-error, quantisation, parity statistics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))   # tests import the code in src/
import tempfile
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.transform import from_origin

import p5_chip_features as p5

H, W = 8, 9
names = p5.NAMES


def value(b, r, c):
    return -12.0 + 0.1 * b + 0.3 * r + 0.2 * c


def make_ee(limit=None, as_bytes=False, calls=None):
    class Img:
        def select(self, n):
            return n
    def computePixels(req):
        n = req["expression"]
        if calls is not None:
            calls.append(len(n))
        if limit and len(n) > limit:
            raise Exception("User memory limit exceeded.")
        rr, cc = np.mgrid[0:H, 0:W]
        arr = np.zeros((H, W), dtype=[(k, "f8") for k in n])
        for k in n:
            arr[k] = value(names.index(k), rr, cc)
        if names[0] in n:
            arr[names[0]][0, 0] = -9999.0
        if as_bytes:
            import io
            b = io.BytesIO(); np.save(b, arr); return b.getvalue()
        return arr
    return SimpleNamespace(data=SimpleNamespace(computePixels=computePixels)), Img()


# grid from a real tif
with tempfile.TemporaryDirectory() as d:
    with rasterio.open(f"{d}/l.tif", "w", driver="GTiff", height=H, width=W, count=1, dtype="uint8", crs="EPSG:32644", transform=from_origin(500000, 3000000, 10, 10)) as f:
        f.write(np.ones((1, H, W), "uint8"))
    g = p5.grid(f"{d}/l.tif")
assert (g["w"], g["h"], g["crs"], g["aff"]["scaleX"], g["aff"]["scaleY"], g["aff"]["translateY"]) == (W, H, "EPSG:32644", 10, -10, 3000000)

ee, img = make_ee()
D = p5.fetch_chip(ee, img, g, names)
assert D.shape == (40, H, W) and np.isnan(D[0, 0, 0]) and np.isclose(D[3, 2, 4], value(3, 2, 4))
calls = []
ee2, _ = make_ee(limit=10, calls=calls)                       # memory error -> bands are split until the request fits
D2 = p5.fetch_chip(ee2, img, g, names)
assert np.allclose(D, D2, equal_nan=True) and max(c for c in calls if c <= 10) <= 10 and calls[0] == 40
assert np.allclose(D, p5.fetch_chip(*make_ee(as_bytes=True)[:1], img, g, names), equal_nan=True)   # NPY bytes also accepted
try:
    p5.fetch_chip(SimpleNamespace(data=SimpleNamespace(computePixels=lambda r: (_ for _ in ()).throw(Exception("Image.select: band not found")))), img, g, names)
    raise AssertionError("fatal error must stop")
except SystemExit:
    pass

q = p5.quantise(D)
assert q.dtype == np.int16 and q[0, 0, 0] == p5.NO and np.nanmax(np.abs(p5.dequantise(q) - D)) <= 0.005 + 1e-6

# parity statistics: identical -> r 1 and shift (0,0) is best; a table taken one pixel off is detected
r, c = np.array([2, 3, 4, 5, 3]), np.array([2, 3, 4, 5, 6])
csv = D[:, r, c].T
s0 = p5.compare(D, csv, r, c)
assert s0["r"] > 0.9999 and s0["mad"] < 1e-5
shifts = {s: p5.compare(D, csv, r, c, s)["mad"] for s in ((0, 0), (1, 0), (0, 1))}
assert min(shifts, key=shifts.get) == (0, 0) and shifts[(1, 0)] > 0.1
# too few pairs give no statistics instead of warnings; pooling pairs from several chips works
assert p5.stats(np.array([1.0]), np.array([1.0])) is None
xs, ys = zip(*[p5.paired(D, csv, r, c) for _ in range(2)])
assert abs(p5.stats(np.concatenate(xs), np.concatenate(ys))["r"] - 1) < 1e-6
print("SELFTEST OK")
