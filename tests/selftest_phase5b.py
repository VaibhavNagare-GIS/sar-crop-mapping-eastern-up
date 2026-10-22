"""Run: python tests/selftest_phase5b.py -> 'SELFTEST OK'. Synthetic dense stacks; checks that the table reproduces the stack values at the labelled pixels."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

import p5_chip_features as cf
import p5_dense_table as dt

rng = np.random.default_rng(0)
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    (d / "phase1").mkdir()
    (d / "phase5" / "dense_asc_mt2").mkdir(parents=True)
    pix = pd.DataFrame(dict(chip=["a", "a", "b"], row=[1, 2, 3], col=[4, 5, 6], lon=0.0, lat=0.0, field=[1, 1, 2], label=[1, 1, 2], crop="x", split="train", block="b"))
    pix.to_csv(d / "phase1" / "pixels.csv", index=False)
    pd.DataFrame({"chip": ["a", "b"], "orbit": "ASC_56", "weak": False}).to_csv(d / "phase1" / "chip_orbits.csv", index=False)
    stacks = {}
    for c in "ab":
        x = rng.normal(-12, 3, (40, 8, 9)).astype("float32")
        x[0, 1, 4] = np.nan
        stacks[c] = cf.dequantise(cf.quantise(x))
        np.save(d / "phase5" / "dense_asc_mt2" / f"{c}.npy", cf.quantise(x))
    out = dt.build(d, "asc", 2)
    assert list(out.pid) == [0, 1, 2] and (out.orbit == "ASC_56").all()
    for _, r in out.iterrows():
        assert np.allclose(r[cf.NAMES].to_numpy(float), stacks[r.chip][:, r.row, r.col], equal_nan=True)
    # Sentinel-2 table: NDVI columns, scale 10000, every chip
    (d / "phase5" / "dense_s2").mkdir()
    for c in "ab":
        x = rng.uniform(0.1, 0.9, (20, 8, 9)).astype("float32")
        x[3, 1, 4] = np.nan
        np.save(d / "phase5" / "dense_s2" / f"{c}.npy", cf.quantise(x, 10000))
    o2 = dt.build(d, "asc", 2, "", None, "s2")
    assert list(o2.columns[-20:]) == cf.NAMES_S2 and (o2.orbit == "S2").all() and len(o2) == 3 and np.isnan(o2.loc[o2.chip == "a", "NDVI_03"]).sum() <= 1
    # an extra set reads and writes its own folders
    import shutil
    shutil.copytree(d / "phase1", d / "phase1_x")
    shutil.copytree(d / "phase5", d / "phase5_x")
    ox = dt.build(d, "asc", 2, "_x")
    assert len(ox) == 3 and np.allclose(ox[cf.NAMES].to_numpy(float), out[cf.NAMES].to_numpy(float), equal_nan=True)
    pd.DataFrame({"chip": ["a", "b"], "ASC_56": [20, 20], "DESC_165": [19, 3]}).to_csv(d / "phase1" / "orbit_counts.csv", index=False)
    only = cf.orbit_table(d, "asc", "", "DESC_165")
    assert list(only.index) == ["a"] and only.orbit.iloc[0] == "DESC_165"
    try:
        cf.orbit_table(d, "asc", "", "ASC_999")
        raise AssertionError("unknown orbit must stop")
    except SystemExit:
        pass
    (d / "phase5" / "dense_asc_mt2" / "b.npy").unlink()
    try:
        dt.build(d, "asc", 2)
        raise AssertionError("missing chips must stop the build")
    except SystemExit:
        pass
print("SELFTEST OK")
