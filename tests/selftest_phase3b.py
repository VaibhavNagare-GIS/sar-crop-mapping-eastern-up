"""Run: python selftest_phase3b.py -> 'SELFTEST OK'. Synthetic chips only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))   # tests import the code in src/
import contextlib, io, tempfile
from pathlib import Path

import env_fix  # must come before rasterio
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

import p3_labelcheck as lc
import p3_snapshot_download as sd


def write(path, arr, x0, dtype):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=dtype,
                       crs="EPSG:32644", transform=from_origin(x0, 3000000, 10, 10)) as d:
        d.write(arr.astype(dtype), 1)


def irregular(h=16):
    """12 fields with unequal widths, so the pattern is not periodic (a periodic grid would align with itself when shifted)."""
    r = np.searchsorted([5, 9], np.arange(h), side="right")
    c = np.searchsorted([3, 8, 11], np.arange(h), side="right")
    return r[:, None] * 4 + c[None, :] + 1


def main():
    # pick_keys: only .tif under source/ for wanted chips
    keys = ["source/ref_x_source_001c1/ref_x_source_001c1_B02_10m.tif", "source/ref_x_source_0023c/a_B02_10m.tif",
            "source/ref_x_source_001c1/readme.json", "train_labels/ref_x_labels_train_001c1.tif"]
    assert sd.pick_keys(keys, {"001c1"}) == [keys[0]]
    # shift convention + alignment: image shifted by (dy=1, dx=2) relative to labels -> minimum at that shift
    rng = np.random.default_rng(0)
    fid = irregular()
    val = rng.uniform(0.1, 0.9, 13)[fid]                                            # one value per field
    big_f, big_v = np.pad(fid, 8), np.pad(val, 8)                                    # margins so nothing is cropped
    arr_in = [(big_f, lc.shift(big_v, 1, 2, m=3))]                                   # image[y,x] = truth[y-1,x-2]
    g = lc.alignment(arr_in)
    dx, dy = np.unravel_index(np.nanargmin(g.to_numpy()), g.shape)
    assert (g.index[dx], g.columns[dy]) == (2, 1) and (g < 1e-6).to_numpy().sum() == 1, g
    g0 = lc.alignment([(big_f, big_v)])
    dx, dy = np.unravel_index(np.nanargmin(g0.to_numpy()), g0.shape)
    assert (g0.index[dx], g0.columns[dy]) == (0, 0) and g0.loc[0, 1] > 1.2           # shifting by 1 clearly worsens it
    # date guess: snapshot follows window 7
    n = 500
    s2 = pd.DataFrame({f"NDVI_{i:02d}": rng.uniform(0, 1, n) for i in range(20)}, index=range(n))
    snap = pd.Series(np.exp(s2.NDVI_07) + rng.normal(0, 0.05, n), index=range(n))
    assert lc.date_guess(snap, s2).idxmax() == "NDVI_07"
    # end to end on mock chips: 24 chips, 16 fields each, class-dependent NDVI
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        cls = [1, 2, 3, 4, 6]
        rows = []
        pid = 0
        for i in range(24):
            cid = f"c{i:03d}"
            f = irregular() + i * 100
            lab = np.vectorize(lambda x: cls[(x % 100) % 5])(f)
            nd = np.vectorize(lambda x: 0.2 + 0.12 * cls.index(cls[(x % 100) % 5]))(f) + rng.normal(0, 0.03, f.shape)
            pre = "ref_agrifieldnet_competition_v1_labels_train_"
            write(d / "train_labels" / f"{pre}{cid}.tif", lab, 300000 + i * 3000, "uint32")
            write(d / "train_labels" / f"{pre}{cid}_field_ids.tif", f, 300000 + i * 3000, "uint32")
            sdir = d / "source" / f"ref_agrifieldnet_competition_v1_source_{cid}"
            red, nir = 1000.0, 1000.0 * (1 + nd) / (1 - nd)
            for b, a in [("B04", np.full(f.shape, red)), ("B08", nir), ("B02", rng.uniform(100, 900, f.shape))]:
                write(sdir / f"ref_agrifieldnet_competition_v1_source_{cid}_{b}_10m.tif", a, 300000 + i * 3000, "uint16")
            for r in range(16):
                for c in range(16):
                    rows.append(dict(chip=cid, row=r, col=c, lon=81.0, lat=27.0, field=int(f[r, c]), label=int(lab[r, c]),
                                     split="train", block=f"b{i}", crop="x"))
        (d / "phase1").mkdir()
        pd.DataFrame(rows).to_csv(d / "phase1" / "pixels.csv", index=False)
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            lc.main(d)
        out = buf.getvalue()
        assert "A. Snapshot classification" in out and "-> aligned" in out and "C. skipped" in out, out
        assert (d / "phase3" / "snapshot.csv").exists()
        snap = pd.read_csv(d / "phase3" / "snapshot.csv", index_col=0)
        assert len(snap) == 24 * 256 and {"B02", "B04", "B08", "NDVI"} <= set(snap.columns)
        line = [l for l in out.splitlines() if "5 classes" in l and "pixel" in l][0]
        assert float(line.split("macro-F1")[1].split()[0]) > 0.8, line              # clean synthetic signal is found
    print("SELFTEST OK")


if __name__ == "__main__":
    main()
