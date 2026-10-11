"""Run: python selftest_phase1.py -> 'SELFTEST OK'. Mock data only (not real). Tests the split rules,
pixel coordinates, label consistency and the orbit chooser."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))   # tests import the code in src/
import tempfile
from pathlib import Path

import env_fix  # must come before rasterio
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

import p0_inventory as inv
import p1_coverage as cov
import p1_pixels as p1
import study


def write(path, arr, x0, y0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="uint32",
                       crs="EPSG:32644", transform=from_origin(x0, y0, 10, 10)) as d:
        d.write(arr.astype("uint32"), 1)


def mock(root):
    cyc = [1, 2, 3, 4, 6, 1]
    n = 0
    for i in range(24):                       # 24 chips, ~30 km apart so each is its own spatial block
        fid, lab = np.zeros((16, 16), int), np.zeros((16, 16), int)
        for j in range(6):
            fid[2 * j:2 * j + 2, :] = i * 10 + j + 1
            lab[2 * j:2 * j + 2, :] = cyc[j]
        cid = f"c{i:03d}"
        pre = f"ref_agrifieldnet_competition_v1_labels_train_{cid}"
        x0, y0 = 300000 + i * 30000, 3000000 + (i % 3) * 20000
        write(root / "train_labels" / f"{pre}.tif", lab, x0, y0)
        write(root / "train_labels" / f"{pre}_field_ids.tif", fid, x0, y0)
        n += 6 * 32
    return n


def main():
    # 1) chooser
    c = pd.DataFrame({"ASC_56": [22, 0, 5, 0], "ASC_158": [3, 21, 5, 0], "DESC_165": [0, 0, 22, 0]},
                     index=["A", "B", "C", "D"])
    ch = cov.choose(c)
    assert list(ch.orbit) == ["ASC_56", "ASC_158", "DESC_165", "ASC_56"], ch
    assert list(ch.fallback) == [False, False, True, False] and list(ch.weak) == [False, False, False, True], ch
    nf = cov.choose(c, prefer="DESC", fallback=False)
    assert list(nf.orbit) == ["DESC_165"] * 4 and list(nf.weak) == [True, True, False, True] and not nf.fallback.any(), nf
    # 2) pixels + split
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        total = mock(root)
        inv.run(root, root / "inventory", cell_deg=0.5, min_fields=1)
        cells = list(pd.read_csv(root / "inventory" / "chips.csv", dtype={"cell": str}).cell.unique())
        pix, ft = p1.build(root, root / "phase1", cells=cells, classes=study.CLASSES, seed=42)
        assert len(pix) == total, (len(pix), total)
        # no leakage: a field or a block never sits in both splits
        assert pix.groupby("field").split.nunique().max() == 1
        assert pix.groupby("block").split.nunique().max() == 1
        assert set(pix.split) == {"train", "test"} and ft.split.isin(["train", "test"]).all()
        assert pix.groupby("crop").split.nunique().min() == 2
        # coordinates: independent sanity (UTM 44N, 10 m pixels, lat ~27 deg)
        a = pix[(pix.chip == "c000") & (pix.row == 0)].sort_values("col")
        dlon = np.diff(a.lon.values)
        assert 75 < a.lon.iloc[0] < 82 and 26.5 < a.lat.iloc[0] < 27.5
        assert np.allclose(dlon, 10 / (111320 * np.cos(np.radians(27.1))), rtol=0.05), dlon[:3]
        # reproducible
        pix2, _ = p1.build(root, root / "phase1b", cells=cells, classes=study.CLASSES, seed=42)
        assert (pix.split.values == pix2.split.values).all()
        saved = pd.read_csv(root / "phase1" / "pixels.csv", dtype={"chip": str})
        assert saved.chip.str.startswith("c").all() and len(saved) == total   # chip ids stay text
        # external set: no split, and no 0.1 degree block shared with the study block
        half = len(cells) // 2
        p1.build(root, root / "phase1", cells=cells[:half], classes=study.CLASSES, seed=42)
        ext, _ = p1.build(root, None, cells=cells[half:], classes=study.CLASSES, set_="_t")
        saved_ext = pd.read_csv(root / "phase1_t" / "pixels.csv", dtype={"chip": str})
        study_blocks = set(pd.read_csv(root / "phase1" / "pixels.csv").block)
        assert len(ext) > 0 and set(ext.split) == {"external"} and not set(ext.block) & study_blocks and len(saved_ext) == len(ext)
    assert study.season_end() == "2022-05-29", study.season_end()
    assert set(study.SETS) == {"", "_near", "_far"} and not set(study.SETS["_near"]) & set(study.BLOCK_CELLS)
    print("SELFTEST OK")


if __name__ == "__main__":
    main()
