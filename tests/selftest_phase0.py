"""Run: python selftest_phase0.py   -> prints 'SELFTEST OK' or fails with an assertion.
Builds a tiny fake dataset in the documented folder layout (not real data) and checks
the inventory logic: field spanning two chips, conflicting labels, 0-label pixels, dates.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))   # tests import the code in src/
import json
import tempfile
from pathlib import Path

import env_fix  # must come before rasterio
import numpy as np
import rasterio
from rasterio.transform import from_origin

import p0_inventory as inv


def write(path, arr, x0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="uint32", crs="EPSG:32644", transform=from_origin(x0, 3000000, 10, 10)) as d:
        d.write(arr.astype("uint32"), 1)


def make_chip(root, cid, fid, lab, x0, date, src_layout):
    pre = "ref_agrifieldnet_competition_v1_labels_train_"
    d = root / "train_labels"
    write(d / f"{pre}{cid}.tif", lab, x0)
    write(d / f"{pre}{cid}_field_ids.tif", fid, x0)
    body = json.dumps({"properties": {"datetime": f"{date}T05:00:00Z"}, "id": f"item_{cid}"})
    s = root / "source"
    if src_layout == "folder":
        (s / f"ref_x_source_{cid}").mkdir(parents=True)
        (s / f"ref_x_source_{cid}" / "stac.json").write_text(body)
    else:
        s.mkdir(exist_ok=True)
        (s / f"ref_x_source_{cid}_stac.json").write_text(body)


def main():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        z = np.zeros((8, 8), int)
        f1, l1 = z.copy(), z.copy()
        f1[:2, :] = 7; l1[:2, :] = 1          # field 7: 16 px wheat in chip a
        f1[3:5, :4] = 9; l1[3:5, :4] = 2      # field 9: 8 px mustard
        f1[6, :2] = 11                        # field 11: id but NO label (2 px) -> excluded
        l1[7, 0] = 5                          # label but NO field id (1 px) -> excluded
        f2, l2 = z.copy(), z.copy()
        f2[:2, :] = 7; l2[:2, :] = 1          # field 7 continues in chip b: +16 px
        f2[4:6, :] = 13; l2[4:6, :] = 36      # field 13: 16 px rice
        f2[7, :] = 15; l2[7, :] = 1           # field 15 conflicts with chip c below
        f3, l3 = z.copy(), z.copy()
        f3[:1, :] = 15; l3[:1, :] = 2         # field 15 labelled mustard here -> conflict
        make_chip(root, "aaaa", f1, l1, 500000, "2022-01-10", "folder")
        make_chip(root, "bbbb", f2, l2, 500080, "2022-01-10", "flat")
        make_chip(root, "cccc", f3, l3, 500160, "2022-02-01", "flat")
        fields, chips, cells = inv.run(root, root / "out", cell_deg=1.0, min_fields=1)
        fd = fields.set_index("field")
        assert set(fd.index) == {7, 9, 13}, fd.index                  # 11 unlabeled, 15 conflicting
        assert fd.loc[7, "npix"] == 32 and fd.loc[7, "n_chips"] == 2   # spans two chips, summed
        assert fd.loc[7, "crop"] == "Wheat" and fd.loc[13, "crop"] == "Rice"
        assert abs(fd.loc[7, "area_ha"] - 0.32) < 1e-9                 # 32 px * 100 m2 = 0.32 ha
        assert chips.px_field_no_label.sum() == 2 and chips.px_label_no_field.sum() == 1
        assert set(chips.s2_date) == {"2022-01-10", "2022-02-01"}
        assert 26.5 < chips.lat.min() < 27.5 and 80.9 < chips.lon.min() < 81.1   # UTM 44N, 500000 E / 3000000 N
        assert (root / "out" / "conflicting_fields.csv").exists()
    with tempfile.TemporaryDirectory() as t:  # a chip with only one of its two files must fail loudly
        d = Path(t) / "train_labels"
        write(d / "ref_x_labels_train_zz1.tif", np.ones((2, 2), int), 500000)
        try:
            inv.find_chips(d)
            raise AssertionError("unpaired chip was not rejected")
        except SystemExit:
            pass
    import p0_download as dl  # real key shapes seen on the server
    pre = "train_labels/ref_agrifieldnet_competition_v1_labels_train_"
    keys = [pre + "001c1.tif", pre + "001c1_field_ids.tif", pre + "0023c.tif", pre + "0023c_field_ids.tif"]
    tifs, js = dl.select(keys, ["source/a_stac.json", "source/b_B01.tif"])
    assert len(tifs) == 4 and js == ["source/a_stac.json"]
    try:
        dl.select(keys[:3], [])
        raise AssertionError("unpaired keys were not rejected")
    except SystemExit:
        pass
    print("SELFTEST OK")


if __name__ == "__main__":
    main()
