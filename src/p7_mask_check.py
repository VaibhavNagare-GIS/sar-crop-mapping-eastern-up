"""Phase 7c: how much of the labelled field area does the ESA WorldCover 2021 cropland class (40) cover?
python src/p7_mask_check.py [--set _near|_far]
Needs data/phase5<set>/dense_wc (python src/p5_chip_features.py download --sensor wc [--set ...]). Prints, per surveyed crop, the share of labelled pixels
that fall inside WorldCover Cropland and in each other class, and writes results/mask_coverage<set>.csv. A low share means a cropland mask would
remove real fields from the map."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WC = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up", 60: "Bare / sparse", 80: "Water", 90: "Wetland", 0: "No data"}


def coverage(pix, wc_of_chip):
    """pix: chip, row, col, crop. wc_of_chip(chip) -> (h, w) class codes. Returns a crop x WorldCover-class table of row shares (%) and the counts."""
    parts = [pd.DataFrame({"crop": g.crop.to_numpy(), "wc": wc_of_chip(c)[g.row.to_numpy(), g.col.to_numpy()]}) for c, g in pix.groupby("chip")]
    df = pd.concat(parts)
    df["wc"] = df.wc.map(lambda v: WC.get(int(v), f"class {int(v)}"))
    return (pd.crosstab(df.crop, df.wc, normalize="index") * 100).round(1), df.crop.value_counts()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_", default="", choices=["", "_near", "_far"])
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    data = Path(a.data)
    pix = pd.read_csv(data / f"phase1{a.set_}" / "pixels.csv", dtype={"chip": str})
    have = {p.stem for p in (data / f"phase5{a.set_}" / "dense_wc").glob("*.npy")}
    pix = pix[pix.chip.isin(have)]
    tab, n = coverage(pix, lambda c: np.load(data / f"phase5{a.set_}" / "dense_wc" / f"{c}.npy"))
    tab.insert(0, "pixels", n)
    print(tab.to_string())
    Path("results").mkdir(exist_ok=True)
    tab.to_csv(Path("results") / f"mask_coverage{a.set_}.csv")
