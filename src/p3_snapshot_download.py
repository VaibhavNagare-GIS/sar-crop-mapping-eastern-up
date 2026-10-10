"""Phase 3b-1: download the dataset's OWN Sentinel-2 chips (12 bands) for the study chips only.

Usage: python p3_snapshot_download.py [data_dir]     (default data; needs data/inventory/chips.csv)
These images sit on exactly the same 10 m grid as the labels, so they test the labels without any resampling.
"""
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import p0_download as dl  # reuses listing + download; its OUT folder is argv[1]
import study


def pick_keys(keys, chips):
    """Pure: .tif keys under source/ whose chip id (last token of the chip folder name) is in `chips`."""
    out = []
    for k in keys:
        parts = k.split("/")
        if len(parts) >= 3 and parts[0] == "source" and k.endswith(".tif") and parts[1].rsplit("_", 1)[-1] in chips:
            out.append(k)
    return out


def main():
    ch = pd.read_csv(dl.OUT / "inventory" / "chips.csv", dtype={"chip": str, "cell": str})
    chips = set(ch.chip[ch.cell.isin(study.BLOCK_CELLS)])
    cand = dl.pick_candidate()
    keys = pick_keys(dl.list_keys("source/", cand), chips)
    per_chip = pd.Series([k.split("/")[1] for k in keys]).value_counts()
    print(f"{len(chips)} study chips, {len(keys)} files, files per chip: {sorted(per_chip.unique())}")
    if len(per_chip) != len(chips) or per_chip.nunique() != 1:
        raise SystemExit("Some study chips have missing or uneven source files.")
    fail = 0
    with ThreadPoolExecutor(8) as ex:
        for i, (k, status, _) in enumerate(ex.map(dl.fetch, keys), 1):
            if status.startswith("FAILED"):
                fail += 1
                print(k, status)
            if i % 500 == 0:
                print(f"{i}/{len(keys)}")
    print(f"finished: {len(keys) - fail} ok or skipped, {fail} failed")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
