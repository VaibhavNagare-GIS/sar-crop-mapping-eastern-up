"""Phase 8c: class signatures through the season, as a small JSON file for the web chart.
python src/p8_signatures.py [--folder data/phase2/2021-10-01] [--out docs]
Reads the Sentinel-1 feature table of TRAIN fields only (same rule as p2_diagnose.py; any other rows are ignored). For each crop and each 12-day window it
keeps the 25th, 50th and 75th percentile of VV and VH backscatter and of VH minus VV, in dB. Writes docs/data/signatures.json."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import features
import study


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=f"data/phase2/{study.SEASON_START}")
    ap.add_argument("--out", default="docs")
    a = ap.parse_args()
    folder = Path(a.folder)
    start = folder.name[:10]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", start):
        raise SystemExit(f"The folder name must start with the season start date (like 2021-10-01), got '{folder.name}'.")
    df = pd.read_csv(folder / "features.csv", dtype={"chip": str})
    df = df[df.split == "train"]
    if "VV_00" not in df or df.empty:
        raise SystemExit("No Sentinel-1 train rows in this table. Use the main Sentinel-1 folder, not the _s2 one.")
    df, dead = features.drop_empty(df)
    ser = {k: v for k, v in features.series(df).items() if k != "NDVI"}
    crops = sorted(df.crop.unique())
    out = dict(start=start, window_days=study.BIN_DAYS, windows=[b[0] for b in study.bins(start)], unit="dB", series={}, counts={})
    for name, arr in ser.items():
        out["series"][name] = {}
        for c in crops:
            lo, med, hi = np.percentile(arr[(df.crop == c).to_numpy()], [25, 50, 75], axis=0)
            out["series"][name][c] = dict(lo=lo.round(2).tolist(), med=med.round(2).tolist(), hi=hi.round(2).tolist())
    out["counts"] = {c: dict(pixels=int((df.crop == c).sum()), fields=int(df[df.crop == c].field.nunique())) for c in crops}
    p = Path(a.out, "data", "signatures.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, separators=(",", ":")))
    print(f"{dead} pixels with no valid window were dropped")
    print(f"{p}: {len(crops)} crops {crops}, {len(out['windows'])} windows from {start}, {p.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
