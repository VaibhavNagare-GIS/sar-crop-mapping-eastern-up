"""Phase 5b: training table taken from the dense chip stacks, so training and map features come from the same code path.
python src/p5_dense_table.py --pass asc|desc [--mt 2]
Reads data/phase5/dense_<pass>_mt<k>/<chip>.npy and writes data/phase2/dense_<pass>_mt<k>/features.csv in the same layout as p2_extract,
so p3_sweep.py and p4_final.py can use it as a feature folder."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import p5_chip_features as cf


def build(data, pass_, mt):
    data = Path(data)
    pix = pd.read_csv(data / "phase1" / "pixels.csv", dtype={"chip": str})
    pix["pid"] = pix.index
    orb = cf.orbit_table(data, pass_)
    src = data / "phase5" / f"dense_{pass_}_mt{mt}"
    missing = [c for c in pix.chip.unique() if c in orb.index and not (src / f"{c}.npy").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} chips are not downloaded yet (first: {missing[0]}); run p5_chip_features.py download")
    pix = pix[pix.chip.isin(orb.index)]
    parts = []
    for chip, g in pix.groupby("chip"):
        arr = cf.dequantise(np.load(src / f"{chip}.npy"))
        parts.append(pd.DataFrame(arr[:, g.row.to_numpy(), g.col.to_numpy()].T, columns=cf.NAMES, index=g.index))
    out = pix.join(pd.concat(parts))
    out.insert(out.columns.get_loc("pid") + 1, "orbit", out.chip.map(orb.orbit))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_", choices=["asc", "desc"], required=True)
    ap.add_argument("--mt", type=int, default=2)
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    df = build(a.data, a.pass_, a.mt)
    dst = Path(a.data) / "phase2" / f"dense_{a.pass_}_mt{a.mt}"
    dst.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst / "features.csv", index=False)
    print(f"{len(df)} pixels, {int(df[cf.NAMES].isna().all(axis=1).sum())} without any valid window -> {dst / 'features.csv'}")
