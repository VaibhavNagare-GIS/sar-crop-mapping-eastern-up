"""Run: python selftest_phase3d.py -> 'SELFTEST OK'. Synthetic data only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))   # tests import the code in src/
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import p3_diag as dg
import selftest_phase3c as t3


def main():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        (d / "f").mkdir(); (d / "phase1").mkdir()
        tab = t3.make(rng)
        tab.to_csv(d / "f" / "features.csv", index=False)
        pd.DataFrame(dict(field=range(80), npix=[8, 20, 40, 80] * 20)).to_csv(d / "phase1" / "fields_study.csv", index=False)
        r = dg.run(str(d / "f"), data=str(d), leaf=5, seeds=2, trees=30)
        assert set(r["size"].fields) == {16} or r["size"].fields.sum() == 64, r["size"]            # 4 classes of 5 -> 64 fields
        assert len(r["size"]) == 4 and r["curve"].shape[0] == 5 and list(r["clean"].index) == [0.0, 0.15, 0.25]
        assert r["curve"].macro_F1.iloc[-1] >= r["curve"].macro_F1.iloc[0] - 0.05                    # more data does not hurt
        assert (r["clean"].dropped_share.iloc[0] == 0) and (r["clean"].dropped_share.diff().dropna() >= 0).all()
        assert r["confusion"].shape == (4, 4) and np.allclose(r["confusion"].sum(axis=1), 1, atol=0.03)
        assert r["merge"]["macro_F1_4class"] > 0.3 and r["merge"]["AUC_merged_3class"] > 0.5
    print("SELFTEST OK")


if __name__ == "__main__":
    main()
