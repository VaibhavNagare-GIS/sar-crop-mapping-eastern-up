"""Run: python selftest_phase3.py -> 'SELFTEST OK'. Synthetic tables only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))   # tests import the code in src/
import io, contextlib, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import features
import p3_models as p3
import study


def table(rng, n_fields=60, px=6, kind="s1", leak_test=True):
    rows = []
    t = np.arange(20)
    curves = {1: -12 + 4 * np.sin(t / 3), 2: -14 + 4 * np.cos(t / 3), 3: np.full(20, -13.0), 4: -12 + 2 * np.sin(t / 2), 6: np.linspace(-12, -9, 20)}
    labs = [1, 2, 3, 4, 6]
    for f in range(n_fields):
        lab = labs[f % 5]
        for k in range(px):
            base = curves[lab] + rng.normal(0, 1.5, 20)
            r = dict(pid=f * px + k, field=f, label=lab, block=str(f // 6), split="train", crop=study.CLASSES[lab], chip="c")
            if kind == "s1":
                r.update({**{f"VV_{i:02d}": base[i] for i in range(20)}, **{f"VH_{i:02d}": base[i] - 6 for i in range(20)}})
            else:
                r.update({f"NDVI_{i:02d}": 0.5 + 0.03 * base[i] for i in range(20)})
            rows.append(r)
    df = pd.DataFrame(rows)
    if leak_test:                                   # test rows with garbage must never be used
        junk = df.sample(30, random_state=0).copy()
        junk["split"], junk["label"], junk["pid"] = "test", 1, junk.pid + 10 ** 6
        df = pd.concat([df, junk], ignore_index=True)
    return df


def main():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as t:
        a, b = Path(t) / "s1", Path(t) / "s2"
        a.mkdir(); b.mkdir()
        s1 = table(rng)
        s2 = s1[["pid", "field", "label", "block", "split", "crop", "chip"]].copy()
        nd = table(np.random.default_rng(0), kind="s2", leak_test=False)
        s2 = s2.merge(nd[["pid"] + features.ndvi_cols()], on="pid", how="inner")
        s1.to_csv(a / "features.csv", index=False)
        s2.to_csv(b / "features.csv", index=False)
        df, dead = p3.load(str(a))
        assert (df.split == "train").all() and len(df) == 360 and dead == 0          # test rows excluded
        fused, _ = p3.load(f"{a}+{b}")
        assert features.make_X(fused)[0].shape[1] == 80                               # VV, VH, VHVV, NDVI
        try:
            p3.load(f"{a}+{a}"); raise AssertionError("same-sensor fusion accepted")
        except SystemExit:
            pass
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            res = p3.run([str(a), f"{a}+{b}"], seeds=2, trees=30, leaf=2, out_dir=str(Path(t) / "out"))
        assert len(res) == 8 and (Path(t) / "out" / "ablation.csv").exists()
        assert res.query("classes=='no fallow'").n.max() < res.query("classes=='5 classes'").n.max()   # fallow rows removed
        assert (res.f1_mean > 0.3).all() and (res.baseline_oa > 0).all() and (res.bal_acc > 0.3).all(), res                                                           # signal is learnable
    # a drop_empty rule for fused tables: one sensor empty is enough to drop the row
    df2 = fused.copy()
    df2.loc[df2.index[0], features.ndvi_cols()] = np.nan
    assert features.drop_empty(df2)[1] == 1
    print("SELFTEST OK")


if __name__ == "__main__":
    main()
