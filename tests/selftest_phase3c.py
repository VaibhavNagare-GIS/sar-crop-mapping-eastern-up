"""Run: python selftest_phase3c.py -> 'SELFTEST OK'. Synthetic tables only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))   # tests import the code in src/
import contextlib, io, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import features
import p3_sweep as sw
import study


def make(rng, kind="s1", shift=0.0, drop=0):
    t = np.arange(20)
    curves = {1: -12 + 4 * np.sin(t / 3), 2: -14 + 4 * np.cos(t / 3), 3: np.full(20, -13.0), 4: -12 + 2 * np.sin(t / 2), 6: np.linspace(-12, -9, 20)}
    rows = []
    for f in range(80):
        lab = [1, 2, 3, 4, 6][f % 5]
        for k in range(5):
            c = curves[lab] + shift + rng.normal(0, 1.5, 20)
            r = dict(pid=f * 5 + k, field=f, label=lab, block=str(f // 8), split="train", chip="c")
            r.update({f"VV_{i:02d}": c[i] for i in range(20)} | {f"VH_{i:02d}": c[i] - 6 for i in range(20)} if kind == "s1"
                     else {f"NDVI_{i:02d}": 0.5 + 0.03 * c[i] for i in range(20)})
            rows.append(r)
    df = pd.DataFrame(rows)
    if drop:
        df = df[df.pid >= drop]                                  # a table that covers fewer pixels (like descending)
    junk = df.sample(20, random_state=1).copy()                  # test rows with wrong labels must never be used
    junk["split"], junk["label"], junk["pid"] = "test", 1, junk.pid + 10 ** 6
    return pd.concat([df, junk], ignore_index=True)


def main():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        for name, tab in [("asc", make(rng)), ("desc", make(rng, shift=0.5, drop=100)), ("s2", make(rng, kind="s2"))]:
            (d / name).mkdir()
            tab.to_csv(d / name / "features.csv", index=False)
        a, b, c = str(d / "asc"), str(d / "desc"), str(d / "s2")
        parts, meta = sw.load(a)
        assert (meta.split == "train").all() and len(meta) == 400
        p2, m2 = sw.load(f"{a}+{b}")
        assert len(m2) == 300 and len(p2) == 2                                        # inner join on pids: desc covers fewer
        X, y, g = sw.build(p2, m2, "pixel", "raw")
        assert X.shape == (300, 120)
        Xf, yf, gf = sw.build(p2, m2, "field", "raw")
        assert Xf.shape == (60, 120) and len(yf) == 60 and len(gf) == 60              # 60 fields remain
        assert sw.build(parts, meta, "pixel", "summary")[0].shape[1] == 33            # 3 series x 11 stats
        assert sw.build(parts, meta, "pixel", "both")[0].shape[1] == 93
        assert sw.build(parts, meta, "pixel", "smooth")[0].shape[1] == 60
        # labels must agree across folders of one spec
        bad = pd.read_csv(d / "desc" / "features.csv"); bad["label"] = 3; bad.to_csv(d / "desc" / "features.csv", index=False)
        try:
            sw.load(f"{a}+{b}"); raise AssertionError("label mismatch accepted")
        except SystemExit:
            pass
        bad["label"] = pd.read_csv(d / "asc" / "features.csv").set_index("pid").label.reindex(bad.pid).fillna(1).astype(int).values
        bad.to_csv(d / "desc" / "features.csv", index=False)
        out = d / "o" / "sweep.csv"
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            r1 = sw.run([a, f"{a}+{c}", b], leaves=(1, 10), seeds=2, kind="raw", levels=("field", "pixel"), classes=("4", "5"),
                        models=("rf",), common=True, trees=20, out=str(out))
        assert "--common: 300 pixels" in buf.getvalue()
        assert len(r1) == 3 * 2 * 2 * 2
        assert r1[(r1.classes == "4") & (r1.level == "pixel")].n.max() < r1[(r1.classes == "5") & (r1.level == "pixel")].n.max()   # fallow rows removed
        assert (r1.f1_mean > 0.3).all() and (r1.auc_macro > 0.6).all() and (r1.baseline_oa > 0.15).all(), r1
        assert set(r1.n_features) >= {60, 80}
        with contextlib.redirect_stdout(io.StringIO()):
            r3 = sw.run([a], leaves=(10,), seeds=1, classes=("3",), levels=("field",), trees=20, out=str(d / "o" / "s3.csv"))
        assert r3.auc_Wh.notna().all() and r3.auc_Mu.notna().all() and r3.auc_Su.notna().all() and "auc_Le" not in r3, r3   # Mustard+Lentil merged
        with contextlib.redirect_stdout(io.StringIO()):
            r2 = sw.run([a], leaves=(10,), seeds=1, kind="summary", levels=("field",), classes=("4",),
                        models=("et", "hgb", "lr"), trees=20, out=str(out))
        assert len(r2) == 3 and (r2.f1_mean > 0.3).all(), r2
        assert len(pd.read_csv(out)) == len(r1) + 3                                    # rows are appended
        try:
            features.make_X(parts[0], "nope"); raise AssertionError("bad kind accepted")
        except ValueError:
            pass
    print("SELFTEST OK")


if __name__ == "__main__":
    main()
