"""Run: python selftest_phase4.py -> 'SELFTEST OK'. Synthetic data only; checks freeze, the single-test guard, leak guard, window masking."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))   # tests import the code in src/
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import features
import p3_windows as pw
import p4_final as p4

rng = np.random.default_rng(1)
labs = [1, 1, 1, 2, 3, 4, 6]


def make(root, name, cols_fn, noise=1.0):
    rows, pid = [], 0
    for f in range(140):
        lab, split = labs[f % 7], ("test" if f % 5 == 0 else "train")
        blk = f"{'t' if split == 'test' else 'r'}{f // 10}"
        for _ in range(6):
            rows.append(dict(pid=pid, chip="c", field=f, label=lab, block=blk, split=split, **cols_fn(lab, noise)))
            pid += 1
    d = Path(root) / name
    d.mkdir()
    pd.DataFrame(rows).to_csv(d / "features.csv", index=False)
    return str(d)


def s1(lab, noise):
    base = {1: -12, 2: -15, 3: -14, 4: -11, 6: -9}[lab]
    return {**{c: base + rng.normal(0, 2 * noise) + (i % 5) for i, c in enumerate(features.band_cols()[0])},
            **{c: base - 6 + rng.normal(0, 2 * noise) for c in features.band_cols()[1]}}


def s2(lab, noise):
    return {c: 0.2 + 0.1 * lab + rng.normal(0, 0.15 * noise) for c in features.ndvi_cols()}


with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()) as buf:
    A, A2, S = make(d, "A", s1), make(d, "A2", s1, 0.8), make(d, "S", s2)
    out = Path(d) / "results"
    # freeze writes once and refuses to overwrite
    sys.argv = ["p4_final.py", "freeze", "--field-spec", A, "--pixel-spec", A2, "--classes", "3", "--trees", "40", "--bench", f"s1_s2={A}+{S}", "--out", str(out)]
    p4.main()
    try:
        p4.main()
        raise AssertionError("second freeze must fail")
    except SystemExit:
        pass
    cfg = json.loads((out / "FROZEN_CONFIG.json").read_text())
    assert cfg["bench"] == {"s1_s2": f"{A}+{S}"} and cfg["classes"] == "3"
    # one test evaluation
    res = p4.test(cfg, out, d, 30, False)
    for f in ("test_metrics.json", "per_class.csv", "confusion_field_primary.csv", "confusion_pixel_pixel.csv", "model.joblib", "TEST_USED.lock"):
        assert (out / f).exists(), f
    assert set(res["field"]) == {"primary", "s1_s2"} and res["field"]["primary"]["n_fields"] == 24     # 28 test fields minus 4 Fallow fields
    lo, hi = res["field"]["primary"]["ci"]["f1"]
    assert lo <= res["field"]["primary"]["f1"] <= hi + 1e-9 and res["field"]["primary"]["f1"] > 0.5      # easy synthetic data
    assert "s1_s2" in res["field_diff_primary_minus"]
    import p4_report
    txt = p4_report.report(out)
    assert (out / "test_summary.md").exists() and "Primary" in txt and "confusion_pixel_pixel" in txt
    # no second look without --again
    try:
        p4.test(cfg, out, d, 30, False)
        raise AssertionError("second test run must be refused")
    except SystemExit:
        pass
    p4.test(cfg, out, d, 30, True)
    # leak guard
    m = pd.DataFrame(dict(field=[1, 2], block=["a", "b"]))
    try:
        p4.check_disjoint(m, m.iloc[:1])
        raise AssertionError("leak must be caught")
    except SystemExit:
        pass

# window masking: an early cut-off must not depend on later windows
df = pd.DataFrame(rng.normal(-12, 2, (10, 40)), columns=sum(features.band_cols(), []))
df2 = df.copy()
late = [c for c in df.columns if int(c.split("_")[1]) >= 10]
df2[late] = 99.0
X1, X2 = (features.make_X(pw.mask_windows(x, 0, 10), "raw")[0] for x in (df, df2))
assert np.allclose(X1, X2), "early cut-off leaks later windows"
assert pw.dates("2021-10-01", 0, 20) == "2021-10-01 to 2022-05-28"
print("SELFTEST OK", file=sys.__stdout__)
