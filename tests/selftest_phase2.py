"""Run: python selftest_phase2.py -> 'SELFTEST OK'. Mock data and a fake Earth Engine (no network)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))   # tests import the code in src/
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

import features
import p2_extract as p2
import p2_season_test as st
import study


class Rec:
    """Fake ee object: records every method call; sampleRegions fabricates values and can fake memory errors."""
    calls = []
    bands = None
    mem_bands, mem_points, fail_once, n_requests = 10 ** 9, 10 ** 9, False, 0

    def select(self, names):
        r = Rec("sel")
        r.bands = list(names)
        return r

    def __init__(self, tag=""):
        self.tag = tag

    def __getattr__(self, n):
        def f(*a, **k):
            Rec.calls.append((n, a))
            return Rec(n)
        return f

    def map(self, fn):
        Rec.calls.append(("map", ()))
        fn(Rec("im"))
        return Rec("mapped")

    def sampleRegions(self, collection, properties, scale, tileScale):
        Rec.n_requests += 1
        if Rec.fail_once:
            Rec.fail_once = False
            raise RuntimeError("Internal error: backend hiccup")
        if len(self.bands) > Rec.mem_bands or len(collection) > Rec.mem_points:
            raise RuntimeError("User memory limit exceeded.")
        feats = []
        for g, props in collection:
            p = props["pid"]
            d = {"pid": p}
            for i in range(study.N_BINS):
                d[f"VV_{i:02d}"] = -9999.0 if (p % 7 == 0 and i == 5) else -12.0 - i * 0.1
                d[f"VH_{i:02d}"] = -18.0 - i * 0.1
                d[f"NDVI_{i:02d}"] = -9999.0 if p % 5 == 0 else 0.3 + i * 0.01
            feats.append({"properties": {k: v for k, v in d.items() if k == "pid" or k in self.bands}})  # only selected bands
        return types.SimpleNamespace(getInfo=lambda: {"features": feats})


def fake_ee():
    ee = types.SimpleNamespace()
    ee.ImageCollection = lambda x: Rec("col")
    ee.Filter = types.SimpleNamespace(eq=lambda *a: 0, listContains=lambda *a: 0)
    class FakeImage:                      # callable like ee.Image(x), with ee.Image.constant / .cat
        def __new__(cls, x): return x
        constant = staticmethod(lambda v: Rec("const"))
        cat = staticmethod(lambda l: Rec("cat"))
    ee.Image = FakeImage
    ee.Geometry = types.SimpleNamespace(Point=lambda c: c, Rectangle=lambda c: c)
    ee.Feature = lambda g, p: (g, p)
    ee.FeatureCollection = lambda l: l
    return ee


def main():
    # bins
    b = study.bins("2021-10-01")
    assert len(b) == 20 and b[0] == ("2021-10-01", "2021-10-13") and b[-1][1] == study.season_end("2021-10-01") == "2022-05-29"
    assert all(b[i][1] == b[i + 1][0] for i in range(19))
    # fill_time / make_X
    a = np.array([[np.nan, 1, np.nan, 3, np.nan]])
    assert list(features.fill_time(a)[0]) == [1, 1, 2, 3, 3]
    try:
        features.fill_time(np.full((1, 3), np.nan)); raise AssertionError("all-NaN row accepted")
    except ValueError:
        pass
    vv, vh = features.band_cols()
    df = pd.DataFrame(np.random.default_rng(0).normal(-15, 2, (4, 40)), columns=vv + vh)
    X, cols = features.make_X(df)
    assert X.shape == (4, 60) and len(cols) == 60 and np.allclose(X[:, 40:], X[:, 20:40] - X[:, :20])
    df.iloc[0, :] = np.nan
    d2, n = features.drop_empty(df)
    assert n == 1 and len(d2) == 3
    # plan
    pix = pd.DataFrame(dict(pid=range(10), chip=["a"] * 6 + ["b"] * 4, lon=1.0, lat=2.0))
    orb = pd.DataFrame({"orbit": ["ASC_56", "ASC_158"]}, index=["a", "b"])
    pl = p2.plan(pix, orb, 4)
    assert [(o, k, len(s)) for o, k, s in pl] == [("ASC_158", 0, 4), ("ASC_56", 0, 4), ("ASC_56", 1, 2)], pl
    try:
        p2.plan(pix, orb.drop("b"), 4); raise AssertionError("missing orbit accepted")
    except ValueError:
        pass
    # parse
    good = [{"properties": {"pid": float(i), **{c: -9999.0 if i == 0 else -1.0 for c in vv + vh}}} for i in range(3)]
    r = p2.parse(good, [0, 1, 2], vv + vh)
    assert r.loc[0].isna().all() and (r.loc[1] == -1).all()
    for bad_feats, pids in [(good[:2], [0, 1, 2]), (good + good[:1], [0, 1, 2]), ([], [0])]:
        try:
            p2.parse(bad_feats, pids, vv + vh); raise AssertionError("bad samples accepted")
        except ValueError:
            pass
    # orbit_image: 20 windows, each merged with the pad, unmasked at the end
    Rec.calls = []
    p2.orbit_image(fake_ee(), "ASC_56", "2021-10-01")
    names = [c[0] for c in Rec.calls]
    assert names.count("filterDate") == 20 and names.count("merge") == 20 and names.count("unmask") == 1, names.count("merge")
    assert [c[1] for c in Rec.calls if c[0] == "filterDate"][0] == ("2021-10-01", "2021-10-13")
    # memory-safe sampler: forced memory errors must be handled by splitting, with identical results
    import io, contextlib
    p2.time.sleep = lambda s: None
    pix8 = pd.DataFrame(dict(pid=range(8), lon=81.0, lat=27.0))
    nmz = sum(features.band_cols(), [])
    Rec.mem_bands = Rec.mem_points = 10 ** 9
    ref = p2.fetch(fake_ee(), Rec("img"), nmz, pix8, 4)
    Rec.mem_bands, Rec.mem_points, Rec.n_requests = 10, 3, 0                 # at most 10 bands and 3 points per request
    with contextlib.redirect_stdout(io.StringIO()):
        got = p2.fetch(fake_ee(), Rec("img"), nmz, pix8, 4)
    assert Rec.n_requests > 1 and got.shape == ref.shape and got.equals(ref), Rec.n_requests
    Rec.mem_bands, Rec.mem_points, Rec.n_requests, Rec.fail_once = 10 ** 9, 10 ** 9, 0, True   # transient error: retried once
    with contextlib.redirect_stdout(io.StringIO()):
        got2 = p2.fetch(fake_ee(), Rec("img"), nmz, pix8, 4)
    assert Rec.n_requests == 2 and got2.equals(ref)
    Rec.mem_bands = Rec.mem_points = 0                                       # impossible even for 1 band, 1 point -> loud exit
    try:
        p2.fetch(fake_ee(), Rec("img"), nmz[:1], pix8.iloc[:1], 4); raise AssertionError("no exit")
    except SystemExit as e:
        assert "Out of memory" in str(e)
    Rec.mem_bands = Rec.mem_points = 10 ** 9
    # manifest: same settings pass; changed settings with NO chunks are rewritten; with chunks they are refused
    with tempfile.TemporaryDirectory() as td:
        p2.manifest_check(td, {"x": 1}); p2.manifest_check(td, {"x": 1})
        p2.manifest_check(td, {"x": 2})                                   # no chunks yet: allowed
        assert p2.json.loads((Path(td) / "manifest.json").read_text()) == {"x": 2}
        (Path(td) / "chunk_ASC_56_0000.csv").write_text("pid\n")
        p2.manifest_check(td, {"x": 2})                                   # same settings with chunks: fine
        try:
            p2.manifest_check(td, {"x": 3}); raise AssertionError("mismatch with chunks accepted")
        except SystemExit as e:
            assert "different settings" in str(e) and "(2, 3)" in str(e)
    # pilot picks train only
    fs = pd.DataFrame(dict(field=range(100), crop=["A", "B"] * 50, split=["train"] * 80 + ["test"] * 20))
    pp = p2.pick_pilot(fs, 5)
    assert (pp.split == "train").all() and pp.groupby("crop").size().max() == 5
    # end to end with fake ee: chunking, resume, merge
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        (d / "phase1").mkdir()
        rows = []
        for i in range(60):
            rows.append(dict(chip="a" if i < 40 else "b", row=0, col=i, lon=81.0, lat=27.0, field=i // 3,
                             label=[1, 2][i % 2], split="train", block="x", crop="A", pid=i))
        pd.DataFrame(rows).drop(columns="pid").to_csv(d / "phase1" / "pixels.csv", index=False)
        pd.DataFrame(dict(chip=["a", "b"], orbit=["ASC_56", "ASC_158"], n_scenes=[22, 20], fallback=False, weak=[False, False])
                     ).to_csv(d / "phase1" / "chip_orbits.csv", index=False)
        study.gee_init = lambda p: fake_ee()
        sys.argv = ["p", "--project", "x", "--data", str(d), "--chunk", "25"]
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            p2.main()
        out = d / "phase2" / study.SEASON_START
        feat = pd.read_csv(out / "features.csv", dtype={"chip": str})
        assert len(feat) == 60 and feat.pid.is_unique and feat.chip.isin(["a", "b"]).all()
        assert feat.VV_05.isna().sum() == len([i for i in range(60) if i % 7 == 0])   # sentinel became NaN
        assert (feat.groupby("chip").orbit.nunique() == 1).all()
        n_chunks = len(list(out.glob("chunk_*.csv")))
        assert n_chunks == 3, n_chunks                                     # 40 px -> 25+15, 20 px -> 20
        with contextlib.redirect_stdout(io.StringIO()) as buf2:           # re-run: nothing recomputed
            p2.main()
        assert "chunk 0" not in buf2.getvalue()
    # smoothing: focalMean built once per window when asked, never by default
    Rec.calls = []; p2.orbit_image(fake_ee(), "ASC_56", "2021-10-01", smooth=1)
    assert [c[0] for c in Rec.calls].count("focalMean") == 20
    Rec.calls = []; p2.orbit_image(fake_ee(), "ASC_56", "2021-10-01")
    assert [c[0] for c in Rec.calls].count("focalMean") == 0
    # multi-temporal filter: ratio image built once, filter applied inside every window; none by default
    Rec.calls = []; p2.orbit_image(fake_ee(), "ASC_56", "2021-10-01", mt=2)
    nm = [c[0] for c in Rec.calls]
    assert nm.count("focalMean") >= 22 and nm.count("filterDate") == 21, (nm.count("focalMean"), nm.count("filterDate"))   # 1 ratio + 20 windows (+1 pad)
    try:
        sys.argv = ["p", "--project", "x", "--smooth", "1", "--mt", "1"]; p2.main(); raise AssertionError("smooth+mt accepted")
    except SystemExit as e:
        assert "either --smooth or --mt" in str(e)
    # Sentinel-1 filler type must equal the real scene type (Float vs Double mismatch would break merge)
    class SImg:
        def __init__(s, ty): s.ty = ty
        def _t(s, o): return "D" if "D" in (s.ty, getattr(o, "ty", "")) else s.ty
        def multiply(s, o): return SImg(s._t(o))
        def divide(s, o): return SImg(s._t(o))
        def exp(s): return SImg(s.ty)
        def focalMean(s, **k): return SImg(s.ty)
        def log10(s): return SImg(s.ty)
        def rename(s, n): return SImg(s.ty)
        def unmask(s, v): return SImg(s.ty)
        def updateMask(s, m): return SImg(s.ty)
    class SCol:
        def __init__(s, ty="raw"): s.ty = ty
        def filter(s, f): return s
        def select(s, b): return s
        def filterDate(s, a, b): return s
        def first(s): return SImg("raw")
        def map(s, fn): return SCol(fn(SImg("raw")).ty)
        def merge(s, o):
            if s.ty != o.ty: raise RuntimeError(f"Mismatched type {s.ty} vs {o.ty}")
            return s
        def mean(s): return SImg("D" if s.ty == "D" else s.ty)
    see = types.SimpleNamespace(ImageCollection=lambda x: SCol() if isinstance(x, str) else SCol(x[0].ty),
                                Filter=types.SimpleNamespace(eq=lambda *a: 0, listContains=lambda *a: 0),
                                Image=type("FI", (), {"__new__": lambda c, x: x, "cat": staticmethod(lambda l: SImg("cat"))}))
    p2.orbit_image(see, "ASC_56", "2021-10-01"); p2.orbit_image(see, "DESC_165", "2021-10-01", mt=1)
    # s2_image: 20 windows, pad merged each time, SCL mask and NDVI built, unmasked at the end
    Rec.calls = []
    p2.s2_image(fake_ee(), "2021-10-01", [0, 0, 1, 1])
    nm = [c[0] for c in Rec.calls]
    assert nm.count("filterDate") == 20 and nm.count("merge") == 20 and nm.count("unmask") == 1 and "normalizedDifference" in nm and nm.count("Or") == 42, nm   # mask built in 20 windows + 1 for the pad
    assert nm.index("filterDate") < nm.index("map"), "must filter by date BEFORE mapping the collection"
    # type-checking fake: reproduces Earth Engine's "Mismatched type" error on merge()
    class TImg:
        def __init__(s, ty): s.ty = ty
        def rename(s, n): return TImg(s.ty)
        def toFloat(s): return TImg("Float")
        def updateMask(s, m): return TImg(s.ty)
        def select(s, n): return TImg("band")
        def eq(s, v): return TImg("mask")
        def Or(s, o): return TImg("mask")
        def normalizedDifference(s, b): return TImg("Float<-1,1>")
        def unmask(s, v): return TImg(s.ty)
    class TCol:
        def __init__(s, ty): s.ty = ty
        def filterBounds(s, g): return s
        def filterDate(s, a, b): return s
        def first(s): return TImg("raw")
        def map(s, fn): return TCol(fn(TImg("raw")).ty)
        def merge(s, o):
            if s.ty != o.ty:
                raise RuntimeError(f"Mismatched type for band 'NDVI': Expected type: {s.ty}. Actual type: {o.ty}.")
            return s
        def median(s): return TImg(s.ty)
    class FImage:
        def __new__(cls, x): return x
        constant = staticmethod(lambda v: TImg("Float"))
        cat = staticmethod(lambda l: TImg("cat"))
    tee = types.SimpleNamespace(
        ImageCollection=lambda x: TCol("raw") if isinstance(x, str) else TCol(x[0].ty),
        Image=FImage, Geometry=types.SimpleNamespace(Rectangle=lambda c: c))
    p2.s2_image(tee, "2021-10-01", [0, 0, 1, 1])        # raises if the pad type differs from the NDVI type
    # fatal Earth Engine errors must not be retried
    class Boom(Rec):
        def select(self, names): return self
        def sampleRegions(self, **k):
            Boom.n += 1
            raise RuntimeError("Expected a homogeneous image collection: Mismatched type")
    Boom.n = 0
    try:
        p2.fetch(fake_ee(), Boom("x"), ["VV_00"], pix8, 4); raise AssertionError("no exit")
    except SystemExit as e:
        assert "retrying cannot help" in str(e) and Boom.n == 1, Boom.n
    # NDVI tables work with the same helpers
    nd = features.ndvi_cols()
    ndf = pd.DataFrame(np.random.default_rng(2).uniform(0.1, 0.8, (5, 20)), columns=nd)
    ndf.iloc[0, 3:6] = np.nan
    assert set(features.series(ndf)) == {"NDVI"} and features.make_X(ndf)[0].shape == (5, 20)
    ndf.iloc[1, :] = np.nan
    assert features.drop_empty(ndf)[1] == 1
    # end to end, sensor s2
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        (d / "phase1").mkdir()
        rows = [dict(chip="a", row=0, col=i, lon=81.0, lat=27.0, field=i // 3, label=[1, 2][i % 2], split="train",
                     block="x", crop="A") for i in range(30)]
        pd.DataFrame(rows).to_csv(d / "phase1" / "pixels.csv", index=False)
        pd.DataFrame(dict(chip=["a"], orbit=["ASC_56"], n_scenes=[20], fallback=False, weak=[False])).to_csv(d / "phase1" / "chip_orbits.csv", index=False)
        study.gee_init = lambda p: fake_ee()
        Rec.mem_bands, Rec.mem_points = 3, 7                                  # force splitting inside main()
        sys.argv = ["p", "--project", "x", "--data", str(d), "--sensor", "s2"]
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            p2.main()
        o = d / "phase2" / (study.SEASON_START + "_s2")
        f2 = pd.read_csv(o / "features.csv")
        assert len(f2) == 30 and f2.NDVI_00.isna().sum() == 6 and "VV_00" not in f2
        assert f2.filter(like="NDVI_").shape[1] == 20 and f2.pid.is_unique
        Rec.mem_bands, Rec.mem_points = 10 ** 9, 10 ** 9
    # diagnose runs on both table types and writes the figure
    import p2_diagnose as dg
    rng = np.random.default_rng(3)
    n = 300
    lab = rng.choice([1, 2, 3], n)
    tt = np.arange(20)
    base = {1: 0.2 + 0.5 * np.exp(-((tt - 8) / 4) ** 2), 2: 0.2 + 0.5 * np.exp(-((tt - 14) / 4) ** 2), 3: np.full(20, 0.25)}
    nd2 = pd.DataFrame(np.stack([base[l] for l in lab]) + rng.normal(0, .05, (n, 20)), columns=nd)
    for k, v in dict(label=lab, field=np.arange(n) // 3, block=(np.arange(n) // 30).astype(str), split="train").items():
        nd2[k] = v
    nd2["crop"] = nd2.label.map({1: "Wheat", 2: "Mustard", 3: "Lentil"})
    with tempfile.TemporaryDirectory() as t:
        fd = Path(t) / "2021-10-01_s2_pilot"
        fd.mkdir()
        nd2.to_csv(fd / "features.csv", index=False)
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()) as bb:
            dg.diagnose(fd)
        assert (fd / "signatures.png").stat().st_size > 5000 and "field macro-F1" in bb.getvalue()
        nd2["split"] = "test"
        nd2.to_csv(fd / "features.csv", index=False)
        try:
            dg.diagnose(fd); raise AssertionError("test fields accepted")
        except SystemExit:
            pass
    # descending pass: coverage counts choose the orbit per chip; chips without enough scenes are dropped
    with tempfile.TemporaryDirectory() as t:
        d = Path(t); (d / "phase1").mkdir()
        rows = [dict(chip=c, row=0, col=i, lon=81.0, lat=27.0, field=i // 3, label=1, split="train", block="x", crop="A")
                for c in ["a", "b"] for i in range(10)]
        pd.DataFrame(rows).to_csv(d / "phase1" / "pixels.csv", index=False)
        pd.DataFrame(dict(chip=["a", "b"], orbit=["ASC_56"] * 2, n_scenes=[20, 20], fallback=False, weak=False)).to_csv(d / "phase1" / "chip_orbits.csv", index=False)
        pd.DataFrame({"chip": ["a", "b"], "ASC_56": [20, 20], "DESC_165": [18, 2], "DESC_92": [3, 3]}).to_csv(d / "phase1" / "orbit_counts.csv", index=False)
        study.gee_init = lambda p: fake_ee()
        sys.argv = ["p", "--project", "x", "--data", str(d), "--pass", "desc"]
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()) as bd:
            p2.main()
        o = d / "phase2" / (study.SEASON_START + "_desc")
        fd = pd.read_csv(o / "features.csv", dtype={"chip": str})
        assert set(fd.chip) == {"a"} and set(fd.orbit) == {"DESC_165"} and len(fd) == 10, fd.chip.unique()
        assert "dropped in 1 weak" in bd.getvalue()
    # old manifests (made before mt/pass options) still match the default settings
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "chunk_ASC_56_0000.csv").write_text("pid\n")
        base = dict(sensor="s1", smooth=0, mt=0, pass_="asc", start="2021-10-01", chunk=2000)
        (Path(td) / "manifest.json").write_text(p2.json.dumps({k: v for k, v in base.items() if k not in ("mt", "pass_")}))
        p2.manifest_check(td, base)
    # season test: train-only guard + a learnable synthetic signal
    rng = np.random.default_rng(1)
    n = 400
    lab = rng.integers(1, 4, n)
    t = np.arange(20)
    base = {1: -12 + 4 * np.sin(t / 3), 2: -14 + 4 * np.cos(t / 3), 3: np.full(20, -13.0)}
    vvm = np.stack([base[l] for l in lab]) + rng.normal(0, 1, (n, 20))
    f = pd.DataFrame(np.hstack([vvm, vvm - 6]), columns=vv + vh)
    f["label"], f["field"], f["block"], f["split"] = lab, np.arange(n) // 4, np.arange(n) // 20, "train"
    p, fl, *_ = st.score(f.copy())
    assert p > 0.6, p
    print("SELFTEST OK")


if __name__ == "__main__":
    main()
