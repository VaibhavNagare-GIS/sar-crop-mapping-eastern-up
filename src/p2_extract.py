"""Phase 2a: 12-day composites at every labelled pixel: Sentinel-1 (VV, VH in dB) or Sentinel-2 (NDVI).

Usage: python p2_extract.py --project P [--sensor s1|s2] [--pass asc|desc] [--mt R | --smooth R] [--start 2021-10-01] [--pilot 60] [--chunk 2000] [--data data]
--pilot N : only N TRAIN fields per class (fast test, and the season test). Test fields are never used in a pilot.
Safe to re-run: finished chunks are skipped; a manifest stops different settings from being mixed in one folder.
Output: data/phase2/<start>[_s2][_pilot]/features.csv
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import features
import study

NODATA = -9999.0
FATAL = ("mismatch", "homogeneous", "invalid", "must be", "not found", "no band")  # deterministic errors
LN10_OVER_10 = 0.2302585092994046   # 10**(dB/10) == exp(dB * ln(10)/10)


def pick_pilot(fields, n, seed=0):
    """Pure: n random TRAIN fields per class."""
    tr = fields[fields.split == "train"]
    parts = [g.sample(min(n, len(g)), random_state=seed) for _, g in tr.groupby("crop")]
    return pd.concat(parts)


def plan(pix, orbits, chunk):
    """Pure: [(orbit, k, pixels)] with at most `chunk` pixels each and ONE orbit per chunk."""
    d = pix.merge(orbits[["orbit"]], left_on="chip", right_index=True, how="left")
    if d.orbit.isna().any():
        raise ValueError(f"{int(d.orbit.isna().sum())} pixels sit in chips without an orbit")
    out = []
    for o, g in d.groupby("orbit"):
        g = g.sort_values("pid")
        out += [(o, k // chunk, g.iloc[k:k + chunk]) for k in range(0, len(g), chunk)]
    return out


def parse(feats, pids, names):
    """Pure: Earth Engine samples -> DataFrame indexed by pid. Every pid must return exactly once."""
    df = pd.DataFrame([f["properties"] for f in feats])
    if df.empty or "pid" not in df:
        raise ValueError("Earth Engine returned no samples")
    df["pid"] = df.pid.astype(int)
    if len(df) != len(pids) or df.pid.duplicated().any() or set(pids) != set(df.pid):
        raise ValueError(f"Expected {len(pids)} samples, got {len(df)} ({len(set(pids) - set(df.pid))} missing)")
    if set(names) - set(df.columns):
        raise ValueError(f"Missing bands: {sorted(set(names) - set(df.columns))[:5]}")
    df[names] = df[names].astype(float).replace(NODATA, np.nan)
    return df.set_index("pid").sort_index()[names]


def manifest_check(folder, manifest):
    """Block mixing settings ONLY when finished chunks exist. A folder with no chunks (e.g. after a failed
    first attempt) just gets its manifest rewritten."""
    folder = Path(folder)
    p = folder / "manifest.json"
    if p.exists():
        dflt = dict(smooth=0, mt=0, pass_="asc")                                      # folders made before these options existed
        old, new = {**dflt, **json.loads(p.read_text())}, {**dflt, **manifest}
        if old != new and any(folder.glob("chunk_*.csv")):
            diff = {k: (old.get(k), new.get(k)) for k in set(old) | set(new) if old.get(k) != new.get(k)}
            raise SystemExit(f"{folder} already has chunks made with different settings (old, new): {diff}\n"
                             f"Delete the folder or use the old settings.")
        if old == new:
            return
    p.write_text(json.dumps(manifest, indent=1))


def orbit_image(ee, orbit, start, smooth=0, mt=0):
    """One image: VV_00..VV_19, VH_00..VH_19 in dB; empty windows get NODATA.
    smooth = radius (pixels) of a spatial mean of the window mean (0 none).
    mt = radius (pixels) for the Quegan-Yu multi-temporal speckle filter (0 none): every scene is replaced by
         (its local mean) x (average over the whole season of scene / local mean), so speckle is removed using time."""
    pas, rel = {"ASC": "ASCENDING", "DESC": "DESCENDING"}[orbit.split("_")[0]], int(orbit.split("_")[1])
    col = (ee.ImageCollection("COPERNICUS/S1_GRD").filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
           .filter(ee.Filter.eq("orbitProperties_pass", pas))
           .filter(ee.Filter.eq("relativeOrbitNumber_start", rel)).select(["VV", "VH"]))
    to_lin = lambda im: im.multiply(LN10_OVER_10).exp()                                  # dB -> linear power
    if mt:
        lm = lambda im: im.focalMean(radius=mt, kernelType="square", units="pixels")
        ratio = col.filterDate(start, study.season_end(start)).map(lambda im: to_lin(im).divide(lm(to_lin(im)))).mean()
        scene = lambda im: lm(to_lin(im)).multiply(ratio)
    else:
        scene = to_lin
    # Filler for empty windows = the SAME function applied to a real scene, fully masked: identical band type by construction.
    pad = ee.ImageCollection([scene(ee.Image(col.first())).updateMask(0)])
    out = []
    for i, (s, e) in enumerate(study.bins(start)):
        m = col.filterDate(s, e).map(scene).merge(pad).mean()
        if smooth:
            m = m.focalMean(radius=smooth, kernelType="square", units="pixels")          # average in linear power, then dB
        out.append(m.log10().multiply(10).rename([f"VV_{i:02d}", f"VH_{i:02d}"]))
    return ee.Image.cat(out).unmask(NODATA)


def s2_image(ee, start, box):
    """One image: NDVI_00..NDVI_19, median per window, clouds removed with the SCL band; empty windows get NODATA."""
    def ndvi(im):
        scl = im.select("SCL")
        ok = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6))          # vegetation, bare soil, water; clouds, shadow, haze out
        return im.normalizedDifference(["B8", "B4"]).updateMask(ok).rename("NDVI")
    col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(ee.Geometry.Rectangle(box))
    # Pad = the same NDVI function applied to one real image, fully masked. Same function => identical band type
    # (a constant image has type Float, NDVI has Float<-1,1>, and merge() refuses mixed types).
    pad = ee.ImageCollection([ndvi(ee.Image(col.first())).updateMask(0)])
    out = [col.filterDate(s, e).map(ndvi).merge(pad).median().rename(f"NDVI_{i:02d}")   # date filter first, then map
           for i, (s, e) in enumerate(study.bins(start))]
    return ee.Image.cat(out).unmask(NODATA)


def fetch(ee, img, names, sub, ts):
    """Sample bands `names` at pixels `sub` -> DataFrame indexed by pid.
    'Memory limit' errors are not retried blindly: the request is split (bands first, then pixels) and re-run.
    Data errors (ValueError from parse) are never retried."""
    for attempt in range(3):
        try:
            fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(x), float(y)]), {"pid": int(p)})
                                       for p, x, y in zip(sub.pid, sub.lon, sub.lat)])
            feats = img.select(names).sampleRegions(collection=fc, properties=["pid"], scale=10,
                                                    tileScale=ts).getInfo()["features"]
            return parse(feats, list(sub.pid), names)
        except ValueError:
            raise
        except Exception as e:
            if "memory" in str(e).lower():
                if len(names) > 1:
                    h = len(names) // 2
                    return pd.concat([fetch(ee, img, names[:h], sub, ts), fetch(ee, img, names[h:], sub, ts)], axis=1)
                if len(sub) > 1:
                    h = len(sub) // 2
                    return pd.concat([fetch(ee, img, names, sub.iloc[:h], ts), fetch(ee, img, names, sub.iloc[h:], ts)])
                raise SystemExit("Out of memory even for 1 band and 1 pixel.")
            if any(k in str(e).lower() for k in FATAL):
                raise SystemExit(f"Earth Engine rejected the request (retrying cannot help): {str(e)[:400]}")
            print(f"   attempt {attempt + 1} failed: {str(e)[:200]}")
            time.sleep(5 * (attempt + 1))
    raise SystemExit("Earth Engine sampling failed 3 times. See the lines above.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--sensor", choices=["s1", "s2"], default="s1")
    ap.add_argument("--smooth", type=int, default=0, help="s1 only: spatial mean radius in pixels (1 = 3x3, 2 = 5x5)")
    ap.add_argument("--mt", type=int, default=0, help="s1 only: Quegan-Yu multi-temporal speckle filter, window radius in pixels (1 = 3x3, 2 = 5x5)")
    ap.add_argument("--pass", dest="pass_", choices=["asc", "desc"], default="asc", help="s1 only: which orbit direction")
    ap.add_argument("--start", default=study.SEASON_START)
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=0, help="pixels per request (default 2000 s1, 1000 s2)")
    ap.add_argument("--group", type=int, default=0, help="bands per request (default all for s1, 5 for s2)")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    a.chunk = a.chunk or (1000 if a.sensor == "s2" else 2000)
    if (a.smooth or a.mt or a.pass_ != "asc") and a.sensor == "s2":
        raise SystemExit("--smooth, --mt and --pass apply to Sentinel-1 only")
    if a.smooth and a.mt:
        raise SystemExit("Use either --smooth or --mt, not both")
    data = Path(a.data)
    pix = pd.read_csv(data / "phase1" / "pixels.csv", dtype={"chip": str})
    pix["pid"] = pix.index                      # stable id = row number of pixels.csv
    orb = pd.read_csv(data / "phase1" / "chip_orbits.csv", dtype={"chip": str}).set_index("chip")
    if a.pass_ == "desc":                       # choose a descending orbit per chip from the coverage counts
        import p1_coverage
        counts = pd.read_csv(data / "phase1" / "orbit_counts.csv", dtype={"chip": str}).set_index("chip")
        orb = p1_coverage.choose(counts, prefer="DESC", fallback=False)   # strictly descending: thin chips are dropped
    weak = set(orb.index[orb.weak])
    n0 = len(pix)
    pix = pix[~pix.chip.isin(weak)]
    print(f"pixels: {n0}, dropped in {len(weak)} weak-coverage chips: {n0 - len(pix)}")
    if a.pilot:
        fs = pd.read_csv(data / "phase1" / "fields_study.csv")
        pix = pix[pix.field.isin(pick_pilot(fs, a.pilot).field)]
        assert (pix.split == "train").all()
    orb = orb[~orb.weak]
    if a.sensor == "s2":                                  # optical has no orbits: one group for all pixels
        orb = pd.DataFrame({"orbit": "S2"}, index=pix.chip.unique())
    chunks = plan(pix, orb, a.chunk)
    box = [pix.lon.min() - .05, pix.lat.min() - .05, pix.lon.max() + .05, pix.lat.max() + .05]
    out = data / "phase2" / (a.start + ("_s2" if a.sensor == "s2" else "") + ("_desc" if a.pass_ == "desc" else "") + (f"_sm{a.smooth}" if a.smooth else "") + (f"_mt{a.mt}" if a.mt else "") + ("_pilot" if a.pilot else ""))
    out.mkdir(parents=True, exist_ok=True)
    plan_hash = hashlib.md5((",".join(map(str, pix.pid)) + "|".join(orb.orbit)).encode()).hexdigest()
    manifest_check(out, dict(sensor=a.sensor, smooth=a.smooth, mt=a.mt, pass_=a.pass_, start=a.start, bin_days=study.BIN_DAYS, n_bins=study.N_BINS, chunk=a.chunk,
                             pilot=a.pilot, plan=plan_hash))
    print(f"{len(pix)} pixels in {len(chunks)} chunks -> {out}")
    names = features.ndvi_cols() if a.sensor == "s2" else sum(features.band_cols(), [])
    group = a.group or (5 if a.sensor == "s2" else len(names))
    ts = 16 if a.sensor == "s2" else 4
    ee, imgs = None, {}
    for n, (o, k, sub) in enumerate(chunks, 1):
        f = out / f"chunk_{o}_{k:04d}.csv"
        if f.exists():
            continue
        if ee is None:
            ee = study.gee_init(a.project)
        t = time.time()
        if o not in imgs:
            imgs[o] = s2_image(ee, a.start, box) if a.sensor == "s2" else orbit_image(ee, o, a.start, a.smooth, a.mt)
        img = imgs[o]
        df = pd.concat([fetch(ee, img, names[i:i + group], sub, ts) for i in range(0, len(names), group)], axis=1)
        df.insert(0, "orbit", o)
        df.to_csv(f.with_suffix(".tmp"))
        os.replace(f.with_suffix(".tmp"), f)
        print(f"{n}/{len(chunks)} {o} chunk {k}: {len(df)} px, {time.time() - t:.0f}s, "
              f"empty-window share {df[names].isna().mean().mean():.1%}")
    parts = [pd.read_csv(out / f"chunk_{o}_{k:04d}.csv") for o, k, _ in chunks]
    feat = pix.merge(pd.concat(parts), on="pid", how="left", validate="one_to_one")
    if len(feat) != len(pix) or feat[names].isna().all(axis=1).sum() == len(feat):
        raise SystemExit("Merged table is inconsistent")
    feat.to_csv(out / "features.csv", index=False)
    dead = int(feat[names].isna().all(axis=1).sum())
    print(f"done: {len(feat)} rows, {dead} pixels with no valid window at all (dropped later by features.drop_empty)")


if __name__ == "__main__":
    main()
