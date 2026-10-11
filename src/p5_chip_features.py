"""Phase 5a: dense Sentinel-1 feature stacks on each chip's own 10 m grid (for the map and for spatial-context models).
python p5_chip_features.py parity   --project P [--pass asc|desc] [--mt 2] [--n 3]       compare with the sampled training features
python p5_chip_features.py download --project P [--pass asc|desc] [--mt 2] [--workers 3 --limit N --set _near] all chips of the study block (or of a --set), resumable
Output: data/phase5<set>/dense_<pass>_mt<k>/<chip>.npy = int16 (bands, h, w), dB x 100, -32768 = no data, band order VV_00..VV_19, VH_00..VH_19."""
import argparse
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import env_fix  # noqa: F401  (must come before rasterio)
import rasterio

import p0_inventory as inv
import p2_extract as pe
import study

NAMES = [f"{p}_{i:02d}" for p in ("VV", "VH") for i in range(study.N_BINS)]
NO = -32768


def grid(label_tif):
    with rasterio.open(label_tif) as a:
        epsg, t = a.crs.to_epsg() if a.crs else None, a.transform
        if epsg is None:
            raise SystemExit(f"{label_tif}: no EPSG code")
        return dict(w=a.width, h=a.height, crs=f"EPSG:{epsg}",
                    aff=dict(scaleX=t.a, shearX=t.b, translateX=t.c, shearY=t.d, scaleY=t.e, translateY=t.f))


def fetch_chip(ee, img, g, names):
    """(len(names), h, w) float32 for one chip grid; NODATA -> NaN. Memory-type errors split the bands; other errors retry 3 times."""
    for attempt in range(3):
        try:
            req = dict(expression=img.select(names), fileFormat="NUMPY_NDARRAY",
                       grid=dict(dimensions=dict(width=g["w"], height=g["h"]), affineTransform=g["aff"], crsCode=g["crs"]))
            arr = ee.data.computePixels(req)
            break
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ("memory", "too large", "exceed", "limit")) and len(names) > 1:
                h = len(names) // 2
                return np.concatenate([fetch_chip(ee, img, g, names[:h]), fetch_chip(ee, img, g, names[h:])])
            if any(k in msg for k in pe.FATAL):
                raise SystemExit(f"Earth Engine rejected the request (retrying cannot help): {str(e)[:400]}")
            if attempt == 2:
                raise SystemExit(f"computePixels failed 3 times: {str(e)[:400]}")
            time.sleep(5 * (attempt + 1))
    if isinstance(arr, (bytes, bytearray)):
        arr = np.load(io.BytesIO(arr))
    out = np.stack([np.asarray(arr[n], dtype=np.float32) for n in names])
    if out.shape != (len(names), g["h"], g["w"]):
        raise SystemExit(f"unexpected shape {out.shape}, wanted {(len(names), g['h'], g['w'])}")
    out[out == pe.NODATA] = np.nan
    return out


def quantise(x):
    return np.where(np.isnan(x), NO, np.clip(np.rint(x * 100), -32000, 32000)).astype(np.int16)


def dequantise(q):
    return np.where(q == NO, np.nan, q / 100.0).astype(np.float32)


def paired(dense, csv, r, c, shift=(0, 0)):
    """Dense values at the labelled pixels (moved by `shift`) and the matching sampled rows; finite pairs only, as flat arrays."""
    rr, cc = r + shift[0], c + shift[1]
    ok = (rr >= 0) & (rr < dense.shape[1]) & (cc >= 0) & (cc < dense.shape[2])
    x, y = dense[:, rr[ok], cc[ok]].T, csv[ok]
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def stats(x, y):
    """Agreement between dense and sampled values: correlation, mean difference, mean absolute difference (dB), share within 0.5 dB."""
    if len(x) < 2:
        return None
    d = x - y
    return dict(r=float(np.corrcoef(x, y)[0, 1]), mean=float(d.mean()), mad=float(np.abs(d).mean()), within=float((np.abs(d) < 0.5).mean()), n=len(x))


def compare(dense, csv, r, c, shift=(0, 0)):
    return stats(*paired(dense, csv, r, c, shift))


def orbit_table(data, pass_, set_=""):
    if pass_ == "desc":
        import p1_coverage
        orb = p1_coverage.choose(pd.read_csv(data / f"phase1{set_}" / "orbit_counts.csv", dtype={"chip": str}).set_index("chip"), prefer="DESC", fallback=False)
    else:
        orb = pd.read_csv(data / f"phase1{set_}" / "chip_orbits.csv", dtype={"chip": str}).set_index("chip")
    return orb[~orb.weak]


SHIFTS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))


def parity(ee, data, pass_, mt, n, start, min_px=100):
    """Compare dense chip features with the sampled training features. The decision uses all chosen chips pooled, because a chip with
    few labelled pixels gives unstable averages. Pass: best shift (0, 0), r >= 0.98, |mean difference| < 0.1 dB, MAD < 0.5 dB."""
    pix = pd.read_csv(data / "phase1" / "pixels.csv", dtype={"chip": str})
    pix["pid"] = pix.index
    orb = orbit_table(data, pass_)
    folder = f"{start}{'_desc' if pass_ == 'desc' else ''}{f'_mt{mt}' if mt else ''}"
    feats = pd.read_csv(data / "phase2" / folder / "features.csv", index_col="pid")
    files = inv.find_chips(data / "train_labels")
    count = pix.chip.value_counts()
    chips = sorted(c for c in set(pix.chip) & set(orb.index) if count[c] >= min_px)
    if len(chips) < n:
        raise SystemExit(f"only {len(chips)} chips have {min_px} or more labelled pixels")
    pick = list(np.random.default_rng(0).choice(chips, n, replace=False))
    imgs, pooled, secs = {}, {s: ([], []) for s in SHIFTS}, []
    for cid in pick:
        o = orb.loc[cid, "orbit"]
        if o not in imgs:
            imgs[o] = pe.orbit_image(ee, o, start, mt=mt)
        t0 = time.time()
        D = fetch_chip(ee, imgs[o], grid(files[cid][1]), NAMES)
        secs.append(time.time() - t0)
        px = pix[pix.chip == cid]
        X, r, c = feats.loc[px.pid, NAMES].to_numpy(), px.row.to_numpy(), px.col.to_numpy()
        for s_ in SHIFTS:
            x, y = paired(D, X, r, c, s_)
            pooled[s_][0].append(x)
            pooled[s_][1].append(y)
        st = stats(*paired(D, X, r, c))
        print(f"chip {cid} {o}: {secs[-1]:.0f}s, {len(px)} labelled px | r {st['r']:.4f} mean diff {st['mean']:+.3f} dB MAD {st['mad']:.3f} dB within 0.5 dB {st['within']:.1%}")
    res = {s_: stats(np.concatenate(v[0]), np.concatenate(v[1])) for s_, v in pooled.items()}
    for s_, st in res.items():
        print(f"pooled shift {s_}: r {st['r']:.4f} mean {st['mean']:+.3f} dB MAD {st['mad']:.3f} dB n {st['n']}")
    best = min(res, key=lambda k: res[k]["mad"])
    p0 = res[(0, 0)]
    ok = best == (0, 0) and p0["r"] >= 0.98 and abs(p0["mean"]) < 0.1 and p0["mad"] < 0.5
    print(f"\nPARITY {'PASS' if ok else 'FAIL'} ({pass_}); {np.mean(secs):.0f} s per chip")
    return ok


def download(ee, data, pass_, mt, workers, start, limit=0, set_=""):
    pix = pd.read_csv(data / f"phase1{set_}" / "pixels.csv", dtype={"chip": str})
    orb = orbit_table(data, pass_, set_)
    chips = sorted(set(pix.chip) & set(orb.index))
    files = inv.find_chips(data / "train_labels")
    out = data / f"phase5{set_}" / f"dense_{pass_}_mt{mt}"
    out.mkdir(parents=True, exist_ok=True)
    imgs = {o: pe.orbit_image(ee, o, start, mt=mt) for o in orb.loc[chips, "orbit"].unique()}
    todo = [c for c in chips if not (out / f"{c}.npy").exists()]
    todo = todo[:limit] if limit else todo
    print(f"{len(chips)} chips, {len(todo)} to do -> {out}")

    def one(cid):
        t0 = time.time()
        D = fetch_chip(ee, imgs[orb.loc[cid, "orbit"]], grid(files[cid][1]), NAMES)
        tmp = out / f"{cid}.tmp.npy"
        np.save(tmp, quantise(D))
        os.replace(tmp, out / f"{cid}.npy")
        return cid, float(np.isnan(D).mean()), time.time() - t0

    done = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, c) for c in todo]
        for f in as_completed(futs):
            try:
                cid, nan, sec = f.result()
            except BaseException:
                for g in futs:
                    g.cancel()
                raise
            done += 1
            print(f"{done}/{len(todo)} chip {cid}: {sec:.0f}s, no-data share {nan:.2%}")
    print("done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["parity", "download"])
    ap.add_argument("--project", required=True)
    ap.add_argument("--pass", dest="pass_", choices=["asc", "desc"], default="asc")
    ap.add_argument("--mt", type=int, default=2)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="download only the first N missing chips (for timing and cost checks)")
    ap.add_argument("--set", dest="set_", default="", choices=sorted(study.SETS))
    ap.add_argument("--start", default=study.SEASON_START)
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    env_fix.check()
    ee = study.gee_init(a.project)
    if a.what == "parity":
        raise SystemExit(0 if parity(ee, Path(a.data), a.pass_, a.mt, a.n, a.start) else 1)
    download(ee, Path(a.data), a.pass_, a.mt, a.workers, a.start, a.limit, a.set_)


if __name__ == "__main__":
    main()
