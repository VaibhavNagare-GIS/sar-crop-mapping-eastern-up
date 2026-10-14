"""Phase 7a: the final pixel model and the map.
python src/p7_map.py [--set _near|_far] [--limit-chips N] [--retrain]
Trains a Random Forest on every TRAIN pixel of the study block (dense Sentinel-1 features, 3 classes), predicts every pixel of every chip of
the chosen set, smooths the class probabilities with a 5 x 5 mean filter, and writes per chip: <chip>_class.tif (1 Wheat, 2 Mustard/Lentil,
6 Sugarcane, 0 none) and <chip>_conf.tif (largest probability x 255) under data/phase7/map<set>/. It also writes data/phase7/labelled<set>.csv with the
raw and smoothed probabilities at every labelled pixel, which p7_eval.py scores. Settings are recorded in results/FINAL_CONFIG.json the first time and
must not change silently afterwards. Chips of the _far set whose ascending slot holds a descending orbit are skipped."""
import argparse
import datetime as dt
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

import env_fix  # noqa: F401  (before rasterio)
import rasterio

import features
import p0_inventory as inv
import p3_sweep as sw
import p4_final as p4
import p5_chip_features as cf
import p6_densecv as dc

CLASSES = dc.CLASSES
SMOOTH = 5


def train_final(parts, meta, leaf, trees, seed):
    """Random Forest on every pixel of (parts, meta) after the 3-class mapping. Returns the model and the number of pixels."""
    parts, meta = p4.prep(parts, meta, "3")
    X, y, _ = sw.build(parts, meta, "pixel", "raw")
    return features.make_model("rf", leaf, trees, seed).fit(X, np.searchsorted(CLASSES, y)), len(y)


def predict_maps(model, stack):
    """(bands, h, w) stack -> raw and smoothed class-probability maps (h, w, 3)."""
    h, w = stack.shape[1:]
    raw = model.predict_proba(dc.dense_features(stack)).reshape(h, w, -1)
    return raw, uniform_filter(raw, size=(SMOOTH, SMOOTH, 1), mode="nearest")


def class_conf(p):
    """Probability map -> class codes (uint8) and confidence (largest probability x 255, uint8)."""
    return CLASSES[p.argmax(-1)].astype(np.uint8), np.rint(p.max(-1) * 255).astype(np.uint8)


def write_tif(path, arr, label_tif):
    with rasterio.open(label_tif) as src:
        prof = src.profile
    prof.update(count=1, dtype="uint8", nodata=0, compress="lzw")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr[None])


def commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "no-git"


def record_config(path, cfg):
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text())
        same = all(old.get(k) == cfg[k] for k in ("leaf", "trees", "seed", "smooth", "classes", "features"))
        if not same:
            raise SystemExit(f"{path} holds different final settings; change them only deliberately (delete the file and say so in the report)")
        return old
    cfg = dict(cfg, created=dt.datetime.now().isoformat(timespec="seconds"), git_commit=commit())
    path.write_text(json.dumps(cfg, indent=2))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_", default="", choices=sorted(__import__("study").SETS))
    ap.add_argument("--leaf", type=int, default=20)
    ap.add_argument("--trees", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-chips", type=int, default=0)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    data = Path(a.data)
    record_config(Path("results") / "FINAL_CONFIG.json", dict(leaf=a.leaf, trees=a.trees, seed=a.seed, smooth=SMOOTH, classes="3",
                                                              features="dense_asc_mt2 + dense_desc_mt2, raw windows plus VH-VV"))
    mp = data / "phase7" / "model_rf.joblib"
    mp.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    if mp.exists() and not a.retrain:
        model = joblib.load(mp)
        print(f"loaded {mp}")
    else:
        parts, meta = sw.load(f"{data / 'phase2' / 'dense_asc_mt2'}+{data / 'phase2' / 'dense_desc_mt2'}")
        t0 = time.time()
        model, n = train_final(parts, meta, a.leaf, a.trees, a.seed)
        joblib.dump(model, mp, compress=3)
        print(f"trained on {n} train pixels in {time.time() - t0:.0f}s, saved {mp} ({mp.stat().st_size / 1e6:.0f} MB)")

    pix = pd.read_csv(data / f"phase1{a.set_}" / "pixels.csv", dtype={"chip": str})
    pix["pid"] = pix.index
    have = lambda n: {p.stem for p in (data / f"phase5{a.set_}" / n).glob("*.npy")}
    chips = sorted(set(pix.chip) & have("dense_asc_mt2") & have("dense_desc_mt2"))
    if a.set_:
        asc = cf.orbit_table(data, "asc", a.set_)
        ok = set(asc.index[asc.orbit.str.startswith("ASC")])
        print(f"{len(chips) - len([c for c in chips if c in ok])} chips skipped: their ascending slot holds a descending orbit")
        chips = [c for c in chips if c in ok]
    if a.limit_chips:
        chips = chips[:a.limit_chips]
    files = inv.find_chips(data / "train_labels")
    out = data / "phase7" / f"map{a.set_}"
    out.mkdir(parents=True, exist_ok=True)
    rows, t1 = [], time.time()
    for i, c in enumerate(chips, 1):
        stack = np.concatenate([cf.dequantise(np.load(data / f"phase5{a.set_}" / n / f"{c}.npy")) for n in ("dense_asc_mt2", "dense_desc_mt2")])
        raw, sm = predict_maps(model, stack)
        cls, conf = class_conf(sm)
        write_tif(out / f"{c}_class.tif", cls, files[c][1])
        write_tif(out / f"{c}_conf.tif", conf, files[c][1])
        px = pix[pix.chip == c]
        r, q = px.row.to_numpy(), px.col.to_numpy()
        rows.append(pd.DataFrame(np.hstack([raw[r, q], sm[r, q]]), index=px.pid, columns=[f"raw{k}" for k in CLASSES] + [f"sm{k}" for k in CLASSES]))
        if i % 25 == 0 or i == len(chips):
            print(f"{i}/{len(chips)} chips, {time.time() - t1:.0f}s, about {(time.time() - t1) / i * (len(chips) - i) / 60:.1f} min left")
    lab = pd.concat(rows)
    lab.index.name = "pid"
    lab.to_csv(data / "phase7" / f"labelled{a.set_}.csv")
    print(f"wrote {len(chips)} chips to {out} and {len(lab)} labelled pixels to labelled{a.set_}.csv")


if __name__ == "__main__":
    main()
