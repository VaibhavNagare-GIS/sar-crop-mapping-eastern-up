"""Phase 7b: score the final map at labelled pixels, with field-level bootstrap intervals.
python src/p7_eval.py [--set _near|_far] [--tag _fused] [--boot 2000]
Study block: the TEST fields (a second look at them: the first look was the frozen Random Forest, so this is not pre-registered). _near and _far: every
field of the set (never used for training). Reads data/phase7/labelled<set>.csv from p7_map.py. Macro scores average over the classes that occur in
the evaluated fields (Sugarcane is absent from _far), while predictions of an absent class still count as errors for the others.
Reports the raw pixel model, the 5 x 5 smoothed map, and the smoothed probabilities averaged per field. Writes results/final_<name>.json and CSVs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import p4_final as p4
from p7_map import CLASSES

NAMES = {1: "Wheat", 2: "Mustard+Lentil", 6: "Sugarcane"}


def metrics(y, P):
    """y: class index 0..2, P: (n, 3). Macro scores over classes with support."""
    r = p4.core(y, P, np.arange(len(CLASSES)))
    present = r["cm"].sum(axis=1) > 0
    return dict(f1=float(r["f1c"][present].mean()), bal=float(r["pa"][present].mean()), oa=r["oa"],
                auc=float(np.nanmean(np.array(r["aucc"], dtype=float)[present])), cm=r["cm"], pa=r["pa"], ua=r["ua"], f1c=r["f1c"], aucc=r["aucc"])


def cluster_boot(y, P, units, draws):
    order = np.argsort(units, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(units[order])) + 1)
    out = np.empty((len(draws), 4))
    for b, d in enumerate(draws):
        idx = np.concatenate([groups[i] for i in d])
        m = metrics(y[idx], P[idx])
        out[b] = (m["f1"], m["bal"], m["oa"], m["auc"])
    return out


def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def evaluate(df, boot=2000, seed=0):
    """df columns: field, label (1, 2 or 6), raw1 raw2 raw6, sm1 sm2 sm6. Returns the result dictionary and confusion frames."""
    y = np.searchsorted(CLASSES, df.label.to_numpy())
    fields = pd.factorize(df.field.to_numpy())[0]
    draws = np.random.default_rng(seed).integers(0, fields.max() + 1, (boot, fields.max() + 1))
    base = float((y == np.bincount(y).argmax()).mean())
    res, cms = {"n_pixels": len(y), "n_fields": int(fields.max() + 1), "baseline_oa": base, "classes_present": [NAMES[CLASSES[i]] for i in np.unique(y)]}, {}
    fv = pd.DataFrame(df[[f"sm{k}" for k in CLASSES]].to_numpy()).groupby(fields).transform("mean").to_numpy()
    for name, P in (("pixel_raw", df[[f"raw{k}" for k in CLASSES]].to_numpy()), ("pixel_smoothed", df[[f"sm{k}" for k in CLASSES]].to_numpy()), ("smoothed_field_vote", fv)):
        m, b = metrics(y, P), cluster_boot(y, P, fields, draws)
        res[name] = {k: m[k] for k in ("f1", "bal", "oa", "auc")}
        res[name]["ci"] = {k: ci(b[:, i]) for i, k in enumerate(("f1", "bal", "oa", "auc"))}
        res[name]["per_class"] = {NAMES[c]: dict(PA=float(m["pa"][i]), UA=float(m["ua"][i]), F1=float(m["f1c"][i]), AUC=float(m["aucc"][i])) for i, c in enumerate(CLASSES)}
        cms[name] = pd.DataFrame(m["cm"], index=[NAMES[c] for c in CLASSES], columns=[NAMES[c] for c in CLASSES])
    return res, cms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_", default="", choices=["", "_near", "_far"])
    ap.add_argument("--tag", default="", choices=["", "_fused"])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    data = Path(a.data)
    pix = pd.read_csv(data / f"phase1{a.set_}" / "pixels.csv", dtype={"chip": str})
    pix["pid"] = pix.index
    lab = pd.read_csv(data / "phase7" / f"labelled{a.tag}{a.set_}.csv", index_col="pid").dropna()
    df = pix.set_index("pid").join(lab, how="inner")
    df = df[df.label != 4]
    df["label"] = df.label.where(df.label != 3, 2)
    if not a.set_:
        df = df[df.split == "test"]
    name = a.tag.strip("_") + ("_" if a.tag else "") + {"": "test", "_near": "near", "_far": "far"}[a.set_]
    res, cms = evaluate(df, a.boot)
    res["second_look"] = not a.set_
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"final_{name}.json").write_text(json.dumps(res, indent=2, default=float))
    for k, cm in cms.items():
        cm.to_csv(out / f"confusion_final_{name}_{k}.csv")
    print(f"{name}: {res['n_fields']} fields, {res['n_pixels']} pixels, classes {res['classes_present']}, always-biggest-class OA {res['baseline_oa']:.3f}")
    for k in ("pixel_raw", "pixel_smoothed", "smoothed_field_vote"):
        r = res[k]
        print(f"{k:20s} F1 {r['f1']:.3f} {np.round(r['ci']['f1'], 3).tolist()}  bal {r['bal']:.3f}  OA {r['oa']:.3f}  AUC {r['auc']:.3f} {np.round(r['ci']['auc'], 3).tolist()}")


if __name__ == "__main__":
    main()
