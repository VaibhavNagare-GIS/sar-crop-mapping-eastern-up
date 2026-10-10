"""Phase 1b: which Sentinel-1 orbit covers each study chip, and pick one orbit per chip.

Usage:  python p1_coverage.py --project vaibhav-gee [--data data]
Writes data/phase1/orbit_counts.csv and chip_orbits.csv.
Rule: use ascending orbits only (one look direction and pass time), take the ascending
orbit with the most valid scenes at ALL 5 sample points of the chip; fall back to descending
only when ascending is too thin, and flag it.
"""
import argparse
from pathlib import Path

import pandas as pd

import study

MIN_SCENES = 15  # my rule of thumb: about two thirds of the ~23 possible dates


def choose(counts, prefer="ASC", min_count=MIN_SCENES, fallback=True):
    """counts: DataFrame (index chip, columns like ASC_56, DESC_165) -> chosen orbit per chip."""
    pref = [c for c in counts.columns if c.startswith(prefer + "_")] or list(counts.columns)
    out = pd.DataFrame(index=counts.index)
    out["orbit"] = counts[pref].idxmax(axis=1)
    out["n_scenes"] = counts[pref].max(axis=1)
    best_all = counts.idxmax(axis=1)
    best_n = counts.max(axis=1)
    fb = (out.n_scenes < min_count) & (best_n > out.n_scenes) & fallback
    out.loc[fb, "orbit"] = best_all[fb]
    out.loc[fb, "n_scenes"] = best_n[fb]
    out["fallback"] = fb
    out["weak"] = out.n_scenes < min_count
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    data = Path(a.data)
    out = data / "phase1"
    out.mkdir(parents=True, exist_ok=True)
    chips = pd.read_csv(data / "inventory" / "chips.csv", dtype={"chip": str, "cell": str})
    chips = chips[chips.cell.isin(study.BLOCK_CELLS)].reset_index(drop=True)
    if chips.empty:
        raise SystemExit("No chips in the study cells. Run p0_inventory.py first.")
    ee = study.gee_init(a.project)
    pts = []
    for r in chips.itertuples():
        for k, (fx, fy) in enumerate([(.5, .5), (.25, .25), (.75, .25), (.25, .75), (.75, .75)]):
            pts.append(ee.Feature(ee.Geometry.Point([r.west + fx * (r.east - r.west), r.south + fy * (r.north - r.south)]),
                                  {"chip": r.chip, "k": k}))
    fc = ee.FeatureCollection(pts)
    box = ee.Geometry.Rectangle([chips.west.min() - .05, chips.south.min() - .05, chips.east.max() + .05, chips.north.max() + .05])
    col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(box).filterDate(study.SEASON_START, study.season_end())
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    rows = col.reduceColumns(ee.Reducer.toList(2), ["orbitProperties_pass", "relativeOrbitNumber_start"]).get("list").getInfo()
    orbits = sorted({(p, int(o)) for p, o in rows})
    if not orbits:
        raise SystemExit("No Sentinel-1 scenes found for the study block")
    short = {"ASCENDING": "ASC", "DESCENDING": "DESC"}
    imgs = [col.filter(ee.Filter.eq("orbitProperties_pass", p)).filter(ee.Filter.eq("relativeOrbitNumber_start", o))
            .select("VV").count().unmask(0).rename(f"{short[p]}_{o}") for p, o in orbits]  # unmask(0): "no scenes" must be 0, not masked
    got = ee.Image.cat(imgs).sampleRegions(collection=fc, properties=["chip", "k"], scale=30, tileScale=4).getInfo()["features"]
    names = [f"{short[p]}_{o}" for p, o in orbits]
    if not got:
        raise SystemExit("sampleRegions returned no points at all.")
    df = pd.DataFrame([f["properties"] for f in got])
    if not ({"chip", "k"} | set(names)) <= set(df.columns):
        raise SystemExit(f"Unexpected columns from Earth Engine: {list(df.columns)}")
    n_pts = df.groupby("chip").size()
    bad = n_pts[n_pts != 5]
    if len(bad) or set(df.chip) != set(chips.chip):
        raise SystemExit(f"Sampling returned incomplete points ({len(bad)} chips with != 5 points, "
                         f"{len(set(chips.chip) - set(df.chip))} chips missing).")
    counts = df.groupby("chip")[names].min()
    counts.to_csv(out / "orbit_counts.csv")
    ch = choose(counts)
    ch.to_csv(out / "chip_orbits.csv")
    print("orbits found:", names)
    print("chips per chosen orbit:\n", ch.orbit.value_counts().to_string())
    print(f"fallback to descending: {int(ch.fallback.sum())} chips,  weak (< {MIN_SCENES} scenes): {int(ch.weak.sum())} chips")
    print("n_scenes (chosen orbit) min/median/max:", int(ch.n_scenes.min()), int(ch.n_scenes.median()), int(ch.n_scenes.max()))


if __name__ == "__main__":
    main()
