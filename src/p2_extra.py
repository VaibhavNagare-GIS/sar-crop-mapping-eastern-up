"""Phase 2c: extra per-pixel features from Earth Engine on the SAME pixel table, with the same chunked, resumable sampling as p2_extract.
python p2_extra.py angle --project P [--pass asc|desc]  -> data/phase2/angle_<pass>.csv  (pid, ANG = mean incidence angle in degrees over the season)
python p2_extra.py embed --project P                    -> data/phase2/embed.csv        (pid, A00_2021 .. A63_2022: Google satellite embeddings, benchmark only)
Safe to re-run: finished chunks are skipped."""
import argparse
import os
from pathlib import Path

import pandas as pd

import p2_extract as pe
import study

EMBED_YEARS = (2021, 2022)   # calendar-year layers that together cover Rabi 2021-22


def angle_image(ee, orbit, start):
    pas, rel = {"ASC": "ASCENDING", "DESC": "DESCENDING"}[orbit.split("_")[0]], int(orbit.split("_")[1])
    col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterDate(start, study.season_end(start))
           .filter(ee.Filter.eq("instrumentMode", "IW")).filter(ee.Filter.eq("orbitProperties_pass", pas))
           .filter(ee.Filter.eq("relativeOrbitNumber_start", rel)))
    return col.select("angle").mean().rename("ANG").unmask(pe.NODATA)


def embed_image(ee):
    col = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    return ee.Image.cat([col.filterDate(f"{y}-01-01", f"{y + 1}-01-01").mosaic().rename([f"A{i:02d}_{y}" for i in range(64)])
                         for y in EMBED_YEARS]).unmask(pe.NODATA)


def sample(ee, pix, orb, make_img, names, folder, group, chunk=2000, ts=4):
    """Bands `names` of make_img(orbit) at every pixel -> table indexed by pid. Fails loudly if any pixel is missing."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    imgs, parts = {}, []
    for o, k, sub in pe.plan(pix, orb, chunk):
        f = folder / f"chunk_{o}_{k:04d}.csv"
        if not f.exists():
            if o not in imgs:
                imgs[o] = make_img(o)
            df = pd.concat([pe.fetch(ee, imgs[o], names[i:i + group], sub, ts) for i in range(0, len(names), group)], axis=1)
            df.to_csv(f.with_suffix(".tmp"))
            os.replace(f.with_suffix(".tmp"), f)
            print(f"{o} chunk {k}: {len(df)} px")
        parts.append(pd.read_csv(f, index_col="pid"))
    res = pd.concat(parts).sort_index()
    if set(res.index) != set(pix.pid) or res.index.duplicated().any():
        raise SystemExit(f"{folder} does not match the pixel table (old chunks from another plan?). Delete the folder and rerun.")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["angle", "embed"])
    ap.add_argument("--project", required=True)
    ap.add_argument("--pass", dest="pass_", choices=["asc", "desc"], default="asc")
    ap.add_argument("--start", default=study.SEASON_START)
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    data = Path(a.data)
    pix = pd.read_csv(data / "phase1" / "pixels.csv", dtype={"chip": str})
    pix["pid"] = pix.index
    if a.what == "angle":
        if a.pass_ == "desc":
            import p1_coverage
            orb = p1_coverage.choose(pd.read_csv(data / "phase1" / "orbit_counts.csv", dtype={"chip": str}).set_index("chip"),
                                     prefer="DESC", fallback=False)
        else:
            orb = pd.read_csv(data / "phase1" / "chip_orbits.csv", dtype={"chip": str}).set_index("chip")
        weak = set(orb.index[orb.weak])
        pix, orb = pix[~pix.chip.isin(weak)], orb[~orb.weak]
        print(f"{len(pix)} pixels ({len(weak)} weak chips skipped, same rule as p2_extract)")
        ee = study.gee_init(a.project)
        res = sample(ee, pix, orb, lambda o: angle_image(ee, o, a.start), ["ANG"], data / "phase2" / f"angle_{a.pass_}_chunks", 1)
        res.to_csv(data / "phase2" / f"angle_{a.pass_}.csv")
        print(f"ANG degrees: min {res.ANG.min():.1f} max {res.ANG.max():.1f} NaN {int(res.ANG.isna().sum())}")
    else:
        orb = pd.DataFrame({"orbit": "EMB"}, index=pix.chip.unique())
        ee = study.gee_init(a.project)
        names = [f"A{i:02d}_{y}" for y in EMBED_YEARS for i in range(64)]
        res = sample(ee, pix, orb, lambda o: embed_image(ee), names, data / "phase2" / "embed_chunks", 16)
        res.to_csv(data / "phase2" / "embed.csv")
        print(f"embedding table {res.shape}, NaN cells {int(res.isna().sum().sum())}")


if __name__ == "__main__":
    main()
