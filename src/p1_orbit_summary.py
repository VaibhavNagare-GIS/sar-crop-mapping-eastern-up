"""Which Sentinel-1 orbits could be added? Reads data/phase1/orbit_counts.csv (written by p1_coverage.py).
python src/p1_orbit_summary.py [--min 18] [--data data]
Prints, per orbit, the number of chips with at least --min scenes in the season, and how many chips have each combination of orbits."""
import argparse
from pathlib import Path

import pandas as pd


def summarise(counts, min_scenes):
    ok = counts >= min_scenes
    per_orbit = ok.sum().sort_values(ascending=False)
    combo = ok.apply(lambda r: "+".join(c for c in counts.columns if r[c]) or "none", axis=1).value_counts()
    return per_orbit, combo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=18)
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    c = pd.read_csv(Path(a.data) / "phase1" / "orbit_counts.csv", dtype={"chip": str}).set_index("chip")
    per, combo = summarise(c, a.min)
    print(f"chips with at least {a.min} scenes, per orbit (of {len(c)} chips):\n{per.to_string()}\n\nchips per orbit combination:\n{combo.to_string()}")
