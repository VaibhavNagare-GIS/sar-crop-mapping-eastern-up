"""Copy the final scores to docs/data/results/ as strict JSON for the web page.
python tools/export_results.py [--src results] [--out docs/data/results]
Python writes NaN for undefined scores (for example the AUC of a class that is absent), but browsers cannot parse NaN, so each NaN becomes null.
The files in results/ are not changed."""
import argparse
import json
import math
import shutil
from pathlib import Path

AREAS = ("test", "near", "far")


def clean(x, count):
    if isinstance(x, float) and not math.isfinite(x):
        count[0] += 1
        return None
    if isinstance(x, dict):
        return {k: clean(v, count) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v, count) for v in x]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results")
    ap.add_argument("--out", default="docs/data/results")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    count = [0]
    for area in AREAS:
        f = src / f"final_{area}.json"
        if not f.is_file():
            raise SystemExit(f"Missing {f}. Run p7_eval.py for this area first.")
        (out / f.name).write_text(json.dumps(clean(json.loads(f.read_text()), count), allow_nan=False))
        for c in src.glob(f"confusion_final_{area}_*.csv"):
            shutil.copy(c, out / c.name)
    print(f"exported {len(AREAS)} areas to {out}; {count[0]} NaN value(s) became null")


if __name__ == "__main__":
    main()
