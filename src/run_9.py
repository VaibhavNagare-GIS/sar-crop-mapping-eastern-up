"""Runbook: Sentinel-2 NDVI on the dense chips as a companion layer (the Sentinel-1 map stays the primary product).
Usage: python src/run_9.py pilot|download|build|compare|all [--set _near|_far] [--project vaibhav-gee] [--dry]
pilot:    10 chips, to read the time and the Earth Engine cost before the full download
download: NDVI for every chip of the set (resumable)
build:    training table from the dense NDVI chips
compare:  pixel-level train CV of Sentinel-1 against Sentinel-1 + Sentinel-2 on the same pixels (study block only)"""
import sys

from run_3b import run

BASE = "data/phase2/dense_asc_mt2+data/phase2/dense_desc_mt2"


def steps(set_, project):
    dl = ["p5_chip_features.py", "download", "--project", project, "--sensor", "s2", "--workers", "3"] + (["--set", set_] if set_ else [])
    tb = ["p5_dense_table.py", "--sensor", "s2"] + (["--set", set_] if set_ else [])
    cmp_ = [["p3_sweep.py", "--level", "pixel", "--seeds", "2", "--classes", "3", "--leaves", "20", "--common", BASE, f"{BASE}+data/phase2/dense_s2"]]
    return {"pilot": [dl + ["--limit", "10"]], "download": [dl], "build": [tb], "compare": cmp_ if not set_ else [], "all": [dl, tb] + (cmp_ if not set_ else [])}


def main(argv):
    if "--selftest" in argv:
        s = steps("", "p")
        assert s["pilot"][0][-2:] == ["--limit", "10"] and s["all"][1][0] == "p5_dense_table.py" and len(s["all"]) == 3 and "--common" in s["compare"][0]
        assert steps("_far", "p")["download"][0][-2:] == ["--set", "_far"] and not steps("_far", "p")["compare"]
        print("SELFTEST OK")
        return 0
    what = next((a for a in argv if a in ("pilot", "download", "build", "compare", "all")), "all")
    set_ = argv[argv.index("--set") + 1] if "--set" in argv else ""
    project = argv[argv.index("--project") + 1] if "--project" in argv else "vaibhav-gee"
    return run(steps(set_, project)[what], dry="--dry" in argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
