"""Runbook: does a third Sentinel-1 orbit help? Downloads DESC_165 and ASC_158 on the chips that have them, builds tables and compares the
two-orbit model with the three-orbit model on the same pixels (train CV). Stops at the first failure; downloads are resumable.
Usage: python src/run_8.py download|build|compare|all [--project vaibhav-gee] [--dry]"""
import sys

from run_3b import run

BASE = "data/phase2/dense_asc_mt2+data/phase2/dense_desc_mt2"
EXTRA = ("DESC_165", "ASC_158")


def steps(project):
    d = [["p5_chip_features.py", "download", "--project", project, "--orbit", o, "--mt", "2", "--workers", "3"] for o in EXTRA]
    b = [["p5_dense_table.py", "--orbit", o, "--mt", "2"] for o in EXTRA]
    c = [["p3_sweep.py", "--level", "pixel", "--seeds", "2", "--classes", "3", "--leaves", "20", "--common", BASE, f"{BASE}+data/phase2/dense_{o}_mt2"] for o in EXTRA]
    return {"download": d, "build": b, "compare": c, "all": d + b + c}


def main(argv):
    if "--selftest" in argv:
        s = steps("p")["all"]
        assert len(s) == 6 and s[0][5] == "DESC_165" and s[-1][-1].endswith("dense_ASC_158_mt2") and "--common" in s[-1]
        print("SELFTEST OK")
        return 0
    what = next((a for a in argv if a in ("download", "build", "compare", "all")), "all")
    project = argv[argv.index("--project") + 1] if "--project" in argv else "vaibhav-gee"
    return run(steps(project)[what], dry="--dry" in argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
