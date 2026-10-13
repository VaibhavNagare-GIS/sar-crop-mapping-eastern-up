"""Runbook for the accuracy experiments on training data (class weights, map-level cross-validation). Stops at the first failure.
Usage: python src/run_6.py [--dry]"""
import sys

from run_3b import run

FIELD = "data/phase2/2021-10-01+data/phase2/2021-10-01_desc"
DENSE = "data/phase2/dense_asc_mt2+data/phase2/dense_desc_mt2"
STEPS = [["p6_weights.py", FIELD, "--level", "field", "--leaf", "10"],
         ["p6_weights.py", DENSE, "--level", "pixel", "--leaf", "20"],
         ["p6_densecv.py"]]


def main(argv):
    if "--selftest" in argv:
        assert len(STEPS) == 3 and all(s[0].endswith(".py") for s in STEPS)
        print("SELFTEST OK")
        return 0
    return run(STEPS, dry="--dry" in argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
