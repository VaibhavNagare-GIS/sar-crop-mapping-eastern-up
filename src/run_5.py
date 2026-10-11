"""Runbook for the dense chip stacks. Stops at the first failure; downloads are resumable.
Usage: python src/run_5.py download|build|compare|all [--project vaibhav-gee] [--dry]
download: dense Sentinel-1 stacks for every chip (ascending and descending, mt2 filter)
build:    training tables taken from those stacks
compare:  pixel-level train CV on the dense tables against the sampled features"""
import sys

from run_3b import run

SAMPLED = "data/phase2/2021-10-01_mt2+data/phase2/2021-10-01_desc_mt2"
DENSE = "data/phase2/dense_asc_mt2+data/phase2/dense_desc_mt2"


def steps(project):
    d = [["p5_chip_features.py", "download", "--project", project, "--pass", p, "--mt", "2", "--workers", "3"] for p in ("asc", "desc")]
    b = [["p5_dense_table.py", "--pass", p, "--mt", "2"] for p in ("asc", "desc")]
    c = [["p3_sweep.py", "--level", "pixel", "--seeds", "2", "--classes", "3", "--leaves", "20", SAMPLED, DENSE]]
    return {"download": d, "build": b, "compare": c, "all": d + b + c}


def main(argv):
    if "--selftest" in argv:
        s = steps("p")["all"]
        assert len(s) == 5 and s[0][5] == "asc" and s[1][5] == "desc" and s[-1][-1] == DENSE
        print("SELFTEST OK")
        return 0
    what = next((a for a in argv if a in ("download", "build", "compare", "all")), "all")
    project = argv[argv.index("--project") + 1] if "--project" in argv else "vaibhav-gee"
    return run(steps(project)[what], dry="--dry" in argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
