"""Runbook: prepare an extra evaluation set (cells that were never used for training). Stops at the first failure; downloads are resumable.
Usage: python src/run_7.py [--set _near|_far] [--project vaibhav-gee] [--dry]"""
import sys

from run_3b import run


def steps(set_, project):
    return ([["p1_pixels.py", "data", "--set", set_],
             ["p1_coverage.py", "--project", project, "--set", set_]]
            + [["p5_chip_features.py", "download", "--project", project, "--pass", p, "--mt", "2", "--workers", "3", "--set", set_] for p in ("asc", "desc")]
            + [["p5_dense_table.py", "--pass", p, "--mt", "2", "--set", set_] for p in ("asc", "desc")])


def main(argv):
    if "--selftest" in argv:
        s = steps("_near", "p")
        assert len(s) == 6 and s[0][-1] == "_near" and s[2][5] == "asc" and s[-1][-1] == "_near"
        print("SELFTEST OK")
        return 0
    set_ = argv[argv.index("--set") + 1] if "--set" in argv else "_near"
    project = argv[argv.index("--project") + 1] if "--project" in argv else "vaibhav-gee"
    return run(steps(set_, project), dry="--dry" in argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
