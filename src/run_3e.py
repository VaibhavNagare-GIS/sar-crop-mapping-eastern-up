"""Phase 3E runbook (next experiments). Stops at the first failure; extraction is resumable.
Usage: python run_3e.py [extract|compare|all] [--project vaibhav-gee] [--dry]
Questions answered: do two orbits plus optical beat either alone, do 3 classes work better than 4, does the
multi-temporal filter help the PIXEL level that the map needs?"""
import sys

from run_3b import run

D = "data/phase2/2021-10-01"
A, A2, DS, DS2, S = D, D + "_mt2", D + "_desc", D + "_desc_mt2", D + "_s2"


def steps(project):
    ex = [["p2_extract.py", "--project", project, "--start", "2021-10-01", "--pass", "desc", "--mt", "2"]]
    cmp_ = [
        ["p3_sweep.py", "--classes", "3,4", "--leaves", "5,10", A + "+" + DS, A + "+" + DS + "+" + S, DS + "+" + S],
        ["p3_sweep.py", "--classes", "3,4", "--leaves", "10", "--kind", "smooth", A + "+" + DS, A + "+" + DS + "+" + S],
        ["p3_sweep.py", "--level", "pixel", "--seeds", "2", "--classes", "3,4", "--leaves", "10",
         A + "+" + DS, A2 + "+" + DS2, A2 + "+" + DS2 + "+" + S],
    ]
    return {"extract": ex, "compare": cmp_, "all": ex + cmp_}


def main(argv):
    if "--selftest" in argv:
        s = steps("p")["all"]
        assert len(s) == 4 and s[0][-4:] == ["--pass", "desc", "--mt", "2"] and all(c[0].endswith(".py") for c in s)
        print("SELFTEST OK")
        return 0
    what = next((a for a in argv if a in ("extract", "compare", "all")), "all")
    project = argv[argv.index("--project") + 1] if "--project" in argv else "vaibhav-gee"
    return run(steps(project)[what], dry="--dry" in argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
