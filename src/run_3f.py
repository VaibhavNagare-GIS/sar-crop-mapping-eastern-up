"""Phase 3F runbook: label-season test, SAR feature ladder, embedding benchmark. Stops at the first failure; extraction is resumable.
Usage: python run_3f.py [extract|compare|all] [--project vaibhav-gee] [--dry]
Run order: python run_3e.py extract  ->  python run_3f.py extract  ->  python run_3f.py compare  ->  python run_3e.py compare"""
import sys

from run_3b import run

D = "data/phase2/2021-10-01"
KH, ZA = "data/phase2/2021-06-01", "data/phase2/2022-03-01"      # Kharif-to-early-Rabi window and Zaid window, 20 x 12 days each
AA, AD = "data/phase2/angle_asc.csv", "data/phase2/angle_desc.csv"


def steps(project):
    ex = [["p2_extract.py", "--project", project, "--start", "2021-06-01"],
          ["p2_extract.py", "--project", project, "--start", "2022-03-01"],
          ["p2_extra.py", "angle", "--project", project, "--pass", "asc"],
          ["p2_extra.py", "angle", "--project", project, "--pass", "desc"],
          ["p2_extra.py", "embed", "--project", project]]
    cmp_ = [["p3_sweep.py", "--classes", "3,4", "--leaves", "10", D, KH, ZA],
            ["p3_feat.py", D, "--angle", AA],
            ["p3_feat.py", D + "+" + D + "_desc", "--angle", AA + "," + AD, "--embed", "data/phase2/embed.csv"]]
    return {"extract": ex, "compare": cmp_, "all": ex + cmp_}


def main(argv):
    if "--selftest" in argv:
        s = steps("p")["all"]
        assert len(s) == 8 and all(c[0].endswith(".py") for c in s) and s[2][1] == "angle"
        print("SELFTEST OK")
        return 0
    what = next((a for a in argv if a in ("extract", "compare", "all")), "all")
    project = argv[argv.index("--project") + 1] if "--project" in argv else "vaibhav-gee"
    return run(steps(project)[what], dry="--dry" in argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
