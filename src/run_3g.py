"""Phase 3G runbook: window/season control, leaf check for the final choice. Stops at the first failure.
Usage: python run_3g.py [--dry] [--selftest]"""
import sys

from run_3b import run

A, D = "data/phase2/2021-10-01", "data/phase2/2021-10-01_desc"
A2, D2 = "data/phase2/2021-10-01_mt2", "data/phase2/2021-10-01_desc_mt2"
STEPS = [["p3_windows.py", A + "+" + D],
         ["p3_windows.py", "data/phase2/2021-06-01", "--ranges", "0:9"],                # control: Jun to mid Sep 2021, before Rabi sowing                                              # which windows carry the signal; earliest reliable date
         ["p3_windows.py", "data/phase2/2022-03-01", "--ranges", "0:20,7:20"],         # control: Kharif 2022 only (24 May to 26 Oct 2022)
         ["p3_sweep.py", "--classes", "3", "--leaves", "5,10,20", A + "+" + D],      # leaf choice for the field model
         ["p3_sweep.py", "--level", "pixel", "--seeds", "2", "--classes", "3", "--leaves", "5,10,20", A2 + "+" + D2]]   # leaf choice for the map model


def main(argv):
    if "--selftest" in argv:
        assert len(STEPS) == 5 and all(s[0].endswith(".py") for s in STEPS)
        print("SELFTEST OK")
        return 0
    return run(STEPS, dry="--dry" in argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
