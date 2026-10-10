"""Phase 3B runbook: runs the remaining extractions and comparisons in order and STOPS at the first failure.
Usage: python run_3b.py [extract|compare|all] [--project vaibhav-gee] [--dry]
Why: typing each long command by hand (and forgetting 'python') was the only error in the last round.
Extractions are resumable, so re-running this after a stop is safe."""
import subprocess
from pathlib import Path
import sys

A = "data/phase2/2021-10-01"


def steps(project):
    ex = [["p2_extract.py", "--project", project, "--start", "2021-10-01", *o] for o in (["--mt", "2"], ["--pass", "desc"])]
    cmp_ = [
        # same-pixel comparison of speckle handling and orbit fusion (descending covers fewer chips)
        ["p3_sweep.py", "--common", "--leaves", "5,10,20", A, A + "_mt1", A + "_mt2", A + "_desc", A + "+" + A + "_desc"],
        ["p3_sweep.py", "--leaves", "10", "--kind", "summary", A, A + "_mt1", A + "_mt2"],
        ["p3_sweep.py", "--leaves", "10", "--kind", "smooth", A, A + "_mt1", A + "_mt2"],
        ["p3_sweep.py", "--leaves", "10", "--models", "et,hgb,lr", A, A + "_mt1"],
        # the map is pixel based, so check pixel level too (2 seeds to save time)
        ["p3_sweep.py", "--level", "pixel", "--seeds", "2", "--leaves", "10", A, A + "_mt1", A + "_mt2"],
    ]
    return {"extract": ex, "compare": cmp_, "all": ex + cmp_}


HERE = Path(__file__).resolve().parent


def run(cmds, dry=False):
    for i, c in enumerate(cmds, 1):
        print(f"\n=== step {i}/{len(cmds)}: python {' '.join(c)}", flush=True)
        if dry:
            continue
        cmd = [str(HERE / c[0]), *c[1:]] if c[0].endswith(".py") else c      # scripts live next to this file, wherever you run from
        if subprocess.run([sys.executable, *cmd]).returncode != 0:
            print(f"\nSTOPPED at step {i}. Fix the error above, then run this script again.")
            return 1
    return 0


def main(argv):
    if "--selftest" in argv:
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()) as buf:      # the simulated failure is expected: keep it off the screen
            assert run([["-c", "pass"], ["-c", "import sys; sys.exit(3)"], ["-c", "print('never runs')"]]) == 1
            assert run([["-c", "pass"]]) == 0
        assert "step 3/3" not in buf.getvalue()                      # step 3 must not have run
        s = steps("p")
        assert len(s["all"]) == 7 and s["all"][0][-2:] == ["--mt", "2"] and all(c[0].endswith(".py") for c in s["all"])
        print("SELFTEST OK")
        return 0
    what = next((a for a in argv if a in ("extract", "compare", "all")), "all")
    project = argv[argv.index("--project") + 1] if "--project" in argv else "vaibhav-gee"
    return run(steps(project)[what], dry="--dry" in argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
