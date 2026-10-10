"""Run every self-test (synthetic data and fake Earth Engine only; no real data touched). Usage: python tests/run_all.py
Every test must print SELFTEST OK. Run after any code change and before every commit."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
jobs = [[str(p)] for p in sorted((ROOT / "tests").glob("selftest_*.py"))] + [[str(p), "--selftest"] for p in sorted((ROOT / "src").glob("run_*.py"))]
bad = []
for cmd in jobs:
    r = subprocess.run([sys.executable, *cmd], capture_output=True, text=True, cwd=ROOT)
    ok = r.returncode == 0 and r.stdout.strip().endswith("SELFTEST OK")
    print(("ok   " if ok else "FAIL ") + Path(cmd[0]).name)
    if not ok:
        bad.append(cmd[0])
        print((r.stdout + r.stderr)[-800:])
print("ALL SELFTESTS OK" if not bad else f"{len(bad)} FAILED")
sys.exit(1 if bad else 0)
