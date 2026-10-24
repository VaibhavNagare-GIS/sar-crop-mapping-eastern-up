"""Run: python tests/selftest_phase8b.py -> 'SELFTEST OK'. Signature export (train rows only, gap handling) and the batched push tool on a local bare remote."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

import p8_signatures as sg
import study

rng = np.random.default_rng(0)
tmp = Path(tempfile.mkdtemp())
n, rows = study.N_BINS, []
for crop, label, base in (("Wheat", 1, -10.0), ("Mustard", 2, -14.0), ("Sugarcane", 6, -8.0)):
    for f in range(6):
        for _ in range(5):
            vv = base + rng.normal(0, 0.5, n)
            rows.append({"chip": "c1", "field": 100 * label + f, "label": label, "crop": crop, "split": "train", "block": f % 3, **{f"VV_{i:02d}": vv[i] for i in range(n)}, **{f"VH_{i:02d}": vv[i] - 6 for i in range(n)}})
rows.append({"chip": "c1", "field": 9999, "label": 1, "crop": "Wheat", "split": "test", "block": 0, **{f"VV_{i:02d}": 50.0 for i in range(n)}, **{f"VH_{i:02d}": 50.0 for i in range(n)}})  # must be ignored
df = pd.DataFrame(rows)
df.loc[3, "VV_05"] = np.nan  # one gap, filled by interpolation
folder = tmp / "data" / "phase2" / "2021-10-01"
folder.mkdir(parents=True)
df.to_csv(folder / "features.csv", index=False)
sys.argv = ["p8_signatures.py", "--folder", str(folder), "--out", str(tmp / "docs")]
sg.main()
j = json.loads((tmp / "docs" / "data" / "signatures.json").read_text())
assert j["windows"][0] == "2021-10-01" and len(j["windows"]) == n and j["unit"] == "dB"
assert set(j["series"]) == {"VV", "VH", "VHVV"} and set(j["series"]["VV"]) == {"Wheat", "Mustard", "Sugarcane"}
assert j["counts"]["Wheat"] == {"pixels": 30, "fields": 6}, j["counts"]  # the test-split row is not counted
w = j["series"]["VV"]["Wheat"]
assert all(lo <= m <= hi for lo, m, hi in zip(w["lo"], w["med"], w["hi"])) and abs(np.mean(w["med"]) + 10) < 0.5
assert abs(np.mean(j["series"]["VHVV"]["Wheat"]["med"]) + 6) < 0.01

# batched push: 3 batches to a local bare remote, nothing outside --paths is touched
spec = importlib.util.spec_from_file_location("push_in_batches", ROOT / "tools" / "push_in_batches.py")
pb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pb)
run = lambda *a, cwd: subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout
bare, work = tmp / "remote.git", tmp / "work"
run("init", "-q", "--bare", str(bare), cwd=tmp)
run("clone", "-q", str(bare), str(work), cwd=tmp)
for k, v in (("user.email", "t@t"), ("user.name", "t")):
    run("config", k, v, cwd=work)
run("checkout", "-q", "-b", "main", cwd=work)
(work / "README").write_text("hi")
run("add", ".", cwd=work)
run("commit", "-qm", "init", cwd=work)
run("push", "-q", "origin", "main", cwd=work)
(work / "docs" / "tiles").mkdir(parents=True)
for i in range(9):
    (work / "docs" / "tiles" / f"t{i}.png").write_bytes(os.urandom(200_000))
(work / "local.txt").write_text("not pushed")
old = os.getcwd()
os.chdir(work)
sys.argv = ["push_in_batches.py", "--limit-mb", "0.5"]
pb.main()
os.chdir(old)
names = run("--git-dir", str(bare), "ls-tree", "-r", "--name-only", "main", cwd=tmp).split()
assert len([x for x in names if x.startswith("docs/tiles/")]) == 9 and "local.txt" not in names
assert len(run("--git-dir", str(bare), "log", "--oneline", "main", cwd=tmp).splitlines()) == 1 + 5, "2 files fit in each 0.5 MB batch, so 9 files make 5 batches"
print("SELFTEST OK")
