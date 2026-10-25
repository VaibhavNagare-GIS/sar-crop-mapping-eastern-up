"""Run: python tests/selftest_export.py -> 'SELFTEST OK'. export_results.py must turn NaN into null so browsers can parse the scores."""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("export_results", ROOT / "tools" / "export_results.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)

tmp = Path(tempfile.mkdtemp())
src, out = tmp / "results", tmp / "docs"
src.mkdir()
for area in ex.AREAS:
    (src / f"final_{area}.json").write_text(json.dumps({"n_fields": 3, "pixel_raw": {"per_class": {"Sugarcane": {"AUC": float("nan"), "F1": 0.5}}, "ci": {"f1": [0.1, float("nan")]}}}))
    (src / f"confusion_final_{area}_pixel_raw.csv").write_text(",A,B\nA,1,0\nB,0,1\n")
(src / "FINAL_CONFIG.json").write_text("{}")
sys.argv = ["export_results.py", "--src", str(src), "--out", str(out)]
ex.main()
assert "NaN" in (src / "final_far.json").read_text(), "the source files must stay untouched"
d = json.loads((out / "final_far.json").read_text(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))  # strict parse, like a browser
assert d["pixel_raw"]["per_class"]["Sugarcane"] == {"AUC": None, "F1": 0.5} and d["pixel_raw"]["ci"]["f1"] == [0.1, None]
assert sorted(p.name for p in out.iterdir()) == sorted([f"final_{a}.json" for a in ex.AREAS] + [f"confusion_final_{a}_pixel_raw.csv" for a in ex.AREAS])
print("SELFTEST OK")
