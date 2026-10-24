"""Run: python tests/selftest_publish.py -> 'SELFTEST OK'. tools/check_site.py must catch the real mistake: an unanchored 'data/' rule in .gitignore hiding docs/data/."""
import importlib.util
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("check_site", ROOT / "tools" / "check_site.py")
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)

work = Path(tempfile.mkdtemp())
git = lambda *a: subprocess.run(["git", *a], cwd=work, check=True, capture_output=True, text=True)
git("init", "-q")
for k, v in (("user.email", "t@t"), ("user.name", "t")):
    git("config", k, v)
for f in ("docs/index.html", "docs/data/meta.json", "docs/data/signatures.json", "data/private.csv"):
    (work / f).parent.mkdir(parents=True, exist_ok=True)
    (work / f).write_text("x")
(work / ".gitignore").write_text("data/\n")
git("add", "-A")
git("commit", "-qm", "init")
os.chdir(work)
bad = cs.problems()
assert any("docs/data" in b and "ignored" in b for b in bad), bad  # the bug: docs/data/ is silently ignored and never reaches Pages
(work / ".gitignore").write_text("/data/\n")  # the fix: only the top-level data/ folder is private
git("add", "-A")
assert any("changed but not committed: docs/data/meta.json" == x for x in cs.problems()), "staged but not yet committed must be reported"
git("commit", "-qm", "fix")
tracked = git("ls-files").stdout.split()
assert cs.problems() == [] and "docs/data/meta.json" in tracked and "data/private.csv" not in tracked
(work / "docs" / "data" / "meta.json").write_text("changed")
assert any("changed but not committed" in b for b in cs.problems())
print("SELFTEST OK")
