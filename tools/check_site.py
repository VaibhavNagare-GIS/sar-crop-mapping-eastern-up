"""Run before every push: python tools/check_site.py
Everything under docs/ is what GitHub Pages serves, so every file there must be committed. This lists files that are ignored by .gitignore,
untracked, or changed but not committed, and checks the files the page needs on load exist. Exit code 1 means the live site would be broken."""
import subprocess
import sys
from pathlib import Path

NEEDED = ["docs/index.html", "docs/data/meta.json", "docs/data/signatures.json"]


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.splitlines()


def problems():
    out = [f"missing file: {f}" for f in NEEDED if not Path(f).is_file()]
    out += [f"ignored by .gitignore: {f}" for f in git("ls-files", "--others", "--ignored", "--exclude-standard", "--directory", "docs")]
    out += [f"not added to git: {f}" for f in git("ls-files", "--others", "--exclude-standard", "--directory", "docs")]
    out += [f"changed but not committed: {f}" for f in sorted(set(git("diff", "--name-only", "HEAD", "--", "docs")))]
    return out


if __name__ == "__main__":
    bad = problems()
    print("\n".join(bad[:25]) + (f"\n... and {len(bad) - 25} more" if len(bad) > 25 else "") if bad else "OK: everything the site needs is committed")
    sys.exit(1 if bad else 0)
