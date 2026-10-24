"""Push many new files to GitHub in small commits, so that no single push runs long enough to time out (HTTP 408).
python tools/push_in_batches.py [--paths docs/tiles] [--limit-mb 15] [--retries 3] [--remote origin] [--branch main]
Only untracked files under --paths are batched; each batch becomes one commit and one push. First it pushes any commits that are still waiting.
If a push keeps failing the script stops: run it again and it carries on where it left off."""
import argparse
import os
import subprocess
import sys
import time


def git(*args, stdin=None, check=True):
    r = subprocess.run(["git", *args], input=stdin, capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f"git {' '.join(args)} failed:\n{r.stderr or r.stdout}")
    return r


def push(remote, branch, retries):
    for attempt in range(1, retries + 1):
        r = git("push", remote, f"HEAD:{branch}", check=False)
        if r.returncode == 0:
            return
        print(f"  push failed (attempt {attempt}/{retries}): {(r.stderr or r.stdout).strip().splitlines()[-1]}")
        time.sleep(5 * attempt)
    raise SystemExit("Push still failing. Check your connection, then run this script again.")


def batches(files, limit):
    cur, size = [], 0
    for f in files:
        n = os.path.getsize(f)
        if cur and size + n > limit:
            yield cur
            cur, size = [], 0
        cur.append(f)
        size += n
    if cur:
        yield cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", default=["docs/tiles"])
    ap.add_argument("--limit-mb", type=float, default=15)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--branch", default="main")
    a = ap.parse_args()
    git("fetch", a.remote)
    ahead = int(git("rev-list", "--count", f"{a.remote}/{a.branch}..HEAD").stdout)
    if ahead:
        print(f"{ahead} commit(s) waiting to be pushed")
        push(a.remote, a.branch, a.retries)
    files = [f for f in git("ls-files", "-z", "--others", "--exclude-standard", "--", *a.paths).stdout.split("\0") if f]
    todo = list(batches(sorted(files), int(a.limit_mb * 1024 * 1024)))
    print(f"{len(files)} new files in {len(todo)} batch(es)")
    for i, batch in enumerate(todo, 1):
        git("add", "--pathspec-from-file=-", "--pathspec-file-nul", stdin="\0".join(batch))
        git("commit", "-q", "-m", f"add generated files, batch {i}/{len(todo)}")
        print(f"batch {i}/{len(todo)}: {len(batch)} files, {sum(os.path.getsize(f) for f in batch) / 1e6:.1f} MB")
        push(a.remote, a.branch, a.retries)
    print("done")


if __name__ == "__main__":
    main()
