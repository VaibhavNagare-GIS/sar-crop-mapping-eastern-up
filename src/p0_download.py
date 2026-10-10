"""Phase 0a: download AgriFieldNet train label rasters and any source .json (dates).

Usage:  python p0_download.py [out_dir]        (default out_dir = data)
Safe to re-run: finished files are skipped, partial files are never kept.
Exits with code 1 if any download failed, so a bad download cannot pass silently.
Layout seen on the real server: train_labels/<prefix>_<id>.tif (crop labels)
                                train_labels/<prefix>_<id>_field_ids.tif (field ids)
"""
import collections
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) agri-sar-project/1.0"}
MAX_JSON = 3000  # refuse to bulk-download more json than this without asking
# Tried in order. 'strip' = key prefixes the server adds to listed keys (longest first).
CANDIDATES = [
    dict(name="source.coop proxy", list="https://data.source.coop/radiantearth/agrifieldnet-competition",
         qp="", files="https://data.source.coop/radiantearth/agrifieldnet-competition",
         strip=("radiantearth/agrifieldnet-competition/", "agrifieldnet-competition/")),
    dict(name="AWS us-west-2 bucket", list="https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop",
         qp="radiantearth/agrifieldnet-competition/",
         files="https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/radiantearth/agrifieldnet-competition",
         strip=("radiantearth/agrifieldnet-competition/",)),
]
BASE = None  # file base of the candidate that worked


def get(url, tries=4):
    err = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                body = r.read()
                want = r.headers.get("Content-Length")
                if want is not None and len(body) != int(want):
                    raise IOError(f"short read {len(body)} of {want}")
                return body
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 403, 404):  # retrying will not help
                raise RuntimeError(f"HTTP {e.code} for {url}: {e.read()[:150]!r}")
            err = e
            time.sleep(2 ** i)
        except Exception as e:  # network errors are retried, then raised
            err = e
            time.sleep(2 ** i)
    raise RuntimeError(f"GET failed after {tries} tries: {url} ({err})")


def list_keys(prefix, cand):
    """S3-style listing with pagination. Returns keys relative to the product root."""
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": cand["qp"] + prefix}
        if token:
            q["continuation-token"] = token
        raw = get(cand["list"] + "?" + urllib.parse.urlencode(q))
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            raise SystemExit(f"Listing is not XML. First 300 bytes:\n{raw[:300]!r}")
        token, truncated = None, False
        for e in root.iter():
            t = e.tag.split("}")[-1]
            if t == "Key":
                keys.append(e.text)
            elif t == "NextContinuationToken":
                token = e.text
            elif t == "IsTruncated":
                truncated = e.text == "true"
        if not truncated:
            break
        if not token:
            raise SystemExit("Truncated listing without a continuation token")
    out = []
    for k in keys:
        for s in sorted(cand["strip"], key=len, reverse=True):
            if k.startswith(s):
                k = k[len(s):]
                break
        out.append(k)
    bad = [k for k in out if not k.startswith(prefix)]
    if bad:
        raise SystemExit(f"Listing returned keys outside '{prefix}', e.g. {bad[:3]}")
    return out


def fetch(key):
    dest = OUT / key
    if dest.exists() and dest.stat().st_size > 0:
        return key, "skipped", 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        body = get(BASE + "/" + urllib.parse.quote(key))
        if not body:
            raise IOError("empty file")
        part = dest.with_name(dest.name + ".part")
        part.write_bytes(body)
        os.replace(part, dest)
        return key, "ok", len(body)
    except Exception as e:
        return key, f"FAILED: {e}", 0


def pick_candidate():
    global BASE
    log = []
    for c in CANDIDATES:
        try:
            if list_keys("train_labels/", c):
                BASE = c["files"]
                print("using:", c["name"])
                return c
            log.append(f"{c['name']}: listing worked but returned no files")
        except (RuntimeError, SystemExit) as e:
            log.append(f"{c['name']}: {e}")
    raise SystemExit("All access methods failed:\n" + "\n".join(log) + "")


def select(label_keys, source_keys):
    """Pure function: decide what to download and validate pairing."""
    tifs = [k for k in label_keys if k.endswith(".tif")]
    fids = [k for k in tifs if k.endswith("_field_ids.tif")]
    labs = [k for k in tifs if not k.endswith("_field_ids.tif")]
    if not tifs:
        raise SystemExit("No .tif files under train_labels/")
    if len(fids) != len(labs) or not labs:
        raise SystemExit(f"Label files are not paired: {len(labs)} label tifs vs {len(fids)} field_ids tifs")
    want = {k[:-len("_field_ids.tif")] for k in fids}
    have = {k[:-len(".tif")] for k in labs}
    if want != have:
        raise SystemExit(f"Unpaired label/field_ids files, e.g. {sorted(want ^ have)[:3]}")
    jsons = [k for k in source_keys if k.endswith(".json")]
    return tifs, jsons


def summarize_source(source_keys):
    ext = collections.Counter(os.path.splitext(k)[1] or "(none)" for k in source_keys)
    print(f"source/: {len(source_keys)} files, by extension: {dict(ext)}")
    for k in source_keys[:8]:
        print("   sample:", k)


def main():
    cand = pick_candidate()
    label_keys = list_keys("train_labels/", cand)
    source_keys = list_keys("source/", cand)
    tifs, jsons = select(label_keys, source_keys)
    print(f"label tifs: {len(tifs)} ({len(tifs) // 2} chips)")
    summarize_source(source_keys)
    if not jsons:
        print("WARNING: no .json files under source/. Sentinel-2 dates are NOT available yet. See the source/ lines above.")
    if len(jsons) > MAX_JSON:
        print(f"WARNING: {len(jsons)} json files is more than {MAX_JSON}; downloading none. See the source/ lines above.")
        jsons = []
    todo = tifs + jsons
    done = fail = total = 0
    with ThreadPoolExecutor(8) as ex:
        for key, status, n in ex.map(fetch, todo):
            done += 1
            total += n
            if status.startswith("FAILED"):
                fail += 1
                print(key, status)
            if done % 200 == 0:
                print(f"{done}/{len(todo)} done")
    print(f"finished: {done - fail} ok or skipped, {fail} failed, {total / 1e6:.1f} MB newly downloaded")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
