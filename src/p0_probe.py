"""Phase 0d: find where the Sentinel-2 acquisition date lives. Downloads ONE small file.

Usage:  python p0_probe.py [data_dir]      (default data)
Prints: top-level folders on the server, and every tag / description inside one source tif,
with any date-like value flagged.
"""
import collections
import re
import sys

import env_fix  # must come before rasterio
import rasterio

import p0_download as dl  # reuses listing + download (its OUT folder = argv[1])

DATE = re.compile(r"(19|20)\d{2}[-/_.:]?(0[1-9]|1[0-2])[-/_.:]?(0[1-9]|[12]\d|3[01])")


def scan(profile_items):
    """Pure function: list (where, value) pairs that look like dates."""
    return [(k, v) for k, v in profile_items if v is not None and DATE.search(str(v))]


def inspect(path):
    items = []
    with rasterio.open(path) as src:
        print(f"\n{path.name}: {src.count} band(s), {src.dtypes}, {src.width}x{src.height}, crs={src.crs}, nodata={src.nodata}")
        for k, v in src.tags().items():
            items.append((f"dataset tag {k}", v))
        for i in range(1, src.count + 1):
            items.append((f"band {i} description", src.descriptions[i - 1]))
            for k, v in src.tags(i).items():
                items.append((f"band {i} tag {k}", v))
        for ns in src.tag_namespaces():
            if ns == "DERIVED_SUBDATASETS":  # GDAL echoes the file path here, not metadata
                continue
            for k, v in src.tags(ns=ns).items():
                items.append((f"namespace {ns} tag {k}", v))
    for k, v in items:
        print(f"   {k}: {v}")
    return items


def main():
    cand = dl.pick_candidate()
    allk = dl.list_keys("", cand)
    top = collections.Counter(k.split("/", 1)[0] if "/" in k else "(root file)" for k in allk)
    print("server top level:", dict(top))
    print("root-level files:", [k for k in allk if "/" not in k][:20])
    others = [k for k in allk if not k.startswith(("source/", "train_labels/", "test_labels/"))]
    print("other keys (first 30):", others[:30])
    key = next(k for k in allk if k.startswith("source/") and k.endswith("_B02_10m.tif"))
    k, status, _ = dl.fetch(key)
    if status.startswith("FAILED"):
        raise SystemExit(status)
    hits = scan(inspect(dl.OUT / key))
    print("\nDATE CANDIDATES:", hits if hits else "none found inside the tif")


def selftest():
    assert scan([("a", "2022-01-12T05:00"), ("b", "hello"), ("c", None), ("d", "20220112")]) == \
        [("a", "2022-01-12T05:00"), ("d", "20220112")]
    assert scan([("x", "12345678")]) == []   # 1234... is not a plausible date
    print("SELFTEST OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
