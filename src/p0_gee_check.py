"""Phase 0c: how many Sentinel-1 IW VV+VH scenes really cover the area, per orbit?

One-time login:   earthengine authenticate
Run:  python p0_gee_check.py --project YOUR_CLOUD_PROJECT --bbox W S E N --start 2021-10-01 --end 2022-05-01
Selftest (no login needed):  python p0_gee_check.py --selftest
"""
import argparse
import collections
import datetime as dt
import statistics


def summarize(rows):
    """rows: [[time_ms, platform, pass, relative_orbit], ...] -> text report. Pure function."""
    if not rows:
        return "NO SCENES FOUND. Check the bbox order (W S E N), the dates and the GEE project."
    by = collections.defaultdict(list)
    for t, plat, ps, ro in rows:
        by[(str(ps), str(ro))].append((t, str(plat)))
    out = [f"total scenes: {len(rows)}", "pass, rel.orbit -> n scenes, platforms, day gaps (min/median/max)"]
    for key, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        days = sorted({dt.datetime.fromtimestamp(t / 1000, dt.timezone.utc).date() for t, _ in v})
        gaps = [(b - a).days for a, b in zip(days, days[1:])] or [0]
        plats = ",".join(sorted({p for _, p in v}))
        out.append(f"{key[0]:11s} {key[1]:>4s} -> {len(v):3d} scenes, {plats}, gaps {min(gaps)}/{statistics.median(gaps)}/{max(gaps)}")
    top = max(by.items(), key=lambda kv: len(kv[1]))
    days = sorted({dt.datetime.fromtimestamp(t / 1000, dt.timezone.utc).date().isoformat() for t, _ in top[1]})
    out.append(f"dates of the busiest orbit {top[0]}: {', '.join(days)}")
    return "\n".join(out)


def selftest():
    ms = lambda d: int(dt.datetime.fromisoformat(d).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    rows = [[ms("2022-01-01"), "A", "ASCENDING", 100], [ms("2022-01-13"), "A", "ASCENDING", 100],
            [ms("2022-01-31"), "A", "ASCENDING", 100], [ms("2022-01-05"), "A", "DESCENDING", 20]]
    rep = summarize(rows)
    assert "total scenes: 4" in rep and "3 scenes" in rep and "gaps 12/15.0/18" in rep, rep
    assert summarize([]).startswith("NO SCENES")
    print(rep)
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.project and a.bbox and a.start and a.end):
        raise SystemExit("need --project, --bbox, --start and --end")
    w, s, e, n = a.bbox
    if not (w < e and s < n):
        raise SystemExit("bbox must be W S E N with W<E and S<N")
    import ee  # imported late so --selftest works without it
    try:
        ee.Initialize(project=a.project)
    except Exception as e:
        hint = ("\nMost likely the saved login is a DIFFERENT Google account from the one that owns the project.\n"
                "Fix: earthengine authenticate --force   (sign in with the same account shown in code.earthengine.google.com)\n"
                "If it still fails: give that account the 'Service Usage Consumer' role in Cloud Console > IAM, "
                "and check the Earth Engine API is enabled.") if "permission" in str(e).lower() else ""
        raise SystemExit(f"Earth Engine could not start for project '{a.project}': {str(e)[:300]}{hint}")
    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(ee.Geometry.Rectangle([w, s, e, n]))
           .filterDate(a.start, a.end)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    rows = col.reduceColumns(ee.Reducer.toList(4), ["system:time_start", "platform_number",
                                                    "orbitProperties_pass", "relativeOrbitNumber_start"]).get("list").getInfo()
    print(summarize(rows))


if __name__ == "__main__":
    main()
