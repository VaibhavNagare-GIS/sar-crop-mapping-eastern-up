"""Shared study settings. Change values HERE only; every phase-1+ script reads them."""
import datetime as dt

BLOCK_CELLS = ["162_54", "162_55", "163_54", "163_55", "164_54", "164_55"]  # 0.5 deg cells, eastern UP
# Extra cell sets, selected with --set. They are never used for training.
SETS = {"": BLOCK_CELLS,
        "_near": ["165_54", "165_55", "162_56", "163_56"],                  # eastern UP cells next to the study block
        "_far": ["152_49", "152_50", "153_48", "153_49", "153_50", "154_48", "154_49", "154_50"]}   # cells around 76.2 to 77.3 E, 24.4 to 25.4 N
CLASSES = {1: "Wheat", 2: "Mustard", 3: "Lentil", 4: "No crop/Fallow", 6: "Sugarcane"}  # label code -> name
SEASON_START = "2021-10-01"   # PROVISIONAL (Rabi 2021-22). Not confirmed until the season test.
BIN_DAYS = 12                 # Sentinel-1A repeat over this area (verified: gaps of 12 days)
N_BINS = 20


def bins(start=None):
    """[(start, end_exclusive), ...] ISO dates of the N_BINS consecutive composite windows."""
    s = dt.date.fromisoformat(start or SEASON_START)
    d = lambda i: (s + dt.timedelta(days=BIN_DAYS * i)).isoformat()
    return [(d(i), d(i + 1)) for i in range(N_BINS)]


def season_end(start=None):
    """Exclusive end date of the last bin."""
    return bins(start)[-1][1]


def gee_init(project):
    import ee
    try:
        ee.Initialize(project=project)
    except Exception as e:
        hint = ("\nMost likely the saved login is a DIFFERENT Google account from the project owner.\n"
                "Fix: earthengine authenticate --force") if "permission" in str(e).lower() else ""
        raise SystemExit(f"Earth Engine could not start for project '{project}': {str(e)[:300]}{hint}")
    return ee
