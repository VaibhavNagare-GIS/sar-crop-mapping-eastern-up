"""Import this FIRST in any script that uses rasterio.

Fixes: CRSError ... proj.db ... "It comes from another PROJ installation"
(PostgreSQL/PostGIS and other GIS installs set PROJ_LIB or PROJ_DATA system-wide).
We point rasterio at the PROJ and GDAL data folders that ship inside its own wheel.
We deliberately avoid pyproj/geopandas in this project, they would need their own fix.
"""
import importlib.util
import os
from pathlib import Path


def fix():
    spec = importlib.util.find_spec("rasterio")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("rasterio is not installed in this Python. Run: pip install rasterio")
    root = Path(list(spec.submodule_search_locations)[0])
    proj, gdal = root / "proj_data", root / "gdal_data"
    if (proj / "proj.db").exists():
        os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"] = str(proj)
    if gdal.exists():
        os.environ["GDAL_DATA"] = str(gdal)


def check():
    """Fail loudly now, not halfway through a run."""
    from rasterio.crs import CRS
    try:
        CRS.from_epsg(4326)
        CRS.from_epsg(32644)
    except Exception as e:
        raise SystemExit(f"PROJ still broken: {e}\nPROJ_LIB={os.environ.get('PROJ_LIB')} "
                         f"PROJ_DATA={os.environ.get('PROJ_DATA')}")


fix()
