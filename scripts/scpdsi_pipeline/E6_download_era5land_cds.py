"""
Author        : Imene DERRAR
Supervisor    : Abdelillah OTMANE CHERIF
Institutions  : NHSM — National Higher School of Mathematics
                IHFR — Institut Hydrométéorologique de Formation et de Recherche
Academic year : 2025–2026
Project       : Final-year engineering thesis in Applied Mathematics —
                Drought modelling and probabilistic forecasting for the Chélif basin

--------------------------------------------------------------------------------

E6_download_era5land_cds.py
================================================================================
ERA5-Land monthly download helper for the scPDSI pipeline (Chélif basin).

Retrieves the two ERA5-Land monthly variables required by the scPDSI
pipeline (notebook E.8):
    - total_precipitation                                        -> tp
    - 2m_temperature                                             -> t2m

PET is later derived from monthly Tmean (t2m) through the
latitude-corrected Thornthwaite formulation, because the CDS
`reanalysis-era5-land-monthly-means` product does not expose monthly
Tmax / Tmin variables. Tmax and Tmin are therefore not retrieved by
the final pipeline.

Properties:
    - One NetCDF per year (CDS rejects very large single-shot requests).
    - Restartable: years already downloaded are skipped.
    - Bounding box buffers the basin polygon by ~0.05 deg.

Prerequisites:
    1. Free CDS account at https://cds.climate.copernicus.eu/
    2. Accept ERA5-Land terms once on the dataset page.
    3. ~/.cdsapirc on Linux  /  %USERPROFILE%\\.cdsapirc on Windows:

           url: https://cds.climate.copernicus.eu/api
           key: <UID>:<API-KEY>

    4. pip install cdsapi netCDF4

Usage:
    python E6_download_era5land_cds.py
================================================================================
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

try:
    import cdsapi
except ImportError:
    print("Missing dependency. Install with:  pip install cdsapi netCDF4", file=sys.stderr)
    raise

# --------------------------------------------------------------------------- #
# CONFIG (edit if your basin or window changes)                                #
# --------------------------------------------------------------------------- #
ROOT       = Path(__file__).resolve().parent
OUT_DIR    = ROOT / "01_data_raw" / "era5_land"
START_YEAR = 1989
END_YEAR   = 2019

# CDS expects [North, West, South, East]
AREA = [36.5, 0.0, 34.0, 3.5]   # tightened Chélif bounding box (N, W, S, E)

# NOTE: the CDS `reanalysis-era5-land-monthly-means` product does NOT expose
# monthly Tmax/Tmin variables, so we only request precipitation and 2-m
# temperature.  PET for ERA5-Land is computed via Thornthwaite (Tmean only).
VARIABLES = [
    "total_precipitation",
    "2m_temperature",
]

DATASET = "reanalysis-era5-land-monthly-means"
PRODUCT = "monthly_averaged_reanalysis"
MONTHS  = [f"{m:02d}" for m in range(1, 13)]

# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("era5land")


def request_year(client: "cdsapi.Client", year: int, target: Path) -> None:
    request = {
        "product_type": PRODUCT,
        "variable":     VARIABLES,
        "year":         str(year),
        "month":        MONTHS,
        "time":         "00:00",
        "format":       "netcdf",
        "area":         AREA,
    }
    log.info("[%d] submitting CDS request -> %s", year, target.name)
    client.retrieve(DATASET, request, str(target))
    log.info("[%d] saved (%.1f MB)", year, target.stat().st_size / 1e6)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("ERA5-Land download helper")
    log.info("  output dir : %s", OUT_DIR)
    log.info("  years      : %d -> %d", START_YEAR, END_YEAR)
    log.info("  bbox (NWSE): %s", AREA)

    client = cdsapi.Client()  # reads ~/.cdsapirc automatically

    todo = []
    for year in range(START_YEAR, END_YEAR + 1):
        target = OUT_DIR / f"era5land_chelif_{year}.nc"
        if target.exists() and target.stat().st_size > 1_000_000:
            log.info("[%d] already present (%.1f MB) - skipping",
                     year, target.stat().st_size / 1e6)
            continue
        todo.append((year, target))

    if not todo:
        log.info("Nothing to do. All years are present.")
        return

    log.info("Years to fetch: %s", ", ".join(str(y) for y, _ in todo))
    for year, target in todo:
        for attempt in range(1, 4):
            try:
                request_year(client, year, target)
                break
            except Exception as exc:                              # noqa: BLE001
                wait = 30 * attempt
                log.warning("[%d] attempt %d failed: %s — retrying in %ds",
                            year, attempt, exc, wait)
                time.sleep(wait)
        else:
            log.error("[%d] all attempts failed — re-run later", year)

    log.info("Done.")


if __name__ == "__main__":
    main()
