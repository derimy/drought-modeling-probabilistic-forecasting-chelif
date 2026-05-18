"""
Author        : Imene DERRAR
Supervisor    : Abdelillah OTMANE CHERIF
Institutions  : NHSM — National Higher School of Mathematics
                IHFR — Institut Hydrométéorologique de Formation et de Recherche
Academic year : 2025–2026
Project       : Final-year engineering thesis in Applied Mathematics —
                Drought modelling and probabilistic forecasting for the Chélif basin

--------------------------------------------------------------------------------

E8_scpdsi_pipeline.py
================================================================================
scPDSI DATA ENGINEERING PIPELINE — Chélif Basin (Algeria)
--------------------------------------------------------------------------------
Thesis : Estimation and Prediction of Drought Indices using Statistical and
         Bayesian Time-Series Models.
Role   : Builds the *forcing* time-series (P, PET, T) consumed by the R
         scPDSI compute step (notebook E.7).

Outputs (under 02_processed/scpdsi_inputs/):
    terraclimate_basin_monthly.csv   date | ppt_mm | pet_mm | tmean_C | tmax_C | tmin_C | qc_flag
    era5land_basin_monthly.csv       date | ppt_mm | pet_mm | tmean_C | tmax_C | tmin_C | qc_flag

Design notes
------------
1. *Spatially aggregate first, then compute scPDSI.*  Computing scPDSI per
   grid cell and averaging the index produces an artificially smoothed series
   because the soil-bucket equations are non-linear.  Following van der Schrier
   et al. (2013), we average the meteorological forcings over the basin with
   cosine-latitude area weights, then run a single scPDSI on the basin series.

2. *Two physically independent products.*
   - TerraClimate (~4 km, climate-informed downscaling, native PET) — primary.
   - ERA5-Land   (~9 km, reanalysis, PET re-derived from temperature)  — validation.

3. *PET formulations differ by product, and this is intentional.*
   TerraClimate is supplied with its native modified Penman–Monteith PET
   and is the primary forcing branch. ERA5-Land monthly does not expose
   Tmax / Tmin, so its PET is derived from monthly Tmean through the
   latitude-corrected Thornthwaite (1948) formulation; the resulting
   summer-PET underestimation in semi-arid regimes is acknowledged and
   handled by treating ERA5-Land as the validation branch rather than the
   primary one.

4. *Alignment with the SPI master file.* The monthly DatetimeIndex of the
   SPI master CSV serves as the canonical reference. The pipeline reads
   it at startup, inherits the `date` column, and uses it as the target
   index for every scPDSI forcing. The start and end dates are inherited from the SPI reference index;
   alignment is verified before export through
       assert df["date"].equals(spi_index)
   and the pipeline stops if the check fails.
================================================================================
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr

# Optional dependencies that are imported lazily so the script can run on a
# machine that only has the cached NetCDF inputs available.
try:
    import rioxarray  # noqa: F401  (registers the .rio accessor on xarray)
    import geopandas as gpd
    from shapely.geometry import box
    HAS_GEOSTACK = True
except Exception:  # pragma: no cover
    HAS_GEOSTACK = False

# ------------------------------------------------------------------------------
# CONFIG  (edit here only)
# ------------------------------------------------------------------------------
# Project paths.
#   LEGACY_ROOT      holds raw NetCDFs, the basin shapefile, the SPI master,
#                    and the scPDSI outputs read/written by the R compute step.
#   STRUCTURED_ROOT  (under _structured_project/) holds the Python scripts
#                    and the post-pivot artefacts (spi_ready_dataset_main_*,
#                    basin_precip_master_*).
# Both are resolved from CHELIF_PROJECT_ROOT when set; edit the fallback
# below if running outside the standard layout.
ROOT            = Path(__file__).resolve().parent
LEGACY_ROOT     = Path(os.environ.get("CHELIF_PROJECT_ROOT", Path.cwd()))
STRUCTURED_ROOT = LEGACY_ROOT / "_structured_project"

@dataclass(frozen=True)
class Config:
    # ---------- canonical reference index (date axis inherited from SPI) ----------
    # The SPI master CSV defines the authoritative monthly index.  We read it
    # at startup and inherit its date column. scPDSI is computed on
    # exactly this set of months.
    spi_master_csv: Path = (
        STRUCTURED_ROOT / "04_outputs" / "02_spi_construction_and_validation" / "spi_regional_series.csv"
    )
    spi_date_col: str = "date"      # name of the date column inside spi_master_csv

    # ---------- basin precipitation master (SHARED with SPI) ----------
    # Station-derived basin-mean monthly precipitation. The same precipitation
    # observations that feed SPI must also feed scPDSI; only PET and AWC are
    # taken from the gridded products. Produced by build_basin_precip_master.py
    # and persisted under STRUCTURED_ROOT (alongside spi_ready_dataset_main_*).
    basin_precip_csv: Path = (
        STRUCTURED_ROOT / "01_data" / "external" / "basin_precip_master_1990_2015.csv"
    )
    basin_precip_col: str  = "precip_basin_mm"

    # ---------- basin geometry ----------
    basin_shp: Path  = LEGACY_ROOT / "01_data_raw" / "basin" / "chelif_basin.shp"
    bbox: Tuple[float, float, float, float] = (0.0, 34.0, 3.5, 36.5)  # lon_min, lat_min, lon_max, lat_max

    # ---------- inputs (NetCDF, legacy tree) ----------
    terraclimate_dir: Path = LEGACY_ROOT / "01_data_raw" / "terraclimate"
    era5_dir:         Path = LEGACY_ROOT / "01_data_raw" / "era5_land"

    # ---------- outputs (structured tree, single canonical location) ----------
    out_dir: Path     = STRUCTURED_ROOT / "02_processed" / "scpdsi_inputs"
    diag_dir: Path    = STRUCTURED_ROOT / "04_outputs"   / "scpdsi" / "diagnostics"

    # ---------- physics ----------
    lat_basin_deg: float = 35.2

    # ---------- QC ----------
    max_missing_frac: float = 0.05

CFG = Config()


def load_spi_index() -> pd.DatetimeIndex:
    """Read the SPI master CSV and return its monthly DatetimeIndex.

    This is the canonical reference for the time axis; all scPDSI
    forcings are aligned to it.  Tries the configured column name first;
    falls back to (year, month) reconstruction if a single 'date' column is
    not present.
    """
    if not CFG.spi_master_csv.exists():
        raise FileNotFoundError(
            f"SPI master CSV not found: {CFG.spi_master_csv}.  This file is "
            "the canonical reference index for scPDSI alignment, so it must be present."
        )
    df = pd.read_csv(CFG.spi_master_csv)
    if CFG.spi_date_col in df.columns:
        idx = pd.to_datetime(df[CFG.spi_date_col])
    elif {"year", "month"}.issubset(df.columns):
        idx = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    else:
        raise ValueError(
            f"SPI master {CFG.spi_master_csv} has neither a '{CFG.spi_date_col}' "
            f"column nor (year, month) columns.  Columns found: {list(df.columns)}"
        )
    idx = pd.DatetimeIndex(idx).to_period("M").to_timestamp(how="start")
    if idx.has_duplicates:
        raise ValueError("SPI master index has duplicate months")
    if not (idx == idx.sort_values()).all():
        raise ValueError("SPI master index is not sorted")
    expected = pd.date_range(idx.min(), idx.max(), freq="MS")
    if not idx.equals(expected):
        missing = expected.difference(idx)
        raise ValueError(
            f"SPI master index has gaps. Missing {len(missing)} month(s), "
            f"e.g. {list(missing[:5])}"
        )
    return idx


# Read the SPI index once and freeze it as a module-level constant.
SPI_INDEX = load_spi_index()
SPI_START = SPI_INDEX.min()
SPI_END   = SPI_INDEX.max()
log_setup_msg = (
    f"[align] SPI master defines the canonical index: "
    f"{SPI_START:%Y-%m-%d} -> {SPI_END:%Y-%m-%d} ({len(SPI_INDEX)} months)"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scpdsi")


def load_basin_precip() -> pd.Series:
    """Load the station-derived basin-mean monthly precipitation series.

    This is the canonical precipitation input for scPDSI, identical at the
    observation level to the precipitation that feeds SPI (May 2026
    methodological pivot: SPI and scPDSI must rest on the same observations
    so their comparison reflects only the structural difference between a
    statistical transform and a recursive water-balance state, not a
    data-source mismatch).

    The CSV is produced by build_basin_precip_master.py from the SPI station
    panel (precip_filled column) and must align byte-for-byte with SPI_INDEX.

    Returns:
        pd.Series of monthly precipitation totals (mm/month), name='ppt_mm',
        indexed by SPI_INDEX.
    """
    if not CFG.basin_precip_csv.exists():
        raise FileNotFoundError(
            f"Basin precip master not found: {CFG.basin_precip_csv}\n"
            f"  Run `python build_basin_precip_master.py` first to produce it."
        )
    df = pd.read_csv(CFG.basin_precip_csv)
    if CFG.spi_date_col not in df.columns:
        raise KeyError(
            f"{CFG.basin_precip_csv} has no '{CFG.spi_date_col}' column. "
            f"Cols: {list(df.columns)}"
        )
    if CFG.basin_precip_col not in df.columns:
        raise KeyError(
            f"{CFG.basin_precip_csv} missing precipitation column "
            f"'{CFG.basin_precip_col}'. Cols: {list(df.columns)}"
        )
    df[CFG.spi_date_col] = (
        pd.to_datetime(df[CFG.spi_date_col]).dt.to_period("M").dt.to_timestamp(how="start")
    )
    s = (
        df.set_index(CFG.spi_date_col)[CFG.basin_precip_col]
          .reindex(SPI_INDEX)
          .astype(float)
    )
    if s.isna().any():
        missing = s.index[s.isna()].tolist()[:5]
        raise ValueError(
            f"basin precip has {int(s.isna().sum())} NaN(s) after alignment to SPI_INDEX. "
            f"First missing: {missing}"
        )
    return s.rename("ppt_mm")


# Freeze the basin precipitation series at module load, same pattern as SPI_INDEX.
BASIN_PRECIP = load_basin_precip()
log.info(
    "[align] basin precip loaded: %d months  mean=%.1f mm  range=[%.1f, %.1f]",
    len(BASIN_PRECIP), float(BASIN_PRECIP.mean()),
    float(BASIN_PRECIP.min()), float(BASIN_PRECIP.max()),
)


# ==============================================================================
# 1. DOWNLOAD HOOKS
#    Download is intentionally factored out: networking environments differ.
#    Replace the bodies of these functions with your project-specific helpers.
# ==============================================================================
def ensure_terraclimate(years: range) -> None:
    """Ensure TerraClimate yearly NetCDFs for ppt, pet, tmax, tmin exist locally.

    Use the THREDDS endpoint:
        http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_terraclimate_<var>_1958_CurrentYear_GLOBE.nc
    Subset to the bounding box defined in CFG before saving locally.
    """
    CFG.terraclimate_dir.mkdir(parents=True, exist_ok=True)
    log.info("[TerraClimate] expecting yearly NetCDFs in %s", CFG.terraclimate_dir)


def ensure_era5land(years: range) -> None:
    """Ensure ERA5-Land monthly NetCDFs for tp, t2m, mx2t, mn2t (and optional
    ssrd, u10, v10, d2m for Penman) exist locally.  Uses the cdsapi client:

        c = cdsapi.Client()
        c.retrieve('reanalysis-era5-land-monthly-means', {...}, target=...)

    """
    CFG.era5_dir.mkdir(parents=True, exist_ok=True)
    log.info("[ERA5-Land]  expecting monthly NetCDFs in %s", CFG.era5_dir)


# ==============================================================================
# 2. BASIN MASK
# ==============================================================================
def load_basin_mask() -> "gpd.GeoDataFrame":
    if not HAS_GEOSTACK:
        raise RuntimeError("geopandas/rioxarray are required for basin clipping")
    if CFG.basin_shp.exists():
        gdf = gpd.read_file(CFG.basin_shp).to_crs("EPSG:4326")
        log.info("[basin] loaded polygon from %s", CFG.basin_shp.name)
        return gdf
    # Fallback: bounding-box polygon — clearly logged as a degraded mode.
    log.warning("[basin] shapefile not found, falling back to bounding box %s", CFG.bbox)
    geom = box(*CFG.bbox)
    return gpd.GeoDataFrame({"name": ["chelif_bbox"]}, geometry=[geom], crs="EPSG:4326")


def clip_to_basin(da: xr.DataArray, gdf) -> xr.DataArray:
    """Clip a (lat, lon, time) DataArray to the basin polygon."""
    da = da.rio.write_crs("EPSG:4326", inplace=False)
    return da.rio.clip(gdf.geometry.values, gdf.crs, drop=True, all_touched=True)


# ==============================================================================
# 3. SPATIAL AGGREGATION  (cosine-latitude weighted mean)
# ==============================================================================
def basin_mean(da: xr.DataArray) -> pd.Series:
    """Area-weighted mean across (lat, lon).  NaNs are ignored.  Returns a
    monthly pandas Series indexed by month-start timestamps."""
    if "lat" in da.dims:
        weights = np.cos(np.deg2rad(da["lat"]))
    elif "latitude" in da.dims:
        weights = np.cos(np.deg2rad(da["latitude"]))
    else:
        weights = xr.ones_like(da.isel(time=0))

    weighted = da.weighted(weights)
    space_dims = [d for d in da.dims if d != "time"]
    s = weighted.mean(dim=space_dims, skipna=True).to_pandas()

    # Force month-start timestamps and clip to the SPI-master window.
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp(how="start")
    s = s.loc[SPI_START: SPI_END]
    return s.astype(float)


def qc_flag(da: xr.DataArray) -> pd.Series:
    """Per-month fraction of missing pixels INSIDE the basin polygon.

    After rio.clip, the array still covers the rectangular bounding box of
    the polygon, so cells outside the polygon (or beyond the basin's
    irregular shape) are NaN forever.  Counting them as 'missing' would
    flag every month for small basins.  We therefore restrict the
    denominator to the set of cells that have *any* valid value across
    time -- i.e. cells that are part of the basin support."""
    space_dims = [d for d in da.dims if d != "time"]
    in_basin = da.notnull().any(dim="time").compute()  # bool mask over space
    n_in     = int(in_basin.sum().values)
    if n_in == 0:
        # Should not happen if the polygon overlaps the grid, but be safe.
        miss_frac = pd.Series(0.0, index=pd.to_datetime(da["time"].values))
    else:
        miss_count = da.where(in_basin).isnull().sum(dim=space_dims).to_pandas()
        miss_frac  = miss_count.astype(float) / float(n_in)
    miss_frac.index = pd.to_datetime(miss_frac.index).to_period("M").to_timestamp(how="start")
    miss_frac = miss_frac.loc[SPI_START: SPI_END]
    return (miss_frac > CFG.max_missing_frac).astype(int)


# ==============================================================================
# 4. TERRACLIMATE: NATIVE PET
# ==============================================================================
def build_terraclimate(gdf) -> pd.DataFrame:
    files = sorted(CFG.terraclimate_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No TerraClimate NetCDFs found in {CFG.terraclimate_dir}")
    ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 60})
    rename = {"ppt": "ppt", "pet": "pet", "tmax": "tmax", "tmin": "tmin"}
    ds = ds[list(rename.keys())]

    log.info("[TerraClimate] clipping to basin")
    ds = ds.rio.write_crs("EPSG:4326", inplace=False)
    ds = ds.rio.clip(gdf.geometry.values, gdf.crs, drop=True, all_touched=True)

    # ---- Precipitation: station-derived basin mean (NOT TerraClimate gridded) ----
    # Same observations as SPI -- see load_basin_precip() and the May 2026
    # methodological pivot. TerraClimate `ppt` is intentionally NOT consumed
    # here; only PET, Tmax, Tmin feed the scPDSI recursion.
    ppt   = BASIN_PRECIP              # mm / month, indexed by SPI_INDEX
    pet   = basin_mean(ds["pet"])     # mm / month  (native modified Penman-Monteith)
    tmax  = basin_mean(ds["tmax"])    # °C
    tmin  = basin_mean(ds["tmin"])    # °C
    tmean = (tmax + tmin) / 2.0
    qc    = qc_flag(ds["pet"])        # QC tracks PET (gridded) coverage; P is station-based

    df = pd.DataFrame({
        "date":     ppt.index,
        "ppt_mm":   ppt.values,
        "pet_mm":   pet.values,
        "tmean_C":  tmean.values,
        "tmax_C":   tmax.values,
        "tmin_C":   tmin.values,
        "qc_flag":  qc.values,
    })
    return _post_qc(df, "TerraClimate")


# ==============================================================================
# 5. ERA5-LAND: PET VIA THORNTHWAITE  (Tmean only — see note below)
# ------------------------------------------------------------------------------
# The CDS `reanalysis-era5-land-monthly-means` product does NOT expose monthly
# Tmax / Tmin as variables (only the hourly product does).  Hargreaves-Samani
# requires the temperature range, so we fall back to Thornthwaite (1948) which
# uses only Tmean.  Thornthwaite is documented in the methodology chapter as
# the acceptable alternative formulation; its known underestimation of summer
# PET in semi-arid regimes is bounded by treating ERA5-Land as the *validation*
# series (TerraClimate with native Penman-Monteith remains primary).
# ==============================================================================
def _daylight_hours(lat_deg: float, doy: np.ndarray) -> np.ndarray:
    """Mean daylight hours per day (FAO-56 Eq. 34).  Returns h."""
    phi = math.radians(lat_deg)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0))
    return (24.0 / np.pi) * ws


def thornthwaite(tmean_C: pd.Series, lat_deg: float) -> pd.Series:
    """Monthly PET (mm/month) via Thornthwaite (1948), latitude-corrected.

    Steps:
      1. Annual heat index I = sum_m (Tm/5)**1.514  (only positive Tm)
      2. Empirical exponent a = 6.75e-7 I^3 - 7.71e-5 I^2 + 1.792e-2 I + 0.49239
      3. Unadjusted PET = 16 * (10*Tm/I)^a   [mm / 30-day, 12-hour reference]
      4. Adjust by daylight hours and days-in-month
    """
    idx = tmean_C.index
    t = tmean_C.values.astype(float).copy()
    t_pos = np.clip(t, 0.0, None)

    # Annual heat index per calendar year, then broadcast back to months.
    yr = idx.year
    df_local = pd.DataFrame({"yr": yr, "t_pos": t_pos})
    monthly_idx = (df_local["t_pos"] / 5.0) ** 1.514
    I_per_year = df_local.assign(idx_term=monthly_idx).groupby("yr")["idx_term"].transform("sum").values

    a = (6.75e-7 * I_per_year**3
         - 7.71e-5 * I_per_year**2
         + 1.792e-2 * I_per_year
         + 0.49239)

    # Unadjusted Thornthwaite: 16 * (10 T / I)^a, mm per "standard" month.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(I_per_year > 0, 10.0 * t_pos / I_per_year, 0.0)
        pet_unadj = 16.0 * np.power(ratio, a)
    pet_unadj = np.where(t_pos <= 0, 0.0, pet_unadj)

    # Daylight + days-in-month correction.
    midmonth = pd.DatetimeIndex(idx) + pd.Timedelta(days=14)
    doy = midmonth.dayofyear.to_numpy()
    N = _daylight_hours(lat_deg, doy)                       # h/day
    days = pd.DatetimeIndex(idx).days_in_month.to_numpy()   # days/month
    pet = pet_unadj * (N / 12.0) * (days / 30.0)
    return pd.Series(pet, index=idx, name="pet_mm")


def _harmonize_era5_dims(ds: xr.Dataset) -> xr.Dataset:
    """CDS Beta delivers ERA5-Land NetCDFs with `valid_time` instead of `time`
    and may add scalar `expver` / `number` coords.  Normalize so the rest of
    the pipeline can speak its usual `time` dialect."""
    if "valid_time" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    elif "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    for stray in ("expver", "number"):
        if stray in ds.coords and stray not in ds.dims:
            ds = ds.drop_vars(stray, errors="ignore")
    return ds


def build_era5land(gdf) -> pd.DataFrame:
    files = sorted(CFG.era5_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No ERA5-Land NetCDFs found in {CFG.era5_dir}")
    ds = xr.open_mfdataset(files, combine="by_coords", chunks={"valid_time": 60})
    ds = _harmonize_era5_dims(ds)

    # CDS Beta returns only tp + t2m for the monthly product (no mx2t/mn2t).
    keep = [v for v in ("tp", "t2m") if v in ds.data_vars]
    if set(keep) != {"tp", "t2m"}:
        raise KeyError(
            f"ERA5-Land file is missing tp/t2m. Found vars: {list(ds.data_vars)}"
        )
    ds = ds[keep]

    log.info("[ERA5-Land] clipping to basin")
    ds = ds.rio.write_crs("EPSG:4326", inplace=False)
    ds = ds.rio.clip(gdf.geometry.values, gdf.crs, drop=True, all_touched=True)

    # ---- Precipitation: station-derived basin mean (NOT ERA5 gridded) ----
    # Same series as TerraClimate variant (and as SPI). ERA5 `tp` is no longer
    # consumed by scPDSI; only `t2m` feeds the Thornthwaite PET.
    tp  = BASIN_PRECIP                                           # mm / month, indexed by SPI_INDEX
    t2m = basin_mean(ds["t2m"]) - 273.15                         # K -> °C

    pet  = thornthwaite(t2m, CFG.lat_basin_deg)                  # mm / month

    qc   = qc_flag(ds["t2m"])                                    # QC tracks T2M (gridded) coverage; P is station-based

    df = pd.DataFrame({
        "date":    tp.index,
        "ppt_mm":  tp.values,
        "pet_mm":  pet.values,
        "tmean_C": t2m.values,
        "tmax_C":  np.nan,             # not provided by ERA5-Land monthly product
        "tmin_C":  np.nan,             # not provided by ERA5-Land monthly product
        "qc_flag": qc.values,
    })
    return _post_qc(df, "ERA5-Land")


# ==============================================================================
# 6. POST-QC AND CLIMATOLOGICAL GAP-FILLING
# ==============================================================================
def _post_qc(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Apply the QC convention from the methodology chapter:
       - Flag any month with >5% missing pixels (already in qc_flag).
       - If <=2 consecutive bad months: fill with calendar-month climatology.
       - If >2 consecutive bad months: leave NaN and let R abort the run.
    """
    df = df.set_index("date").sort_index()
    bad = df["qc_flag"].astype(bool)

    # Identify bad-month runs
    run_id = (bad != bad.shift()).cumsum()
    run_len = bad.groupby(run_id).transform("sum")
    short_bad = bad & (run_len <= 2)
    long_bad  = bad & (run_len >  2)

    # Climatological monthly mean using only clean months.
    # NOTE: `ppt_mm` is now station-derived (see load_basin_precip) and is
    # complete by construction; it is NOT subject to gridded-coverage QC.
    # Only PET / T columns are gap-filled when qc_flag flags a short bad run.
    # Skip columns that are entirely NaN (e.g. ERA5 has no tmax/tmin).
    fill_cols = [c for c in ("pet_mm", "tmean_C", "tmax_C", "tmin_C")
                 if df[c].notna().any()]
    clean = df.loc[~bad, fill_cols]
    climo = clean.groupby(clean.index.month).mean()

    for col in fill_cols:
        fill_vals = df.index[short_bad].month.map(climo[col])
        df.loc[short_bad, col] = fill_vals.values

    n_long = int(long_bad.sum())
    n_short = int(short_bad.sum())
    log.info("[%s QC] short-gap filled: %d   long-gap retained NaN: %d", label, n_short, n_long)

    return df.reset_index()


# ==============================================================================
# 7. WRITE OUTPUTS  (+ alignment check)
# ==============================================================================
def _write(df: pd.DataFrame, name: str) -> Path:
    CFG.out_dir.mkdir(parents=True, exist_ok=True)
    path = CFG.out_dir / name
    df.to_csv(path, index=False, date_format="%Y-%m-%d")
    log.info("[write] %s  (rows=%d)", path, len(df))
    return path


def assert_alignment(df: pd.DataFrame, label: str) -> None:
    """Alignment check: scPDSI forcing index must match the SPI index.

    No tolerance, no fuzzy matching - this is the alignment guarantee that
    the modeling chapters depend on.
    """
    got = pd.DatetimeIndex(pd.to_datetime(df["date"]))
    if not got.equals(SPI_INDEX):
        diff_n = len(got.symmetric_difference(SPI_INDEX))
        raise AssertionError(
            f"[{label}] monthly index does NOT match the SPI master.\n"
            f"  SPI index : {SPI_START:%Y-%m} -> {SPI_END:%Y-%m}  ({len(SPI_INDEX)} months)\n"
            f"  Got       : {got.min():%Y-%m} -> {got.max():%Y-%m}  ({len(got)} months)\n"
            f"  symmetric difference: {diff_n} month(s)"
        )
    log.info("[%s] monthly index strictly identical to SPI master (%d months)",
             label, len(SPI_INDEX))


# ==============================================================================
# 8. MAIN
# ==============================================================================
def main() -> None:
    log.info("=" * 78)
    log.info("scPDSI forcing pipeline - Chelif basin")
    log.info(log_setup_msg)
    log.info("=" * 78)

    years = range(SPI_START.year, SPI_END.year + 1)
    ensure_terraclimate(years)
    ensure_era5land(years)

    gdf = load_basin_mask()

    df_tc = build_terraclimate(gdf)
    assert_alignment(df_tc, "TerraClimate")
    _write(df_tc, "terraclimate_basin_monthly.csv")

    df_e5 = build_era5land(gdf)
    assert_alignment(df_e5, "ERA5-Land")
    _write(df_e5, "era5land_basin_monthly.csv")

    CFG.diag_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.concat({
        "TerraClimate": df_tc.set_index("date").describe().T,
        "ERA5-Land":    df_e5.set_index("date").describe().T,
    })
    summary.to_csv(CFG.diag_dir / "forcing_summary.csv")
    log.info("[diag] summary written to %s", CFG.diag_dir / "forcing_summary.csv")
    log.info("done.")


if __name__ == "__main__":
    main()
