"""
Author        : Imene DERRAR
Supervisor    : Dr. Abdelillah OTMANE CHERIF
Institutions  : NHSM — National Higher School of Mathematics
                IHFR — Institut Hydrométéorologique de Formation et de Recherche
Academic year : 2025–2026
Project       : Final-year engineering thesis in Applied Mathematics
                Specialization: Statistics, Econometrics and Actuarial Science
                Drought modelling and probabilistic forecasting for the Chélif basin

--------------------------------------------------------------------------------

R2 — CMIP6 model evaluation against ONM in-situ precipitation (per station)
=====================================================================================
Chéliff basin, evaluation period 1990-01 .. 2014-12 (CMIP6 historical end).

Reference dataset: ONM rain-gauge network — the same data used to build the
observed SPI in the rest of the thesis.  Each of the 9 stations is evaluated
against each of the 14 CMIP6 models *independently*: the model series at the
station is the time series of the nearest CMIP6 grid cell to the station
coordinates (no spatial aggregation).  This preserves the pipeline consistency
("the model is benchmarked against the same observation that defines our SPI")
and respects the supervisor's requirement to treat each station alone.

Pipeline
--------
1.  Load ONM monthly precipitation in long format (station, date, precip_mm)
    and pivot to wide (date x station).  Use all 9 stations present in the file.
2.  Load station metadata to obtain (latitude, longitude) for every station.
3.  For each CMIP6 model:
      a) open `pr` historical, harmonise dim names, longitude wrap and
         calendar, convert kg/m^2/s -> mm/month, regrid to a common 0.25 deg
         lattice over the basin box;
      b) for each station, sample the nearest grid cell -> per-station model series;
      c) compute the six skill metrics vs the corresponding observation series.
4.  Write a long-format `skill_metrics.csv` with one row per (station, model) pair.
5.  Plot:
      - a 3 x 3 panel of Taylor diagrams (one per station),
      - a station x model heatmap of the composite skill score,
      - a per-station annual-cycle figure,
      - a per-station bias and dry-month frequency figure.
6.  Selection:
      - per-station best 3 (one tuple of three models for each station),
      - global best 3 (lowest mean rank of composite_S across stations),
        with bias / cycle / dry-frequency guards applied.

Inputs (paths relative to project root):
    01_data/external/spi_ready_dataset_main_1990_2015.csv          (precipitation, long)
    01_data/external/master_station_metadata_final_plus_pluvio.csv (station coords)
        ... or 03_outputs/arcgis/spi6_with_coordinates.csv as fallback.
    01_data/cmip6/historical/<model_folder>/pr_*.nc                (CMIP6 historical)

Outputs (under 04_outputs/r2_taylor_cmip6_evaluation/):
    tables/skill_metrics.csv          long format, 1 row / (station, model)
    tables/best_three_per_station.csv 9 rows, top 3 models for each station
    tables/best_three_global.csv      3 rows, models passing the guards by mean rank
    figures/taylor_per_station.png    3 x 3 small Taylor diagrams
    figures/skill_heatmap.png         station x model composite-S heatmap
    figures/annual_cycles.png         9 small panels of station + 12-model cycles
    figures/bias_dryfreq.png          two stacked station x model heatmaps
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist import floating_axes, grid_finder

# --------------------------------------------------------------------------- #
# 0. CONFIGURATION
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    # Spatial domain (CDS order: N, W, S, E) — same as the CMIP6 download
    bbox_nwse: Tuple[float, float, float, float] = (36.5, 0.0, 34.0, 3.5)

    # Evaluation window: CMIP6 `historical` ends 2014-12
    eval_start: str = "1990-01"
    eval_end:   str = "2014-12"

    # Common regridding lattice (deg) for the basin domain
    common_grid_deg: float = 0.25

    # Composite-skill weights (must sum to 1)
    w_taylor:   float = 0.50
    w_bias:     float = 0.20
    w_cycle:    float = 0.20
    w_dry_freq: float = 0.10

    # Guards for the global "best 3"
    guard_abs_bias_max:    float = 0.50    # |bias| < 50 %  averaged across stations
    guard_cycle_corr_min:  float = 0.70    # mean cycle_corr > 0.70
    guard_dryfreq_err_max: float = 0.20    # mean dry-freq error <= 0.20

    # Dry-month threshold (mm)
    dry_month_threshold_mm: float = 30.0

    # Final 14-model ensemble (display name -> CDS folder name).
    # Some candidate models were excluded because complete SSP availability
    # or variable consistency could not be ensured across the full workflow.
    models: Dict[str, str] = field(default_factory=lambda: {
        "ACCESS-CM2":       "access_cm2",
        "CESM2":            "cesm2",
        "CMCC-ESM2":        "cmcc_esm2",
        "CNRM-CM6-1":       "cnrm_cm6_1",
        "EC-Earth3-CC":     "ec_earth3_cc",
        "EC-Earth3-Veg-LR": "ec_earth3_veg_lr",
        "GFDL-ESM4":        "gfdl_esm4",
        "HadGEM3-GC31-LL":  "hadgem3_gc31_ll",
        "INM-CM5-0":        "inm_cm5_0",
        "IPSL-CM6A-LR":     "ipsl_cm6a_lr",
        "MIROC6":           "miroc6",
        "MPI-ESM1-2-LR":    "mpi_esm1_2_lr",
        "MRI-ESM2-0":       "mri_esm2_0",
        "NorESM2-MM":       "noresm2_mm",
    })

    # Paths.  parents[3] resolves to `_structured_project` when this script
    # lives at `_structured_project/02_processing/scripts_python/projection/`.
    project_root: Path = Path(__file__).resolve().parents[3]
    onm_file:     Path = field(init=False)
    cmip6_dir:    Path = field(init=False)
    out_dir:      Path = field(init=False)

    # Candidate locations for station metadata, tried in order
    meta_candidates: Tuple[Path, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.onm_file  = self.project_root / "01_data" / "external" / "spi_ready_dataset_main_1990_2015.csv"
        self.cmip6_dir = self.project_root / "01_data" / "cmip6"    / "historical"
        self.out_dir   = self.project_root / "04_outputs" / "r2_taylor_cmip6_evaluation"
        (self.out_dir / "tables").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "figures").mkdir(parents=True, exist_ok=True)

        # Try the file we KNOW has clean WGS84 lat/lon first; fall back to others.
        # The metadata loader will reject any candidate whose values fall outside
        # plausible decimal-degree ranges (filters out Lambert-projected coords).
        self.meta_candidates = (
            self.project_root / "03_outputs" / "arcgis" / "spi6_with_coordinates.csv",
            self.project_root / "01_data" / "external" / "master_station_metadata_final_plus_pluvio.csv",
            self.project_root / "01_data" / "external" / "master_station_metadata_candidate.csv",
        )


CFG = Config()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("r2_taylor")


# --------------------------------------------------------------------------- #
# 1. UTILITIES (xarray harmonisation)
# --------------------------------------------------------------------------- #

def _normalize_dim_names(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    if "latitude"   in ds.coords: rename["latitude"]  = "lat"
    if "longitude"  in ds.coords: rename["longitude"] = "lon"
    if "valid_time" in ds.coords and "time" not in ds.coords:
        rename["valid_time"] = "time"
    return ds.rename(rename) if rename else ds


def _wrap_longitude_0_360_to_180(ds: xr.Dataset) -> xr.Dataset:
    if "lon" not in ds.coords:
        return ds
    if float(ds["lon"].max()) > 180.0:
        new_lon = (((ds["lon"] + 180.0) % 360.0) - 180.0)
        ds = ds.assign_coords(lon=new_lon).sortby("lon")
    return ds


def _clip_box(ds: xr.Dataset, nwse: Tuple[float, float, float, float]) -> xr.Dataset:
    n, w, s, e = nwse
    lat = ds["lat"]
    lat_slice = slice(n, s) if float(lat[0]) > float(lat[-1]) else slice(s, n)
    return ds.sel(lat=lat_slice, lon=slice(w, e))


def _regrid_to_common(da: xr.DataArray, grid_deg: float, nwse) -> xr.DataArray:
    n, w, s, e = nwse
    new_lat = np.arange(s, n + 1e-6, grid_deg)
    new_lon = np.arange(w, e + 1e-6, grid_deg)
    return da.interp(lat=new_lat, lon=new_lon, method="linear")


def _to_mm_per_month(da: xr.DataArray) -> xr.DataArray:
    """CMIP6 `pr` is kg m-2 s-1 -> mm/month."""
    days = da["time"].dt.days_in_month
    return da * 86400.0 * days


# --------------------------------------------------------------------------- #
# 2. OBSERVATIONS (per station)
# --------------------------------------------------------------------------- #

def load_station_observations(cfg: Config) -> pd.DataFrame:
    """Wide-format DataFrame: index = month-start datetime, columns = stations."""
    log.info("Loading ONM observations: %s", cfg.onm_file)
    df = pd.read_csv(cfg.onm_file, parse_dates=["date"])

    expected = {"master_station_id", "date", "precip_mm"}
    missing  = expected - set(df.columns)
    if missing:
        raise ValueError(f"ONM file is missing columns: {missing}")

    wide = (df.pivot_table(index="date", columns="master_station_id",
                           values="precip_mm", aggfunc="mean")
              .sort_index())
    wide.index = pd.to_datetime(wide.index).to_period("M").to_timestamp()
    wide = wide.loc[cfg.eval_start: cfg.eval_end]

    log.info("ONM stations found: %d  |  months in window: %d",
             wide.shape[1], wide.shape[0])
    return wide


def _try_load_wgs84(df: pd.DataFrame, id_col: str) -> pd.DataFrame | None:
    """Look for native WGS84 columns in `df` and return (station_id, lat, lon)
    rows that pass the range check, or None if no usable WGS84 is present."""
    LAT_COLS = ("latitude", "lat", "y_lat")
    LON_COLS = ("longitude", "lon", "long", "x_lon")
    lat_col = next((c for c in LAT_COLS if c in df.columns), None)
    lon_col = next((c for c in LON_COLS if c in df.columns), None)
    if not (lat_col and lon_col):
        return None
    lat_v = pd.to_numeric(df[lat_col], errors="coerce")
    lon_v = pd.to_numeric(df[lon_col], errors="coerce")
    valid = lat_v.between(-90.0, 90.0) & lon_v.between(-180.0, 180.0)
    if valid.sum() == 0:
        return None
    return pd.DataFrame({
        "station_id": df.loc[valid, id_col].astype(str).values,
        "lat":        lat_v[valid].values,
        "lon":        lon_v[valid].values,
    })


def _try_load_lambert_nord_algerie(df: pd.DataFrame, id_col: str) -> pd.DataFrame | None:
    """Detect Lambert Nord Algérie columns (in km) and convert to WGS84.

    EPSG:30791 = Nord Algérie (Lambert Conformal Conic on Voirol 1875 datum).
    The ANRH/ONM/IHFR network uses km from the projection origin
    (false_easting=500 km, false_northing=300 km).  We multiply by 1000 to
    pass metres to pyproj.
    """
    X_COLS = ("x", "x_lambert_km", "x_local")
    Y_COLS = ("y", "y_lambert_km", "y_local")
    x_col = next((c for c in X_COLS if c in df.columns), None)
    y_col = next((c for c in Y_COLS if c in df.columns), None)
    if not (x_col and y_col):
        return None

    x_km = pd.to_numeric(df[x_col], errors="coerce")
    y_km = pd.to_numeric(df[y_col], errors="coerce")
    valid = x_km.notna() & y_km.notna()
    if valid.sum() == 0:
        return None

    try:
        import pyproj
    except ImportError:
        log.error("pyproj is required to convert Lambert -> WGS84.  "
                  "Install it with:  pip install pyproj")
        return None

    transformer = pyproj.Transformer.from_crs("EPSG:30791", "EPSG:4326",
                                              always_xy=True)
    lon_v, lat_v = transformer.transform(
        x_km[valid].values * 1000.0,
        y_km[valid].values * 1000.0,
    )

    rng = ((lat_v >= -90.0) & (lat_v <= 90.0)
           & (lon_v >= -180.0) & (lon_v <= 180.0))
    if rng.sum() == 0:
        log.warning("Lambert -> WGS84 conversion produced out-of-range values; "
                    "check that the input is really EPSG:30791 in km")
        return None

    return pd.DataFrame({
        "station_id": df.loc[valid, id_col].astype(str).values[rng],
        "lat":        lat_v[rng],
        "lon":        lon_v[rng],
    })


def load_station_metadata(cfg: Config, station_ids: List[str]) -> pd.DataFrame:
    """Return DataFrame indexed by station_id with WGS84 `lat` and `lon`.

    Tries each candidate metadata file in order; for each file:
      1. Try native WGS84 columns (latitude/longitude or lat/lon).
      2. If absent or all out-of-range, try Lambert Nord Algérie x/y in km
         and convert to WGS84 via EPSG:30791 -> EPSG:4326.
    First file producing matches against the precipitation station_ids wins.
    """
    ID_COLS = ("master_station_id", "station_id", "id", "code")

    for p in cfg.meta_candidates:
        if not p.exists():
            continue
        df = pd.read_csv(p)

        id_col = next((c for c in ID_COLS if c in df.columns), None)
        if id_col is None:
            log.warning("[meta] %s: no id column matching %s", p.name, ID_COLS)
            continue

        result = _try_load_wgs84(df, id_col)
        source_kind = "WGS84"
        if result is None:
            result = _try_load_lambert_nord_algerie(df, id_col)
            source_kind = "Lambert Nord Algérie EPSG:30791 -> WGS84"
        if result is None:
            log.warning("[meta] %s: neither WGS84 nor Lambert columns usable, skipping", p.name)
            continue

        out = (result.drop_duplicates("station_id")
                     .set_index("station_id"))
        out = out.reindex([s for s in station_ids if s in out.index])
        if len(out) == 0:
            log.warning("[meta] %s: no rows match the precipitation station IDs", p.name)
            continue

        log.info("[meta] using %s (%s) — coordinates resolved for %d / %d stations",
                 p.name, source_kind, len(out), len(station_ids))
        return out

    raise FileNotFoundError(
        "No usable station-metadata file found among: "
        + ", ".join(str(p) for p in cfg.meta_candidates)
    )


# --------------------------------------------------------------------------- #
# 3. MODEL EXTRACTION (nearest grid cell at each station)
# --------------------------------------------------------------------------- #

def load_model_field(cfg: Config, model_folder: str) -> xr.DataArray | None:
    """Return CMIP6 historical precipitation in mm/month over the basin domain.

    No regridding to a common lattice is performed.  Each model is sampled at
    its NATIVE nearest grid cell to the station coordinates downstream — this
    is a suitable treatment for station-scale evaluation
    (we ask "what does the model produce at this location at its own
    resolution?") and it avoids a known NaN-poisoning artefact: when a model's
    native grid is coarse (e.g. MPI-ESM1-2-LR at T63 ≈ 1.875°), CDS subsetting
    can leave only one or two source cells inside the basin box, and bilinear
    interpolation to a 0.25° common lattice then returns NaN at points outside
    the source-cell bracket — silently dropping that model from most stations.
    """
    folder = cfg.cmip6_dir / model_folder
    files = sorted(folder.glob("pr_*.nc"))
    if not files:
        log.warning("[%s] no pr_*.nc under %s", model_folder, folder)
        return None

    ds = xr.open_mfdataset(files, combine="by_coords",
                           use_cftime=True, chunks={"time": 60})
    ds = _normalize_dim_names(ds)
    ds = _wrap_longitude_0_360_to_180(ds)

    da = ds["pr"]
    da = _clip_box(da, cfg.bbox_nwse)

    # Calendar harmonisation (CMIP6 uses noleap / 360_day / proleptic_gregorian)
    if hasattr(da["time"].values[0], "calendar") or da["time"].dtype == object:
        da = da.convert_calendar("standard", align_on="date")

    da = _to_mm_per_month(da)
    return da


def _force_pandas_datetime_index(s: pd.Series) -> pd.Series:
    """Force `s` to have a month-start `pd.DatetimeIndex`, robust to CMIP6 calendars.

    `xr.open_mfdataset(..., use_cftime=True)` returns cftime objects whose
    calendar may be `noleap` (ACCESS, MIROC, GFDL...), `360_day` (HadGEM3),
    `proleptic_gregorian`, etc.  After `convert_calendar("standard")` they are
    still cftime objects with `calendar="standard"`, and `pd.to_datetime` on
    them is unreliable across calendars — silent NaTs or shifted dates that
    break alignment with the ONM month-start index.

    The robust route is to read each timestamp's `.year` and `.month` (every
    cftime variant exposes these, as does `pd.Timestamp` for numpy datetime64),
    discard the day, and rebuild a `YYYY-MM-01` index.  This produces a
    consistent month-start alignment regardless of the source calendar.
    """
    if isinstance(s.index, pd.DatetimeIndex):
        s.index = s.index.to_period("M").to_timestamp()
        return s

    yms: List[Tuple[int, int]] = []
    for t in s.index:
        try:
            yms.append((int(t.year), int(t.month)))
        except AttributeError:
            ts = pd.Timestamp(t)
            yms.append((ts.year, ts.month))

    s.index = pd.to_datetime([f"{y:04d}-{m:02d}-01" for y, m in yms])
    return s


def extract_at_stations(da: xr.DataArray, meta: pd.DataFrame,
                        cfg: Config) -> Dict[str, pd.Series]:
    """Sample `da` at each station's coordinates (nearest grid cell)."""
    out: Dict[str, pd.Series] = {}
    for sid, row in meta.iterrows():
        s = (da.sel(lat=float(row["lat"]), lon=float(row["lon"]),
                    method="nearest")
               .to_pandas())
        s = _force_pandas_datetime_index(s)
        s = s.loc[cfg.eval_start: cfg.eval_end]
        s.name = sid
        out[sid] = s
    return out


# --------------------------------------------------------------------------- #
# 4. METRICS
# --------------------------------------------------------------------------- #

def taylor_skill_score(rho: float, sigma_ratio: float, rho_max: float = 1.0) -> float:
    if sigma_ratio <= 0 or np.isnan(sigma_ratio):
        return np.nan
    num = 4.0 * (1.0 + rho) ** 4
    den = (sigma_ratio + 1.0 / sigma_ratio) ** 2 * (1.0 + rho_max) ** 4
    return num / den


def annual_cycle_corr(model: pd.Series, obs: pd.Series) -> float:
    cm = model.groupby(model.index.month).mean()
    co = obs.groupby(obs.index.month).mean()
    return float(np.corrcoef(cm.values, co.values)[0, 1])


def dry_freq(s: pd.Series, threshold_mm: float) -> float:
    return float((s < threshold_mm).mean())


def metric_row(station_id: str, model_name: str,
               model: pd.Series, obs: pd.Series, cfg: Config) -> Dict[str, float]:
    df = pd.concat([obs.rename("obs"), model.rename("mod")], axis=1).dropna()
    if len(df) < 24:
        log.warning("[%s / %s] only %d aligned months (obs n=%d, mod n=%d) — "
                    "skipping. Check that obs and model time indexes overlap.",
                    station_id, model_name, len(df),
                    int(obs.notna().sum()), int(model.notna().sum()))
        return {"station": station_id, "model": model_name, "n": len(df)}
    o, m = df["obs"].values, df["mod"].values

    sigma_o = float(o.std(ddof=1))
    sigma_m = float(m.std(ddof=1))
    rho     = float(np.corrcoef(o, m)[0, 1])
    crmse   = float(np.sqrt(sigma_m**2 + sigma_o**2 - 2 * sigma_m * sigma_o * rho))
    bias    = float((m.mean() - o.mean()) / o.mean()) if o.mean() != 0 else np.nan
    cycle_r = annual_cycle_corr(df["mod"], df["obs"])
    dfreq_e = abs(dry_freq(df["mod"], cfg.dry_month_threshold_mm) -
                  dry_freq(df["obs"], cfg.dry_month_threshold_mm))
    s2      = taylor_skill_score(rho, sigma_m / sigma_o if sigma_o > 0 else np.nan)

    composite = (cfg.w_taylor   * (s2 if np.isfinite(s2) else 0.0)
                 + cfg.w_bias    * (1.0 - min(abs(bias) if np.isfinite(bias) else 1.0, 1.0))
                 + cfg.w_cycle   * max(cycle_r if np.isfinite(cycle_r) else 0.0, 0.0)
                 + cfg.w_dry_freq * (1.0 - min(dfreq_e, 1.0)))

    return {
        "station":      station_id,
        "model":        model_name,
        "n":            len(df),
        "sigma_obs":    sigma_o,
        "sigma_model":  sigma_m,
        "sigma_ratio":  sigma_m / sigma_o if sigma_o > 0 else np.nan,
        "rho":          rho,
        "crmse":        crmse,
        "bias_rel":     bias,
        "cycle_corr":   cycle_r,
        "dryfreq_err":  dfreq_e,
        "taylor_s2":    s2,
        "composite_S":  composite,
    }


# --------------------------------------------------------------------------- #
# 5. TAYLOR DIAGRAM (re-usable for sub-panels)
# --------------------------------------------------------------------------- #

class TaylorPanel:
    """A single Taylor sub-panel placed inside an outer figure layout."""

    def __init__(self, fig: plt.Figure, rect: int,
                 sigma_max: float = 1.6, label: str | None = None,
                 title:    str | None = None) -> None:
        tr = PolarAxes.PolarTransform()

        rho_ticks   = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
        theta_ticks = np.arccos(rho_ticks)
        gl1 = grid_finder.FixedLocator(theta_ticks)
        tf1 = grid_finder.DictFormatter(dict(zip(theta_ticks, [str(r) for r in rho_ticks])))

        ghelper = floating_axes.GridHelperCurveLinear(
            tr,
            extremes=(0, np.pi / 2, 0, sigma_max),
            grid_locator1=gl1, tick_formatter1=tf1,
        )
        ax = floating_axes.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("ρ")

        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text("σ / σ_obs")

        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")

        ax.axis["bottom"].set_visible(False)
        ax.grid(True)

        self._ax = ax
        self.ax  = ax.get_aux_axes(tr)

        # Reference point + unit-STD arc
        self.ax.plot(0, 1.0, "k*", ms=10, label=label or "obs")
        t = np.linspace(0, np.pi / 2, 100)
        self.ax.plot(t, np.ones_like(t), "k--", lw=1, alpha=0.5)

        # RMSE iso-contours
        rs, ts = np.meshgrid(np.linspace(0, sigma_max, 100),
                             np.linspace(0, np.pi / 2, 100))
        rmse = np.sqrt(1.0 + rs**2 - 2.0 * rs * np.cos(ts))
        cs = self.ax.contour(ts, rs, rmse,
                             levels=[0.5, 1.0, 1.5],
                             colors="0.7", linewidths=0.6)
        self.ax.clabel(cs, inline=True, fontsize=6, fmt="%.1f")

        if title:
            ax.set_title(title, fontsize=9, pad=18)

    def add_model(self, sigma_ratio: float, rho: float, name: str,
                  marker: str = "o", color: str | None = None) -> None:
        if not (np.isfinite(sigma_ratio) and np.isfinite(rho)):
            return
        theta = np.arccos(np.clip(rho, -1.0, 1.0))
        self.ax.plot(theta, sigma_ratio, marker=marker, ms=6,
                     color=color, label=name)


def plot_taylor_per_station(metrics: pd.DataFrame, stations: List[str],
                             out_path: Path) -> None:
    n_stations = len(stations)
    n_cols = 3
    n_rows = int(np.ceil(n_stations / n_cols))

    fig = plt.figure(figsize=(4.5 * n_cols, 4.0 * n_rows))
    sigma_max = max(1.8, float(metrics["sigma_ratio"].max(skipna=True)) * 1.05)
    cmap = plt.get_cmap("tab20")
    model_names = metrics["model"].unique().tolist()

    for i, sid in enumerate(stations):
        rect = int(f"{n_rows}{n_cols}{i + 1}")
        panel = TaylorPanel(fig, rect, sigma_max=sigma_max,
                            title=str(sid))
        sub = metrics[metrics["station"] == sid].reset_index(drop=True)
        for j, row in sub.iterrows():
            panel.add_model(row["sigma_ratio"], row["rho"], row["model"],
                            color=cmap(model_names.index(row["model"]) % 20))

    # Single legend for the whole figure (use the last panel's handles)
    handles, labels = panel.ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, 0.0), fontsize=8, frameon=False)
    fig.suptitle("Taylor diagrams — CMIP6 vs ONM, monthly precipitation, "
                 "1990–2014\n(nearest grid cell at each station)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_skill_heatmap(metrics: pd.DataFrame, out_path: Path) -> None:
    pivot = metrics.pivot(index="station", columns="model", values="composite_S")
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=("white" if v < 0.55 else "black"), fontsize=7)
    fig.colorbar(im, ax=ax, label="Composite skill score S")
    ax.set_title("Composite skill score by station × model", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_annual_cycles_per_station(obs_wide: pd.DataFrame,
                                   model_series_per_station: Dict[str, Dict[str, pd.Series]],
                                   stations: List[str],
                                   out_path: Path) -> None:
    n_cols = 3
    n_rows = int(np.ceil(len(stations) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.0 * n_cols, 2.7 * n_rows),
                             sharex=True)
    axes = axes.flatten()
    months = np.arange(1, 13)
    cmap = plt.get_cmap("tab20")

    for i, sid in enumerate(stations):
        ax = axes[i]
        obs = obs_wide[sid]
        ax.plot(months, obs.groupby(obs.index.month).mean().reindex(months).values,
                "k-", lw=2, label="ONM")
        per_model = model_series_per_station.get(sid, {})
        for j, (mname, s) in enumerate(per_model.items()):
            ax.plot(months,
                    s.groupby(s.index.month).mean().reindex(months).values,
                    color=cmap(j % 20), alpha=0.8, lw=0.9)
        ax.set_title(sid, fontsize=9)
        ax.set_xticks(months)
        ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"],
                            fontsize=7)
        ax.tick_params(axis="y", labelsize=7)

    for k in range(len(stations), len(axes)):
        axes[k].set_visible(False)

    fig.suptitle("Annual cycle of monthly precipitation, 1990–2014",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_bias_dryfreq(metrics: pd.DataFrame, out_path: Path) -> None:
    bias_pivot = metrics.pivot(index="station", columns="model", values="bias_rel")
    df_pivot   = metrics.pivot(index="station", columns="model", values="dryfreq_err")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    for ax, df_, title, vmin, vmax, cmap in [
        (axes[0], bias_pivot * 100, "Mean relative bias (%)",      -100, 100, "RdBu_r"),
        (axes[1], df_pivot   * 100, "Dry-month frequency error (%-pts)", 0,   30, "magma_r"),
    ]:
        im = ax.imshow(df_.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(df_.shape[1]))
        ax.set_xticklabels(df_.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(df_.shape[0]))
        ax.set_yticklabels(df_.index, fontsize=8)
        for i in range(df_.shape[0]):
            for j in range(df_.shape[1]):
                v = df_.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.0f}" if "bias" in title else f"{v:.0f}",
                            ha="center", va="center", fontsize=6,
                            color=("white" if abs(v) > (vmax - vmin) * 0.4 else "black"))
        fig.colorbar(im, ax=ax)
        ax.set_title(title, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 6. SELECTION
# --------------------------------------------------------------------------- #

def per_station_best3(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, sub in metrics.groupby("station"):
        top = (sub.sort_values("composite_S", ascending=False)
                  .head(3)
                  .reset_index(drop=True))
        for rank, row in top.iterrows():
            rows.append({
                "station":      sid,
                "rank":         rank + 1,
                "model":        row["model"],
                "composite_S":  row["composite_S"],
                "rho":          row["rho"],
                "sigma_ratio":  row["sigma_ratio"],
                "bias_rel":     row["bias_rel"],
                "cycle_corr":   row["cycle_corr"],
                "dryfreq_err":  row["dryfreq_err"],
            })
    return pd.DataFrame(rows)


def global_best3(metrics: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Average rank of composite_S across stations, then apply the guards."""
    rank_per_station = (metrics
                        .assign(rank=lambda d: d.groupby("station")["composite_S"]
                                                .rank(ascending=False, method="min"))
                        .pivot(index="model", columns="station", values="rank"))
    mean_rank = rank_per_station.mean(axis=1).rename("mean_rank")
    mean_S    = (metrics.groupby("model")["composite_S"].mean()
                        .rename("mean_composite_S"))
    mean_bias = metrics.groupby("model")["bias_rel"].mean().rename("mean_bias_rel")
    mean_cyc  = metrics.groupby("model")["cycle_corr"].mean().rename("mean_cycle_corr")
    mean_df   = metrics.groupby("model")["dryfreq_err"].mean().rename("mean_dryfreq_err")

    summary = (pd.concat([mean_rank, mean_S, mean_bias, mean_cyc, mean_df], axis=1)
                 .sort_values("mean_rank"))

    selected, notes = [], []
    for model_name, row in summary.iterrows():
        ok, why = True, []
        if abs(row["mean_bias_rel"])  > cfg.guard_abs_bias_max:
            ok = False; why.append(f"|bias|={row['mean_bias_rel']:+.0%}")
        if row["mean_cycle_corr"]     < cfg.guard_cycle_corr_min:
            ok = False; why.append(f"cycle_corr={row['mean_cycle_corr']:.2f}")
        if row["mean_dryfreq_err"]    > cfg.guard_dryfreq_err_max:
            ok = False; why.append(f"dryfreq_err={row['mean_dryfreq_err']:.2f}")
        notes.append("retained" if ok else "dropped: " + "; ".join(why))
        if ok:
            selected.append(dict(model=model_name, **row.to_dict(), note="retained"))
        else:
            selected.append(dict(model=model_name, **row.to_dict(),
                                 note="dropped: " + "; ".join(why)))
        if sum(1 for s in selected if s["note"] == "retained") == 3:
            break

    out = pd.DataFrame(selected)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


# --------------------------------------------------------------------------- #
# 7. MAIN
# --------------------------------------------------------------------------- #

def main() -> None:
    log.info("Project root: %s",          CFG.project_root)
    log.info("Bounding box (N,W,S,E): %s", CFG.bbox_nwse)
    log.info("Evaluation period: %s .. %s", CFG.eval_start, CFG.eval_end)
    log.info("Reference dataset: ONM rain-gauge network (per-station evaluation)")

    obs_wide = load_station_observations(CFG)
    stations = list(obs_wide.columns)
    if len(stations) == 0:
        log.error("No stations found in the ONM file."); return

    meta = load_station_metadata(CFG, stations)
    stations_with_coords = list(meta.index)
    obs_wide = obs_wide[stations_with_coords]
    log.info("Stations evaluated: %d", len(stations_with_coords))
    for sid in stations_with_coords:
        log.info("    %-25s lat=%6.3f  lon=%6.3f", sid,
                 meta.loc[sid, "lat"], meta.loc[sid, "lon"])

    rows: List[Dict] = []
    series_per_station: Dict[str, Dict[str, pd.Series]] = {s: {} for s in stations_with_coords}

    for display, folder in CFG.models.items():
        try:
            field_da = load_model_field(CFG, folder)
        except Exception as exc:
            log.error("[%s] failed to load: %s", display, exc); continue
        if field_da is None:
            continue

        per_station = extract_at_stations(field_da, meta, CFG)
        for sid, s_mod in per_station.items():
            row = metric_row(sid, display, s_mod, obs_wide[sid], CFG)
            rows.append(row)
            series_per_station[sid][display] = s_mod

        # quick log
        avg_S = np.nanmean([r["composite_S"] for r in rows
                            if r.get("model") == display
                            and "composite_S" in r])
        log.info("[%-15s] mean composite S across stations = %.3f", display, avg_S)

    if not rows:
        log.error("No metrics produced — check CMIP6 paths."); return

    metrics = pd.DataFrame(rows)
    metrics_path = CFG.out_dir / "tables" / "skill_metrics.csv"
    metrics.to_csv(metrics_path, index=False, float_format="%.4f")
    log.info("Skill metrics (long)        -> %s", metrics_path)

    best_per_station = per_station_best3(metrics)
    best_per_station.to_csv(CFG.out_dir / "tables" / "best_three_per_station.csv",
                            index=False, float_format="%.4f")
    log.info("Per-station best 3          -> tables/best_three_per_station.csv")

    best_global = global_best3(metrics, CFG)
    best_global.to_csv(CFG.out_dir / "tables" / "best_three_global.csv",
                       index=False, float_format="%.4f")
    log.info("Global best 3 (by mean rank)-> tables/best_three_global.csv")

    plot_taylor_per_station(metrics, stations_with_coords,
                             CFG.out_dir / "figures" / "taylor_per_station.png")
    plot_skill_heatmap(metrics, CFG.out_dir / "figures" / "skill_heatmap.png")
    plot_annual_cycles_per_station(obs_wide, series_per_station, stations_with_coords,
                                    CFG.out_dir / "figures" / "annual_cycles.png")
    plot_bias_dryfreq(metrics, CFG.out_dir / "figures" / "bias_dryfreq.png")
    log.info("Figures                     -> %s", CFG.out_dir / "figures")

    print("\n=== Per-station top-3 (by composite S) ===")
    print(best_per_station.to_string(index=False, float_format="%.3f"))

    print("\n=== Global top-3 retained for SSP projection ===")
    print(best_global.head(6).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
