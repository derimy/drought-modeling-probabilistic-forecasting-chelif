"""
Author        : Imene DERRAR
Supervisor    : Abdelillah OTMANE CHERIF
Institutions  : NHSM — National Higher School of Mathematics
                IHFR — Institut Hydrométéorologique de Formation et de Recherche
Academic year : 2025–2026
Project       : Final-year engineering thesis in Applied Mathematics —
                Drought modelling and probabilistic forecasting for the Chélif basin

--------------------------------------------------------------------------------

R3 — Bias correction (Empirical Quantile Mapping)
==========================================================

For each retained CMIP6 model and each ONM station, calibrate an empirical
quantile-mapping (EQM) function on the historical period (1990–2014) and
apply it to the SSP runs (2015–2040), month by calendar month.  Output:
bias-corrected monthly precipitation ready for SPI computation in R4.

Why per calendar month
----------------------
Mediterranean precipitation has a strong annual cycle (winter wet, summer
dry).  A bias correction calibrated on the pooled monthly distribution would
underweight the rare, intense winter months and dilute the calibration with
abundant near-zero summer values.  Calibrating one EQM per calendar month
preserves the seasonal structure of the bias.

Why EQM (not delta or scaling)
------------------------------
Two reasons.  (1) EQM corrects the *full distribution* of precipitation, not
just the mean — essential for SPI, whose tails determine drought class.
(2) EQM does not assume a parametric distribution; it maps empirical
quantiles directly, which is robust at small sample sizes (25 years × 1
calendar month = 25 values per quantile bucket — limited, but sufficient for empirical quantile estimation).

Pipeline
--------
1.  Load ONM monthly precipitation per station (same source as R2).
2.  Load station coordinates and convert Lambert Nord Algérie -> WGS84.
3.  For each retained model {EC-Earth3-Veg-LR, EC-Earth3-CC, CESM2}:
      a. Open the historical NetCDF.  Extract the time series at the nearest
         native grid cell to each station.
      b. For each SSP scenario {ssp2_4_5, ssp5_8_5}: open the projection
         NetCDF, extract at the same station coordinate.
4.  For each (station, model, scenario) and each calendar month:
      - Calibrate EQM on (obs_hist, mod_hist) for 1990–2014, that month only.
      - Apply EQM to (mod_ssp) for 2015–2040, that month only.
5.  Stack everything to a long DataFrame and write to disk.
6.  Plot diagnostics: CDFs (raw vs corrected vs obs) and SSP trajectories.

Inputs
------
    01_data/external/spi_ready_dataset_main_1990_2015.csv
    01_data/external/master_station_metadata_final_plus_pluvio.csv
    01_data/cmip6/historical/<model>/pr_*.nc
    01_data/cmip6/ssp2_4_5/<model>/pr_*.nc
    01_data/cmip6/ssp5_8_5/<model>/pr_*.nc

Outputs (under 04_outputs/R3_bias_correction/)
    tables/bias_corrected_long.csv      Long format: station, model, scenario,
                                        date, pr_raw_mm, pr_corrected_mm.
    tables/eqm_summary.csv              Per (station × model × month): obs_mean,
                                        mod_hist_mean, mod_corrected_hist_mean,
                                        residual sanity check.
    figures/eqm_cdfs.png                CDF before/after for one station per model.
    figures/ssp_trajectories.png        9 panels (stations) × 2 scenarios.
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

# --------------------------------------------------------------------------- #
# 0. CONFIGURATION
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    bbox_nwse: Tuple[float, float, float, float] = (36.5, 0.0, 34.0, 3.5)

    calib_start: str = "1990-01"
    calib_end:   str = "2014-12"
    proj_start:  str = "2015-01"
    proj_end:    str = "2040-12"

    # Retained models selected from the R2 evaluation stage.
    retained_models: Dict[str, str] = field(default_factory=lambda: {
        "EC-Earth3-Veg-LR": "ec_earth3_veg_lr",
        "EC-Earth3-CC":     "ec_earth3_cc",
        "CESM2":            "cesm2",
    })

    scenarios: Tuple[str, ...] = ("historical", "ssp2_4_5", "ssp5_8_5")

    project_root: Path = Path(__file__).resolve().parents[3]
    onm_file:     Path = field(init=False)
    cmip6_root:   Path = field(init=False)
    out_dir:      Path = field(init=False)
    meta_candidates: Tuple[Path, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.onm_file   = self.project_root / "01_data" / "external" / "spi_ready_dataset_main_1990_2015.csv"
        self.cmip6_root = self.project_root / "01_data" / "cmip6"
        self.out_dir    = self.project_root / "04_outputs" / "R3_bias_correction"
        (self.out_dir / "tables").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "figures").mkdir(parents=True, exist_ok=True)

        self.meta_candidates = (
            self.project_root / "03_outputs" / "arcgis" / "spi6_with_coordinates.csv",
            self.project_root / "01_data"    / "external" / "master_station_metadata_final_plus_pluvio.csv",
            self.project_root / "01_data"    / "external" / "master_station_metadata_candidate.csv",
        )


CFG = Config()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("r3_bias")


# --------------------------------------------------------------------------- #
# 1. UTILITIES (duplicated from R2 to keep this script self-contained)
# --------------------------------------------------------------------------- #

def _normalize_dim_names(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    if "latitude"   in ds.coords: rename["latitude"]  = "lat"
    if "longitude"  in ds.coords: rename["longitude"] = "lon"
    if "valid_time" in ds.coords and "time" not in ds.coords:
        rename["valid_time"] = "time"
    return ds.rename(rename) if rename else ds


def _wrap_longitude_0_360_to_180(ds: xr.Dataset) -> xr.Dataset:
    if "lon" not in ds.coords: return ds
    if float(ds["lon"].max()) > 180.0:
        new_lon = (((ds["lon"] + 180.0) % 360.0) - 180.0)
        ds = ds.assign_coords(lon=new_lon).sortby("lon")
    return ds


def _clip_box(ds: xr.Dataset, nwse) -> xr.Dataset:
    n, w, s, e = nwse
    lat = ds["lat"]
    lat_slice = slice(n, s) if float(lat[0]) > float(lat[-1]) else slice(s, n)
    return ds.sel(lat=lat_slice, lon=slice(w, e))


def _to_mm_per_month(da: xr.DataArray) -> xr.DataArray:
    days = da["time"].dt.days_in_month
    return da * 86400.0 * days


def _force_pandas_datetime_index(s: pd.Series) -> pd.Series:
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


def load_station_observations(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.onm_file, parse_dates=["date"])
    wide = (df.pivot_table(index="date", columns="master_station_id",
                           values="precip_mm", aggfunc="mean")
              .sort_index())
    wide.index = pd.to_datetime(wide.index).to_period("M").to_timestamp()
    return wide


def load_station_metadata(cfg: Config, station_ids: List[str]) -> pd.DataFrame:
    """Lambert Nord Algérie -> WGS84 with range guard, identical to R2."""
    ID_COLS = ("master_station_id", "station_id", "id", "code")

    def _try_wgs84(df: pd.DataFrame, id_col: str):
        for lat_col in ("latitude", "lat"):
            for lon_col in ("longitude", "lon"):
                if lat_col in df.columns and lon_col in df.columns:
                    lat_v = pd.to_numeric(df[lat_col], errors="coerce")
                    lon_v = pd.to_numeric(df[lon_col], errors="coerce")
                    valid = lat_v.between(-90, 90) & lon_v.between(-180, 180)
                    if valid.sum() > 0:
                        return pd.DataFrame({
                            "station_id": df.loc[valid, id_col].astype(str).values,
                            "lat":        lat_v[valid].values,
                            "lon":        lon_v[valid].values,
                        })
        return None

    def _try_lambert(df: pd.DataFrame, id_col: str):
        for x_col in ("x", "x_lambert_km", "x_local"):
            for y_col in ("y", "y_lambert_km", "y_local"):
                if x_col in df.columns and y_col in df.columns:
                    x_km = pd.to_numeric(df[x_col], errors="coerce")
                    y_km = pd.to_numeric(df[y_col], errors="coerce")
                    valid = x_km.notna() & y_km.notna()
                    if valid.sum() > 0:
                        import pyproj
                        tr = pyproj.Transformer.from_crs("EPSG:30791", "EPSG:4326",
                                                         always_xy=True)
                        lon_v, lat_v = tr.transform(
                            x_km[valid].values * 1000.0,
                            y_km[valid].values * 1000.0,
                        )
                        rng = ((lat_v >= -90) & (lat_v <= 90)
                               & (lon_v >= -180) & (lon_v <= 180))
                        if rng.sum() > 0:
                            return pd.DataFrame({
                                "station_id": df.loc[valid, id_col].astype(str).values[rng],
                                "lat":        lat_v[rng],
                                "lon":        lon_v[rng],
                            })
        return None

    for p in cfg.meta_candidates:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        id_col = next((c for c in ID_COLS if c in df.columns), None)
        if id_col is None:
            continue
        result = _try_wgs84(df, id_col) or _try_lambert(df, id_col)
        if result is None:
            continue
        out = (result.drop_duplicates("station_id")
                     .set_index("station_id")
                     .reindex([s for s in station_ids if s in result["station_id"].values]))
        if len(out) == 0:
            continue
        log.info("[meta] %s -> %d / %d stations resolved", p.name,
                 len(out), len(station_ids))
        return out
    raise FileNotFoundError("No usable station-metadata file.")


def load_model_field_native(cfg: Config, model_folder: str,
                             scenario: str) -> xr.DataArray | None:
    """CMIP6 precipitation as mm/month at native resolution within the basin box."""
    folder = cfg.cmip6_root / scenario / model_folder
    files = sorted(folder.glob("pr_*.nc"))
    if not files:
        log.warning("[%s/%s] no pr_*.nc under %s", model_folder, scenario, folder)
        return None

    ds = xr.open_mfdataset(files, combine="by_coords",
                           use_cftime=True, chunks={"time": 60})
    ds = _normalize_dim_names(ds)
    ds = _wrap_longitude_0_360_to_180(ds)
    da = ds["pr"]
    da = _clip_box(da, cfg.bbox_nwse)
    if hasattr(da["time"].values[0], "calendar") or da["time"].dtype == object:
        da = da.convert_calendar("standard", align_on="date")
    da = _to_mm_per_month(da)
    return da


def extract_at_stations(da: xr.DataArray, meta: pd.DataFrame) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for sid, row in meta.iterrows():
        s = (da.sel(lat=float(row["lat"]), lon=float(row["lon"]),
                    method="nearest").to_pandas())
        out[sid] = _force_pandas_datetime_index(s)
    return out


# --------------------------------------------------------------------------- #
# 2. EMPIRICAL QUANTILE MAPPING
# --------------------------------------------------------------------------- #

def empirical_quantile_map(
    obs_calib:  np.ndarray,
    mod_calib:  np.ndarray,
    mod_target: np.ndarray,
) -> np.ndarray:
    """Empirical Quantile Mapping with linear interpolation between empirical quantiles.

    For each value x in `mod_target`:
      1. Compute the cumulative probability F_mod(x) on the calibration model
         distribution `mod_calib` via linear interpolation between sorted
         calibration values.
      2. Map that probability through the obs CDF: x_corrected = F_obs⁻¹(F_mod(x)).

    Out-of-range values (x below the calibration min or above the calibration
    max) are extrapolated by clipping to the boundary obs values — a
    conservative choice for monthly precipitation that avoids unphysical
    negative values and prevents the bias correction from amplifying tail
    samples beyond the obs envelope.
    """
    obs_calib  = obs_calib[~np.isnan(obs_calib)]
    mod_calib  = mod_calib[~np.isnan(mod_calib)]

    if obs_calib.size < 5 or mod_calib.size < 5:
        return np.full_like(mod_target, np.nan, dtype=float)

    obs_sorted = np.sort(obs_calib)
    mod_sorted = np.sort(mod_calib)

    n_obs = obs_sorted.size
    n_mod = mod_sorted.size
    p_obs = (np.arange(n_obs) + 0.5) / n_obs
    p_mod = (np.arange(n_mod) + 0.5) / n_mod

    p_target  = np.interp(mod_target, mod_sorted, p_mod,
                          left=p_mod[0], right=p_mod[-1])
    corrected = np.interp(p_target, p_obs, obs_sorted,
                          left=obs_sorted[0], right=obs_sorted[-1])

    corrected = np.clip(corrected, 0.0, None)
    return corrected


def correct_monthly(
    obs_series: pd.Series,
    mod_calib:  pd.Series,
    mod_target: pd.Series,
    calib_window: Tuple[str, str],
) -> pd.Series:
    """Apply EQM month-by-calendar-month, returning a Series aligned to mod_target."""
    o_calib = obs_series.loc[calib_window[0]: calib_window[1]]
    m_calib = mod_calib.loc[calib_window[0]: calib_window[1]]

    out = pd.Series(np.nan, index=mod_target.index, dtype=float)
    for month in range(1, 13):
        oc = o_calib[o_calib.index.month == month].dropna().values
        mc = m_calib[m_calib.index.month == month].dropna().values
        mask = (mod_target.index.month == month)
        mt = mod_target[mask].values
        out.loc[mask] = empirical_quantile_map(oc, mc, mt)
    return out


# --------------------------------------------------------------------------- #
# 3. PLOTTING
# --------------------------------------------------------------------------- #

def plot_eqm_cdfs(obs: Dict[str, pd.Series], mod_hist: Dict[str, Dict[str, pd.Series]],
                  mod_corrected_hist: Dict[str, Dict[str, pd.Series]],
                  station_id: str, out_path: Path) -> None:
    """One row per retained model: CDF of obs / mod-raw / mod-corrected at one station."""
    models = list(mod_hist.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 4),
                              sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        o = np.sort(obs[station_id].dropna().values)
        r = np.sort(mod_hist[m][station_id].dropna().values)
        c = np.sort(mod_corrected_hist[m][station_id].dropna().values)
        ax.plot(o, np.linspace(0, 1, len(o)), "k",  lw=2.0, label="ONM obs")
        ax.plot(r, np.linspace(0, 1, len(r)), "C3", lw=1.2, label="model raw")
        ax.plot(c, np.linspace(0, 1, len(c)), "C0--", lw=1.2, label="model bias-corrected")
        ax.set_xlabel("Monthly precipitation (mm)")
        ax.set_title(m, fontsize=10)
        ax.set_xlim(left=0)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Cumulative probability")
    fig.suptitle(f"EQM diagnostic at {station_id} — historical 1990–2014",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_ssp_trajectories(corrected: pd.DataFrame, stations: List[str],
                          out_path: Path) -> None:
    """3 x 3 station panels, each showing the bias-corrected SSP trajectories."""
    n_cols = 3
    n_rows = int(np.ceil(len(stations) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 2.7 * n_rows),
                             sharey=True)
    axes = axes.flatten()

    palette = {
        ("EC-Earth3-Veg-LR", "ssp2_4_5"): ("C0", "-"),
        ("EC-Earth3-Veg-LR", "ssp5_8_5"): ("C0", "--"),
        ("EC-Earth3-CC",     "ssp2_4_5"): ("C2", "-"),
        ("EC-Earth3-CC",     "ssp5_8_5"): ("C2", "--"),
        ("CESM2",            "ssp2_4_5"): ("C1", "-"),
        ("CESM2",            "ssp5_8_5"): ("C1", "--"),
    }

    for i, sid in enumerate(stations):
        ax = axes[i]
        sub = corrected[(corrected["station"] == sid)
                        & (corrected["scenario"] != "historical")]
        for (model, scenario), grp in sub.groupby(["model", "scenario"]):
            grp_yr = (grp.set_index("date")["pr_corrected_mm"]
                         .resample("YE").sum())
            color, linestyle = palette.get((model, scenario), ("k", "-"))
            label = f"{model} / {scenario}"
            ax.plot(grp_yr.index.year, grp_yr.values,
                    color=color, linestyle=linestyle, lw=1.0, alpha=0.8,
                    label=label if i == 0 else None)
        ax.set_title(sid, fontsize=9)
        ax.tick_params(labelsize=7)
    for k in range(len(stations), len(axes)):
        axes[k].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               frameon=False)
    fig.suptitle("Bias-corrected annual precipitation, 2015–2040 (SSPs)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. MAIN
# --------------------------------------------------------------------------- #

def main() -> None:
    log.info("Project root:     %s", CFG.project_root)
    log.info("Calibration:      %s .. %s", CFG.calib_start, CFG.calib_end)
    log.info("Projection:       %s .. %s", CFG.proj_start,  CFG.proj_end)
    log.info("Retained models:  %s", list(CFG.retained_models))

    obs_wide = load_station_observations(CFG)
    obs_wide = obs_wide.loc[CFG.calib_start: CFG.calib_end]
    stations = list(obs_wide.columns)
    if not stations:
        log.error("No ONM stations found."); return

    meta = load_station_metadata(CFG, stations)
    stations = list(meta.index)
    obs = {sid: obs_wide[sid] for sid in stations}

    long_rows: List[Dict] = []
    mod_hist:  Dict[str, Dict[str, pd.Series]] = {}
    mod_corrected_hist: Dict[str, Dict[str, pd.Series]] = {}

    for display, folder in CFG.retained_models.items():
        log.info("=== Model: %s (%s) ===", display, folder)
        per_scenario: Dict[str, Dict[str, pd.Series]] = {}
        for scenario in CFG.scenarios:
            da = load_model_field_native(CFG, folder, scenario)
            if da is None:
                log.warning("[%s/%s] not loaded — skipping", display, scenario)
                continue
            per_scenario[scenario] = extract_at_stations(da, meta)

        if "historical" not in per_scenario:
            log.error("[%s] missing historical run — cannot calibrate EQM.", display)
            continue

        mod_hist[display] = per_scenario["historical"]
        mod_corrected_hist[display] = {}

        for sid in stations:
            obs_s = obs[sid]
            mod_h = per_scenario["historical"].get(sid)
            if mod_h is None:
                continue

            # Bias-correct the historical run on itself (sanity / appendix).
            corr_h = correct_monthly(obs_s, mod_h, mod_h,
                                     (CFG.calib_start, CFG.calib_end))
            corr_h = corr_h.loc[CFG.calib_start: CFG.calib_end]
            mod_corrected_hist[display][sid] = corr_h
            for d, raw, c in zip(mod_h.index, mod_h.values, corr_h.values):
                if CFG.calib_start <= str(d)[:7] <= CFG.calib_end:
                    long_rows.append({
                        "station":          sid,
                        "model":            display,
                        "scenario":         "historical",
                        "date":             d,
                        "pr_raw_mm":        float(raw)  if np.isfinite(raw)  else np.nan,
                        "pr_corrected_mm":  float(c)    if np.isfinite(c)    else np.nan,
                    })

            for scenario in ("ssp2_4_5", "ssp5_8_5"):
                if scenario not in per_scenario:
                    continue
                mod_f = per_scenario[scenario].get(sid)
                if mod_f is None:
                    continue
                mod_f_proj = mod_f.loc[CFG.proj_start: CFG.proj_end]
                corr_f = correct_monthly(obs_s, mod_h, mod_f_proj,
                                         (CFG.calib_start, CFG.calib_end))
                for d, raw, c in zip(mod_f_proj.index, mod_f_proj.values, corr_f.values):
                    long_rows.append({
                        "station":          sid,
                        "model":            display,
                        "scenario":         scenario,
                        "date":             d,
                        "pr_raw_mm":        float(raw)  if np.isfinite(raw)  else np.nan,
                        "pr_corrected_mm":  float(c)    if np.isfinite(c)    else np.nan,
                    })

    if not long_rows:
        log.error("No bias-corrected rows produced — abort."); return

    long_df = pd.DataFrame(long_rows)
    long_df["date"] = pd.to_datetime(long_df["date"])
    long_path = CFG.out_dir / "tables" / "bias_corrected_long.csv"
    long_df.to_csv(long_path, index=False, float_format="%.4f")
    log.info("Bias-corrected long table -> %s  (%d rows)", long_path, len(long_df))

    # Per (station × model × month) summary for the appendix
    summary_rows: List[Dict] = []
    for (sid, model), sub in long_df[long_df["scenario"] == "historical"].groupby(["station", "model"]):
        sub = sub.assign(month=sub["date"].dt.month)
        for m, ssub in sub.groupby("month"):
            obs_v = obs[sid][obs[sid].index.month == m].loc[CFG.calib_start: CFG.calib_end]
            summary_rows.append({
                "station":     sid,
                "model":       model,
                "month":       m,
                "obs_mean":            float(obs_v.mean()),
                "mod_raw_hist_mean":   float(ssub["pr_raw_mm"].mean()),
                "mod_corr_hist_mean":  float(ssub["pr_corrected_mm"].mean()),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = CFG.out_dir / "tables" / "eqm_summary.csv"
    summary_df.to_csv(summary_path, index=False, float_format="%.3f")
    log.info("EQM summary               -> %s", summary_path)

    # Diagnostic CDFs at one representative station per model
    if stations:
        rep_station = stations[0]
        plot_eqm_cdfs(obs, mod_hist, mod_corrected_hist, rep_station,
                      CFG.out_dir / "figures" / "eqm_cdfs.png")
        log.info("EQM CDF figure            -> figures/eqm_cdfs.png  (station: %s)",
                 rep_station)

    # SSP trajectory figure
    plot_ssp_trajectories(long_df, stations,
                          CFG.out_dir / "figures" / "ssp_trajectories.png")
    log.info("SSP trajectories figure   -> figures/ssp_trajectories.png")

    print("\n=== Bias correction summary ===")
    print(f"Rows produced: {len(long_df)}")
    print(f"Models       : {', '.join(CFG.retained_models)}")
    print(f"Scenarios    : {', '.join(s for s in CFG.scenarios if s in long_df['scenario'].unique())}")
    print(f"Stations     : {len(stations)}")
    print(f"Calibration  : {CFG.calib_start} .. {CFG.calib_end}")
    print(f"Projection   : {CFG.proj_start}  .. {CFG.proj_end}")


if __name__ == "__main__":
    main()
