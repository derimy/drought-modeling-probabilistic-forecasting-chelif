"""
Author        : Imene DERRAR
Supervisor    : Dr. Abdelillah OTMANE CHERIF
Institutions  : NHSM — National Higher School of Mathematics
                IHFR — Institut Hydrométéorologique de Formation et de Recherche
Academic year : 2025–2026
Project       : Final-year engineering thesis in Applied Mathematics —
                Drought modelling and probabilistic forecasting for the Chélif basin

--------------------------------------------------------------------------------

R9 — Temperature bias correction (EQM) for scPDSI projection
=====================================================================

For each retained CMIP6 model × each temperature variable (tas, tasmax,
tasmin) × each scenario (historical + ssp2_4_5 + ssp5_8_5), apply
empirical quantile mapping per calendar month, calibrated on the
1990–2014 overlap of the model historical run and the basin-mean
observational reference.

References (same as the R8 Taylor diagram):
    tas    → ERA5-Land    .tmean_C   (basin-mean, monthly)
    tasmax → TerraClimate .tmax_C    (basin-mean, monthly)
    tasmin → TerraClimate .tmin_C    (basin-mean, monthly)

EQM with delta-shift extrapolation at the tails
-----------------------------------------------
For values inside the calibration range: standard rank-based EQM.
For values BELOW the calibration min: bias_corrected = obs_min + (x − mod_min)
For values ABOVE the calibration max: bias_corrected = obs_max + (x − mod_max)

The delta-shift treatment preserves the warming signal at the tail of the
projection — clipping the corrected output at obs boundaries (the approach
used for precipitation in R3) would suppress projected extreme
temperatures and may underestimate projected warming extremes.
This is the standard ISIMIP3BASD / Cannon-2015 trend-preserving treatment.

Inputs
------
    02_processed/scpdsi_inputs/era5land_basin_monthly.csv
    02_processed/scpdsi_inputs/terraclimate_basin_monthly.csv
    01_data/cmip6/{historical|ssp2_4_5|ssp5_8_5}/<model>/<var>/*.nc

Outputs (under 04_outputs/r9_temp_bias_scpdsi_temperature_bias_correction/)
    tables/temperature_bias_corrected_long.csv
        Long format: model, scenario, variable, date, raw_C, corrected_C
    tables/eqm_summary_temperature.csv
        Per (model × variable × calendar month): obs_mean, raw_hist_mean,
        corrected_hist_mean, residual.  Should be ≈ 0 on the calibration period.
    figures/eqm_cdfs_temperature.png
        Diagnostic CDFs (obs / raw / corrected) for one model × variable.
    figures/temperature_trajectories.png
        Annual-mean tas trajectories under both SSPs per model — sanity check
        that the projected warming signal is preserved by the bias correction.
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

    # Retained models from the R8 top-3
    retained_models: Dict[str, str] = field(default_factory=lambda: {
        "ACCESS-CM2": "access_cm2",
        "CMCC-ESM2":  "cmcc_esm2",
        "GFDL-ESM4":  "gfdl_esm4",
    })

    # var_short -> (reference source, reference column)
    variables: Dict[str, Tuple[str, str]] = field(default_factory=lambda: {
        "tas":    ("ERA5-Land",    "tmean_C"),
        "tasmax": ("TerraClimate", "tmax_C"),
        "tasmin": ("TerraClimate", "tmin_C"),
    })

    scenarios: Tuple[str, ...] = ("historical", "ssp2_4_5", "ssp5_8_5")

    project_root: Path = Path(__file__).resolve().parents[3]
    era5_csv:     Path = field(init=False)
    terra_csv:    Path = field(init=False)
    cmip6_root:   Path = field(init=False)
    out_dir:      Path = field(init=False)

    def __post_init__(self) -> None:
        self.era5_csv  = (self.project_root / "02_processed" / "scpdsi_inputs"
                          / "era5land_basin_monthly.csv")
        self.terra_csv = (self.project_root / "02_processed" / "scpdsi_inputs"
                          / "terraclimate_basin_monthly.csv")
        self.cmip6_root = self.project_root / "01_data" / "cmip6"
        self.out_dir   = (self.project_root / "04_outputs"
                          / "r9_temp_bias_scpdsi_temperature_bias_correction")
        (self.out_dir / "tables").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "figures").mkdir(parents=True, exist_ok=True)


CFG = Config()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("r9_temp_bias")


# --------------------------------------------------------------------------- #
# 1. UTILITIES
# --------------------------------------------------------------------------- #

def _normalize_dims(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    if "latitude" in ds.coords:   rename["latitude"]  = "lat"
    if "longitude" in ds.coords:  rename["longitude"] = "lon"
    if "valid_time" in ds.coords and "time" not in ds.coords:
        rename["valid_time"] = "time"
    return ds.rename(rename) if rename else ds


def _wrap_lon(ds: xr.Dataset) -> xr.Dataset:
    if "lon" not in ds.coords:
        return ds
    if float(ds["lon"].max()) > 180.0:
        new_lon = (((ds["lon"] + 180.0) % 360.0) - 180.0)
        ds = ds.assign_coords(lon=new_lon).sortby("lon")
    return ds


def _clip_box(ds: xr.Dataset, nwse) -> xr.Dataset:
    n, w, s, e = nwse
    lat = ds["lat"]
    lat_slice = slice(n, s) if float(lat[0]) > float(lat[-1]) else slice(s, n)
    return ds.sel(lat=lat_slice, lon=slice(w, e))


def _basin_mean(da: xr.DataArray) -> pd.Series:
    weights = np.cos(np.deg2rad(da["lat"]))
    weights = weights.where(weights > 0, 0)
    s = da.weighted(weights).mean(dim=("lat", "lon")).to_pandas()
    return _force_dt_index(s)


def _force_dt_index(s: pd.Series) -> pd.Series:
    if not isinstance(s.index, pd.DatetimeIndex):
        yms = []
        for t in s.index:
            try:
                yms.append((int(t.year), int(t.month)))
            except AttributeError:
                ts = pd.Timestamp(t)
                yms.append((ts.year, ts.month))
        s.index = pd.to_datetime([f"{y:04d}-{m:02d}-01" for y, m in yms])
    s.index = s.index.to_period("M").to_timestamp()
    return s


def load_reference_basin_mean(cfg: Config) -> Dict[str, pd.Series]:
    era5 = pd.read_csv(cfg.era5_csv,  parse_dates=["date"]).set_index("date")
    terra = pd.read_csv(cfg.terra_csv, parse_dates=["date"]).set_index("date")
    era5.index  = pd.to_datetime(era5.index).to_period("M").to_timestamp()
    terra.index = pd.to_datetime(terra.index).to_period("M").to_timestamp()
    out: Dict[str, pd.Series] = {}
    for var, (src, col) in cfg.variables.items():
        df = era5 if src == "ERA5-Land" else terra
        s = df[col].astype(float).loc[cfg.calib_start: cfg.calib_end].dropna()
        out[var] = s
        log.info("[ref] %-7s -> %-13s.%-8s  n=%d  mean=%.2f °C",
                 var, src, col, len(s), s.mean())
    return out


_CMIP6_NCVAR = {"tas": "tas", "tasmax": "tasmax", "tasmin": "tasmin"}


def load_model_basin_mean(cfg: Config, model_folder: str, scenario: str,
                           var_short: str) -> pd.Series | None:
    folder = cfg.cmip6_root / scenario / model_folder / var_short
    files = sorted(folder.glob(f"{_CMIP6_NCVAR[var_short]}_*.nc"))
    if not files:
        log.warning("[%s/%s/%s] no NetCDF under %s",
                    model_folder, scenario, var_short, folder)
        return None
    ds = xr.open_mfdataset(files, combine="by_coords",
                           use_cftime=True, chunks={"time": 60})
    ds = _normalize_dims(ds)
    ds = _wrap_lon(ds)
    if _CMIP6_NCVAR[var_short] not in ds.data_vars:
        return None
    da = ds[_CMIP6_NCVAR[var_short]]
    da = _clip_box(da, cfg.bbox_nwse)
    if hasattr(da["time"].values[0], "calendar") or da["time"].dtype == object:
        da = da.convert_calendar("standard", align_on="date")
    s = _basin_mean(da)
    if s.mean() > 100:
        s = s - 273.15           # K -> °C
    return s.astype(float)


# --------------------------------------------------------------------------- #
# 2. EMPIRICAL QUANTILE MAPPING (with delta-shift extrapolation at tails)
# --------------------------------------------------------------------------- #

def eqm_temperature(obs_calib: np.ndarray, mod_calib: np.ndarray,
                     mod_target: np.ndarray) -> np.ndarray:
    """EQM with trend-preserving delta-shift extrapolation."""
    obs = obs_calib[np.isfinite(obs_calib)]
    mod = mod_calib[np.isfinite(mod_calib)]
    if obs.size < 5 or mod.size < 5:
        return np.full_like(mod_target, np.nan, dtype=float)

    obs_sorted = np.sort(obs)
    mod_sorted = np.sort(mod)
    n_obs = obs_sorted.size
    n_mod = mod_sorted.size
    p_obs = (np.arange(n_obs) + 0.5) / n_obs
    p_mod = (np.arange(n_mod) + 0.5) / n_mod

    # Standard EQM inside calibration range (np.interp default-clips at ends)
    p_tgt     = np.interp(mod_target, mod_sorted, p_mod,
                          left=p_mod[0], right=p_mod[-1])
    corrected = np.interp(p_tgt, p_obs, obs_sorted,
                          left=obs_sorted[0], right=obs_sorted[-1])

    # Delta-shift extrapolation at the tails (trend-preserving)
    mod_target = np.asarray(mod_target, dtype=float)
    below = mod_target < mod_sorted[0]
    above = mod_target > mod_sorted[-1]
    corrected[below] = obs_sorted[0]  + (mod_target[below] - mod_sorted[0])
    corrected[above] = obs_sorted[-1] + (mod_target[above] - mod_sorted[-1])

    return corrected


def correct_monthly_temperature(obs_calib: pd.Series, mod_calib: pd.Series,
                                  mod_target: pd.Series,
                                  calib_window: Tuple[str, str]) -> pd.Series:
    o_calib = obs_calib.loc[calib_window[0]: calib_window[1]]
    m_calib = mod_calib.loc[calib_window[0]: calib_window[1]]
    out = pd.Series(np.nan, index=mod_target.index, dtype=float)
    for month in range(1, 13):
        oc = o_calib[o_calib.index.month == month].dropna().values
        mc = m_calib[m_calib.index.month == month].dropna().values
        mask = mod_target.index.month == month
        mt = mod_target[mask].values
        out.loc[mask] = eqm_temperature(oc, mc, mt)
    return out


# --------------------------------------------------------------------------- #
# 3. PLOTS
# --------------------------------------------------------------------------- #

def plot_eqm_cdfs(obs: Dict[str, pd.Series], mod_hist_raw: Dict[str, Dict[str, pd.Series]],
                   mod_hist_cor: Dict[str, Dict[str, pd.Series]],
                   cfg: Config, out_path: Path) -> None:
    """CDF panels: one per variable, for the first retained model."""
    model = list(mod_hist_raw)[0]
    fig, axes = plt.subplots(1, len(cfg.variables),
                             figsize=(4.5 * len(cfg.variables), 4))
    for ax, var in zip(axes, cfg.variables):
        o = np.sort(obs[var].dropna().values)
        r = np.sort(mod_hist_raw[model][var].dropna().values)
        c = np.sort(mod_hist_cor[model][var].dropna().values)
        ax.plot(o, np.linspace(0, 1, len(o)), "k",  lw=2.0, label="reference")
        ax.plot(r, np.linspace(0, 1, len(r)), "C3", lw=1.2, label="model raw")
        ax.plot(c, np.linspace(0, 1, len(c)), "C0--", lw=1.2, label="model bias-corrected")
        ax.set_xlabel(f"{var} (°C)")
        ax.set_title(var, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Cumulative probability")
    fig.suptitle(f"EQM diagnostic CDFs at basin scale — {model} historical 1990–2014",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_temperature_trajectories(corrected: pd.DataFrame, cfg: Config,
                                    out_path: Path) -> None:
    sub = corrected[corrected["variable"] == "tas"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    colors_scen = {"historical": "k", "ssp2_4_5": "C0", "ssp5_8_5": "C3"}
    for ax, model in zip(axes, cfg.retained_models):
        mod_sub = sub[sub["model"] == model]
        for scen, color in colors_scen.items():
            s = (mod_sub[mod_sub["scenario"] == scen]
                 .set_index("date")["corrected_C"]
                 .resample("YE").mean())
            if not s.empty:
                ax.plot(s.index.year, s.values, color=color, lw=1.1, label=scen)
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Year")
        ax.set_ylabel("Annual-mean bias-corrected tas (°C)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Annual-mean bias-corrected tas under each SSP, "
                 "with delta-shift extrapolation at the tails",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. MAIN
# --------------------------------------------------------------------------- #

def main() -> None:
    log.info("Project root: %s", CFG.project_root)
    log.info("Models:       %s", list(CFG.retained_models))
    log.info("Variables:    %s", list(CFG.variables))
    log.info("Scenarios:    %s", CFG.scenarios)

    obs = load_reference_basin_mean(CFG)

    long_rows: List[Dict] = []
    mod_hist_raw:  Dict[str, Dict[str, pd.Series]] = {}
    mod_hist_cor:  Dict[str, Dict[str, pd.Series]] = {}

    for model_label, model_folder in CFG.retained_models.items():
        log.info("=== %s (%s) ===", model_label, model_folder)
        mod_hist_raw[model_label] = {}
        mod_hist_cor[model_label] = {}

        for var in CFG.variables:
            # 1) Historical raw
            s_hist = load_model_basin_mean(CFG, model_folder, "historical", var)
            if s_hist is None:
                log.error("[%s/%s] historical missing", model_label, var); continue
            s_hist_calib = s_hist.loc[CFG.calib_start: CFG.calib_end]
            mod_hist_raw[model_label][var] = s_hist_calib

            # 2) Calibrate EQM on historical 1990-2014, apply to historical (sanity)
            corr_hist = correct_monthly_temperature(obs[var], s_hist_calib,
                                                     s_hist_calib,
                                                     (CFG.calib_start, CFG.calib_end))
            mod_hist_cor[model_label][var] = corr_hist
            for d, raw, c in zip(s_hist_calib.index, s_hist_calib.values, corr_hist.values):
                long_rows.append({
                    "model":      model_label,
                    "scenario":   "historical",
                    "variable":   var,
                    "date":       d,
                    "raw_C":      float(raw) if np.isfinite(raw) else np.nan,
                    "corrected_C": float(c)   if np.isfinite(c)   else np.nan,
                })

            # 3) Apply EQM to each SSP run
            for scen in ("ssp2_4_5", "ssp5_8_5"):
                s_proj = load_model_basin_mean(CFG, model_folder, scen, var)
                if s_proj is None:
                    log.warning("[%s/%s/%s] missing", model_label, scen, var); continue
                s_proj = s_proj.loc[CFG.proj_start: CFG.proj_end]
                corr_proj = correct_monthly_temperature(obs[var], s_hist_calib,
                                                         s_proj,
                                                         (CFG.calib_start, CFG.calib_end))
                for d, raw, c in zip(s_proj.index, s_proj.values, corr_proj.values):
                    long_rows.append({
                        "model":      model_label,
                        "scenario":   scen,
                        "variable":   var,
                        "date":       d,
                        "raw_C":      float(raw) if np.isfinite(raw) else np.nan,
                        "corrected_C": float(c)   if np.isfinite(c)   else np.nan,
                    })

    long_df = pd.DataFrame(long_rows)
    long_df["date"] = pd.to_datetime(long_df["date"])
    long_path = CFG.out_dir / "tables" / "temperature_bias_corrected_long.csv"
    long_df.to_csv(long_path, index=False, float_format="%.4f")
    log.info("Long table -> %s  (%d rows)", long_path, len(long_df))

    # Sanity summary per (model × var × calendar month)
    sub_hist = long_df[long_df["scenario"] == "historical"].copy()
    sub_hist["month"] = sub_hist["date"].dt.month
    summary_rows: List[Dict] = []
    for (model, var), grp in sub_hist.groupby(["model", "variable"]):
        for m, g in grp.groupby("month"):
            o = obs[var][obs[var].index.month == m].loc[CFG.calib_start: CFG.calib_end]
            summary_rows.append({
                "model":              model,
                "variable":           var,
                "month":              m,
                "obs_mean_C":         float(o.mean()),
                "raw_hist_mean_C":    float(g["raw_C"].mean()),
                "corr_hist_mean_C":   float(g["corrected_C"].mean()),
                "residual_C":         float(g["corrected_C"].mean() - o.mean()),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = CFG.out_dir / "tables" / "eqm_summary_temperature.csv"
    summary_df.to_csv(summary_path, index=False, float_format="%.4f")
    log.info("EQM summary -> %s  (max |residual|=%.4f °C)",
             summary_path, summary_df["residual_C"].abs().max())

    # Figures
    plot_eqm_cdfs(obs, mod_hist_raw, mod_hist_cor, CFG,
                   CFG.out_dir / "figures" / "eqm_cdfs_temperature.png")
    plot_temperature_trajectories(long_df, CFG,
                                    CFG.out_dir / "figures" / "temperature_trajectories.png")
    log.info("Figures -> %s", CFG.out_dir / "figures")

    # Console summary
    print("\n=== Temperature bias-correction summary ===")
    print(f"Rows produced:         {len(long_df)}")
    print(f"Models  :              {', '.join(CFG.retained_models)}")
    print(f"Variables:             {', '.join(CFG.variables)}")
    print(f"Scenarios:             {', '.join(CFG.scenarios)}")
    print(f"Calibration window:    {CFG.calib_start} .. {CFG.calib_end}")
    print(f"Projection window:     {CFG.proj_start} .. {CFG.proj_end}")
    print(f"Max |residual| (hist): {summary_df['residual_C'].abs().max():.4f} °C "
          "(should be near 0 by EQM construction)")


if __name__ == "__main__":
    main()
