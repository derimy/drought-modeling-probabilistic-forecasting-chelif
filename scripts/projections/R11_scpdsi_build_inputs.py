"""
Author        : Imene DERRAR
Supervisor    : Abdelillah OTMANE CHERIF
Institutions  : NHSM — National Higher School of Mathematics
                IHFR — Institut Hydrométéorologique de Formation et de Recherche
Academic year : 2025–2026
Project       : Final-year engineering thesis in Applied Mathematics —
                Drought modelling and probabilistic forecasting for the Chélif basin

--------------------------------------------------------------------------------

R11 — Build scPDSI projection input files
==================================================

Assemble the 6 (model × scenario) basin-mean monthly time series that
drive the scPDSI projection, formatted exactly like the historical
`terraclimate_basin_monthly.csv` and `era5land_basin_monthly.csv` files
that the scPDSI compute step (notebook E.7) already reads.

For each retained temperature model (ACCESS-CM2, CMCC-ESM2, GFDL-ESM4)
and each SSP scenario (ssp2_4_5, ssp5_8_5), write a single CSV with:

    date | ppt_mm | pet_mm | tmean_C | tmax_C | tmin_C | qc_flag

covering the continuous window 1990-01 .. 2040-12, where:

    - 1990–2014 = bias-corrected historical run for that model
    - 2015–2040 = bias-corrected SSP run for that (model × scenario)

The continuous concatenation is essential: scPDSI is a recursive water
balance model, so the soil-moisture state at end of 2014 must flow into
2015 without a discontinuity.

Sources per column
------------------
    ppt_mm   : EQM bias-correction of CMIP6 `pr` for these 3 models,
               calibrated against ONM basin-mean precipitation
               (`basin_precip_master_1990_2015.csv`) per calendar month.
    pet_mm   : R10 output (`pet_projection_long.csv`),
               column `pet_pm_equiv_mm` (HS aligned onto PM scale).
    tmean_C  : R9 output, variable `tas`.
    tmax_C   : R9 output, variable `tasmax`.
    tmin_C   : R9 output, variable `tasmin`.
    qc_flag  : 1 if all values finite, 0 otherwise.

Outputs (under 02_processed/scpdsi_inputs/projection/)
    access_cm2__ssp2_4_5.csv
    access_cm2__ssp5_8_5.csv
    cmcc_esm2__ssp2_4_5.csv
    cmcc_esm2__ssp5_8_5.csv
    gfdl_esm4__ssp2_4_5.csv
    gfdl_esm4__ssp5_8_5.csv

Plus a single combined long table for inspection:
    04_outputs/r11_inputs_scpdsi_inputs/tables/all_models_long.csv
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

    # Retained temperature models from R8 top-3
    retained_models: Dict[str, str] = field(default_factory=lambda: {
        "ACCESS-CM2": "access_cm2",
        "CMCC-ESM2":  "cmcc_esm2",
        "GFDL-ESM4":  "gfdl_esm4",
    })
    scenarios: Tuple[str, ...] = ("ssp2_4_5", "ssp5_8_5")

    project_root: Path = Path(__file__).resolve().parents[3]
    onm_basin_csv: Path = field(init=False)
    cmip6_root:    Path = field(init=False)
    pet_csv:       Path = field(init=False)
    temp_csv:      Path = field(init=False)
    out_scpdsi_dir: Path = field(init=False)
    out_diag_dir:   Path = field(init=False)

    def __post_init__(self) -> None:
        self.onm_basin_csv = (self.project_root / "01_data" / "external"
                              / "basin_precip_master_1990_2015.csv")
        self.cmip6_root    = self.project_root / "01_data" / "cmip6"
        self.pet_csv       = (self.project_root / "04_outputs"
                              / "r10_pet_scpdsi_pet_hargreaves"
                              / "tables" / "pet_projection_long.csv")
        self.temp_csv      = (self.project_root / "04_outputs"
                              / "r9_temp_bias_scpdsi_temperature_bias_correction"
                              / "tables" / "temperature_bias_corrected_long.csv")
        self.out_scpdsi_dir = (self.project_root / "02_processed"
                                / "scpdsi_inputs" / "projection")
        self.out_diag_dir   = (self.project_root / "04_outputs"
                                / "r11_inputs_scpdsi_inputs")
        self.out_scpdsi_dir.mkdir(parents=True, exist_ok=True)
        (self.out_diag_dir / "tables").mkdir(parents=True, exist_ok=True)
        (self.out_diag_dir / "figures").mkdir(parents=True, exist_ok=True)


CFG = Config()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("r11_inputs")


# --------------------------------------------------------------------------- #
# 1. UTILITIES
# --------------------------------------------------------------------------- #

def _normalize_dims(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    if "latitude"   in ds.coords: rename["latitude"]  = "lat"
    if "longitude"  in ds.coords: rename["longitude"] = "lon"
    if "valid_time" in ds.coords and "time" not in ds.coords:
        rename["valid_time"] = "time"
    return ds.rename(rename) if rename else ds


def _wrap_lon(ds: xr.Dataset) -> xr.Dataset:
    if "lon" not in ds.coords: return ds
    if float(ds["lon"].max()) > 180.0:
        ds = ds.assign_coords(lon=(((ds["lon"] + 180) % 360) - 180)).sortby("lon")
    return ds


def _clip_box(ds: xr.Dataset, nwse) -> xr.Dataset:
    n, w, s, e = nwse
    lat = ds["lat"]
    lat_slice = slice(n, s) if float(lat[0]) > float(lat[-1]) else slice(s, n)
    return ds.sel(lat=lat_slice, lon=slice(w, e))


def _to_mm_per_month(da: xr.DataArray) -> xr.DataArray:
    days = da["time"].dt.days_in_month
    return da * 86400.0 * days


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


def _basin_mean(da: xr.DataArray) -> pd.Series:
    weights = np.cos(np.deg2rad(da["lat"]))
    weights = weights.where(weights > 0, 0)
    s = da.weighted(weights).mean(dim=("lat", "lon")).to_pandas()
    return _force_dt_index(s)


# --------------------------------------------------------------------------- #
# 2. PRECIPITATION EQM (basin-mean)
# --------------------------------------------------------------------------- #

def eqm_precip_per_month(obs_calib: pd.Series, mod_calib: pd.Series,
                          mod_target: pd.Series,
                          calib_window: Tuple[str, str]) -> pd.Series:
    """EQM clipped at 0 (precipitation can't be negative).  Same as R3."""
    o_calib = obs_calib.loc[calib_window[0]: calib_window[1]]
    m_calib = mod_calib.loc[calib_window[0]: calib_window[1]]
    out = pd.Series(np.nan, index=mod_target.index, dtype=float)
    for month in range(1, 13):
        oc = o_calib[o_calib.index.month == month].dropna().values
        mc = m_calib[m_calib.index.month == month].dropna().values
        mask = mod_target.index.month == month
        mt = mod_target[mask].values
        if oc.size < 5 or mc.size < 5:
            continue
        obs_sorted = np.sort(oc)
        mod_sorted = np.sort(mc)
        n_obs = obs_sorted.size; n_mod = mod_sorted.size
        p_obs = (np.arange(n_obs) + 0.5) / n_obs
        p_mod = (np.arange(n_mod) + 0.5) / n_mod
        p_tgt = np.interp(mt, mod_sorted, p_mod,
                          left=p_mod[0], right=p_mod[-1])
        corr  = np.interp(p_tgt, p_obs, obs_sorted,
                          left=obs_sorted[0], right=obs_sorted[-1])
        out.loc[mask] = np.clip(corr, 0.0, None)
    return out


def load_model_precip_basin_mean(cfg: Config, model_folder: str,
                                  scenario: str) -> pd.Series | None:
    """Load `pr` NetCDFs, aggregate to basin-mean, convert to mm/month."""
    folder = cfg.cmip6_root / scenario / model_folder
    files = sorted(folder.glob("pr_*.nc"))
    if not files:
        log.warning("[%s/%s] no pr_*.nc", model_folder, scenario); return None
    ds = xr.open_mfdataset(files, combine="by_coords",
                            use_cftime=True, chunks={"time": 60})
    ds = _normalize_dims(ds); ds = _wrap_lon(ds)
    if "pr" not in ds.data_vars:
        log.warning("[%s/%s] 'pr' missing", model_folder, scenario); return None
    da = ds["pr"]
    da = _clip_box(da, cfg.bbox_nwse)
    if hasattr(da["time"].values[0], "calendar") or da["time"].dtype == object:
        da = da.convert_calendar("standard", align_on="date")
    da = _to_mm_per_month(da)
    s = _basin_mean(da)
    return s.astype(float)


# --------------------------------------------------------------------------- #
# 3. LOAD ONM BASIN PRECIP REFERENCE
# --------------------------------------------------------------------------- #

def load_onm_basin_precip(cfg: Config) -> pd.Series:
    """Load the basin-mean ONM precipitation that the historical scPDSI
    used (same source as SPI)."""
    df = pd.read_csv(cfg.onm_basin_csv, parse_dates=["date"]).set_index("date")
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    # Column name from notebook_02b_scpdsi_pipeline.py CFG.basin_precip_col
    col = "precip_basin_mm" if "precip_basin_mm" in df.columns else df.columns[0]
    s = df[col].astype(float).loc[cfg.calib_start: cfg.calib_end].dropna()
    log.info("ONM basin precip reference: n=%d  mean=%.1f mm/month",
             len(s), s.mean())
    return s


# --------------------------------------------------------------------------- #
# 4. MAIN
# --------------------------------------------------------------------------- #

def main() -> None:
    log.info("Project root:    %s", CFG.project_root)
    log.info("Retained models: %s", list(CFG.retained_models))
    log.info("Scenarios:       %s", CFG.scenarios)

    # ---- References ------------------------------------------------------- #
    obs_ppt = load_onm_basin_precip(CFG)

    # ---- R9 and R10 outputs -------------------------------------- #
    temp_df = pd.read_csv(CFG.temp_csv, parse_dates=["date"])
    pet_df  = pd.read_csv(CFG.pet_csv,  parse_dates=["date"])
    temp_df["date"] = pd.to_datetime(temp_df["date"]).dt.to_period("M").dt.to_timestamp()
    pet_df["date"]  = pd.to_datetime(pet_df["date"]).dt.to_period("M").dt.to_timestamp()

    # ---- Per (model × scenario) ------------------------------------------ #
    all_rows: List[Dict] = []

    for model_label, model_folder in CFG.retained_models.items():
        log.info("=== %s (%s) ===", model_label, model_folder)

        # Bias-correct precipitation for this model — historical first as calibration
        ppt_hist_raw = load_model_precip_basin_mean(CFG, model_folder, "historical")
        if ppt_hist_raw is None: continue
        ppt_hist_raw_calib = ppt_hist_raw.loc[CFG.calib_start: CFG.calib_end]

        # Apply EQM to historical (sanity)
        ppt_hist_corr = eqm_precip_per_month(obs_ppt, ppt_hist_raw_calib,
                                              ppt_hist_raw_calib,
                                              (CFG.calib_start, CFG.calib_end))
        log.info("[%s/historical] ppt mean: raw=%.1f  corrected=%.1f  obs=%.1f",
                 model_label, ppt_hist_raw_calib.mean(),
                 ppt_hist_corr.mean(), obs_ppt.mean())

        # Temperature and PET subsets for this model
        tmp_m = temp_df[temp_df["model"] == model_label]
        pet_m = pet_df[pet_df["model"] == model_label]

        for scen in CFG.scenarios:
            log.info("--- scenario: %s", scen)
            # 1) Projection P (load and bias-correct)
            ppt_ssp_raw = load_model_precip_basin_mean(CFG, model_folder, scen)
            if ppt_ssp_raw is None: continue
            ppt_ssp_raw = ppt_ssp_raw.loc[CFG.proj_start: CFG.proj_end]
            ppt_ssp_corr = eqm_precip_per_month(obs_ppt, ppt_hist_raw_calib,
                                                  ppt_ssp_raw,
                                                  (CFG.calib_start, CFG.calib_end))

            # 2) Continuous 1990–2040 series for every variable
            full_dates = pd.date_range(CFG.calib_start, CFG.proj_end, freq="MS")
            df_out = pd.DataFrame({"date": full_dates})
            df_out["date"] = pd.to_datetime(df_out["date"]).dt.to_period("M").dt.to_timestamp()
            df_out = df_out.set_index("date")

            # Precipitation (bias-corrected): hist + ssp
            ppt_full = pd.concat([
                ppt_hist_corr.loc[CFG.calib_start: CFG.calib_end],
                ppt_ssp_corr.loc[CFG.proj_start: CFG.proj_end],
            ])
            df_out["ppt_mm"] = ppt_full.reindex(df_out.index)

            # PET (from R10 — pet_pm_equiv_mm column)
            pet_full = pd.concat([
                (pet_m[pet_m["scenario"] == "historical"]
                 .set_index("date")["pet_pm_equiv_mm"]
                 .loc[CFG.calib_start: CFG.calib_end]),
                (pet_m[pet_m["scenario"] == scen]
                 .set_index("date")["pet_pm_equiv_mm"]
                 .loc[CFG.proj_start: CFG.proj_end]),
            ])
            df_out["pet_mm"] = pet_full.reindex(df_out.index)

            # Temperature (from R9 — corrected_C, variable tas/tasmax/tasmin)
            for var, col_out in (("tas", "tmean_C"),
                                  ("tasmax", "tmax_C"),
                                  ("tasmin", "tmin_C")):
                t_full = pd.concat([
                    (tmp_m[(tmp_m["scenario"] == "historical") & (tmp_m["variable"] == var)]
                     .set_index("date")["corrected_C"]
                     .loc[CFG.calib_start: CFG.calib_end]),
                    (tmp_m[(tmp_m["scenario"] == scen) & (tmp_m["variable"] == var)]
                     .set_index("date")["corrected_C"]
                     .loc[CFG.proj_start: CFG.proj_end]),
                ])
                df_out[col_out] = t_full.reindex(df_out.index)

            # QC flag
            req = ["ppt_mm", "pet_mm", "tmean_C", "tmax_C", "tmin_C"]
            df_out["qc_flag"] = df_out[req].notna().all(axis=1).astype(int)

            # Save the scPDSI-input CSV (column order matches the historical file)
            df_out = df_out.reset_index()
            out_csv = CFG.out_scpdsi_dir / f"{model_folder}__{scen}.csv"
            df_out[["date", "ppt_mm", "pet_mm", "tmean_C", "tmax_C", "tmin_C",
                    "qc_flag"]].to_csv(out_csv, index=False, float_format="%.4f")
            log.info("[%s/%s] scPDSI input CSV -> %s  (n_months=%d, qc_ok=%d)",
                     model_label, scen, out_csv,
                     len(df_out), int(df_out["qc_flag"].sum()))

            # Add to long table for the diagnostic
            for _, row in df_out.iterrows():
                all_rows.append({"model": model_label, "scenario": scen, **row.to_dict()})

    if not all_rows:
        log.error("No scPDSI input produced — check sources."); return

    long_df = pd.DataFrame(all_rows)
    long_df["date"] = pd.to_datetime(long_df["date"])
    long_path = CFG.out_diag_dir / "tables" / "all_models_long.csv"
    long_df.to_csv(long_path, index=False, float_format="%.4f")
    log.info("Combined long table -> %s  (%d rows)", long_path, len(long_df))

    # ---- Diagnostic figure: P, PET trajectories per (model × scenario) --- #
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for col, model in enumerate(CFG.retained_models):
        mod_sub = long_df[long_df["model"] == model]
        # Row 0 — P annual
        ax = axes[0, col]
        for scen, color in [("ssp2_4_5", "C0"), ("ssp5_8_5", "C3")]:
            s = (mod_sub[mod_sub["scenario"] == scen]
                 .set_index("date")["ppt_mm"]
                 .resample("YE").sum())
            ax.plot(s.index.year, s.values, color=color, lw=1.0, label=scen)
        ax.set_title(f"{model} — Annual P (mm/yr)", fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        if col == 0: ax.set_ylabel("P annual (mm/yr)")

        # Row 1 — PET annual
        ax = axes[1, col]
        for scen, color in [("ssp2_4_5", "C0"), ("ssp5_8_5", "C3")]:
            s = (mod_sub[mod_sub["scenario"] == scen]
                 .set_index("date")["pet_mm"]
                 .resample("YE").sum())
            ax.plot(s.index.year, s.values, color=color, lw=1.0, label=scen)
        ax.set_title(f"{model} — Annual PET (mm/yr)", fontsize=10)
        ax.set_xlabel("Year"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        if col == 0: ax.set_ylabel("PET annual (mm/yr)")

    fig.suptitle("scPDSI projection inputs — bias-corrected annual P and PET, "
                 "1990–2040 (continuous)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(CFG.out_diag_dir / "figures" / "p_pet_inputs.png", dpi=200)
    plt.close(fig)
    log.info("Figure -> %s", CFG.out_diag_dir / "figures" / "p_pet_inputs.png")

    # Console summary
    print("\n=== scPDSI input build summary ===")
    print(f"Output files (under {CFG.out_scpdsi_dir}):")
    for f in sorted(CFG.out_scpdsi_dir.glob("*.csv")):
        print(f"  {f.name}")
    print("\nSchema per CSV: date, ppt_mm, pet_mm, tmean_C, tmax_C, tmin_C, qc_flag")
    print("Range:           1990-01 .. 2040-12  (continuous)")
    print("Calibration:     1990-2014 (for EQM on P)")
    print("Projection:      2015-2040  (SSP run, bias-corrected)")
    print("\nNext step: run the R scPDSI compute script (or Python equivalent)")
    print("on each of the 6 output files, AWC=120 mm, calibrate scPDSI parameters")
    print("on 1990-2014.  Output: scPDSI series 1990-2040 per (model × scenario).")


if __name__ == "__main__":
    main()
