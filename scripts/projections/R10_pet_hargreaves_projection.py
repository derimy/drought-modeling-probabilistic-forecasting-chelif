"""
Author        : Imene DERRAR
Supervisor    : Abdelillah OTMANE CHERIF
Institutions  : NHSM — National Higher School of Mathematics
                IHFR — Institut Hydrométéorologique de Formation et de Recherche
Academic year : 2025–2026
Project       : Final-year engineering thesis in Applied Mathematics —
                Drought modelling and probabilistic forecasting for the Chélif basin

--------------------------------------------------------------------------------

R10 — Hargreaves-Samani PET projection and EQM correction against
TerraClimate Penman-Monteith historical
==========================================================================

For each retained CMIP6 model × scenario × month, compute monthly PET via
Hargreaves-Samani (FAO-56 form) from the bias-corrected tmean, tmax, tmin
produced by R9.  Then apply a second EQM correction that aligns
the HS-derived PET distribution onto the TerraClimate Penman-Monteith
PET distribution of the historical period — so the scPDSI projection in
R11 receives PET on the same scale as the historical scPDSI run
(which was driven by TerraClimate PM).

Why this two-stage strategy
---------------------------
For the projection we only have CMIP6 temperature (no wind, no solar
radiation, no humidity) — so Penman-Monteith is not computable directly.
Hargreaves-Samani is the standard temperature-only substitute, but it
produces systematically different absolute PET values from PM (typically
HS underestimates summer PM by 10–30 % in semi-arid Mediterranean climates).
Applying an EQM that maps HS_raw onto historical PM gives a projected PET
on the PM distribution — same scale as the scPDSI compute step (notebook E.7) used historically.

Hargreaves-Samani (FAO-56):
    PET_HS [mm/day] = 0.0023 · Ra(φ, doy) · (Tmean + 17.8) · √(Tmax − Tmin)

where Ra is extraterrestrial radiation (mm/day equivalent, computed at the
basin centroid latitude φ = 35.2° N and mid-month day of year).

Inputs
------
    04_outputs/r9_temp_bias_scpdsi_temperature_bias_correction/tables/temperature_bias_corrected_long.csv
    02_processed/scpdsi_inputs/terraclimate_basin_monthly.csv

Outputs (under 04_outputs/r10_pet_scpdsi_pet_hargreaves/)
    tables/pet_projection_long.csv
        Long format: model, scenario, date,
            pet_hs_raw_mm    (Hargreaves-Samani from bias-corrected T)
            pet_pm_equiv_mm  (after EQM against TerraClimate PM historical)
    tables/eqm_summary_pet.csv
        Per (model × calendar month) sanity check on the calibration period.
    figures/pet_eqm_cdfs.png
    figures/pet_trajectories.png
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 0. CONFIGURATION
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    calib_start: str = "1990-01"
    calib_end:   str = "2014-12"
    proj_start:  str = "2015-01"
    proj_end:    str = "2040-12"

    # Basin centroid latitude (same as the scPDSI historical pipeline; notebook E.7)
    lat_basin_deg: float = 35.2

    # Retained models from R8 top-3 (temperature Taylor)
    retained_models: Tuple[str, ...] = ("ACCESS-CM2", "CMCC-ESM2", "GFDL-ESM4")
    scenarios:       Tuple[str, ...] = ("historical", "ssp2_4_5", "ssp5_8_5")

    project_root: Path = Path(__file__).resolve().parents[3]
    bias_corr_csv: Path = field(init=False)
    terra_csv:     Path = field(init=False)
    out_dir:       Path = field(init=False)

    def __post_init__(self) -> None:
        self.bias_corr_csv = (self.project_root / "04_outputs"
                              / "r9_temp_bias_scpdsi_temperature_bias_correction"
                              / "tables" / "temperature_bias_corrected_long.csv")
        self.terra_csv     = (self.project_root / "02_processed" / "scpdsi_inputs"
                              / "terraclimate_basin_monthly.csv")
        self.out_dir       = (self.project_root / "04_outputs"
                              / "r10_pet_scpdsi_pet_hargreaves")
        (self.out_dir / "tables").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "figures").mkdir(parents=True, exist_ok=True)


CFG = Config()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("r10_pet")


# --------------------------------------------------------------------------- #
# 1. EXTRATERRESTRIAL RADIATION (FAO-56 Eq. 21)
# --------------------------------------------------------------------------- #

# Day-of-year at the middle of each calendar month (1..12) — Julian days.
_MID_MONTH_DOY = (15, 46, 75, 106, 136, 167, 197, 228, 259, 289, 320, 350)


def extraterrestrial_radiation_mm_per_day(lat_deg: float, doy: int) -> float:
    """Allen et al. 1998 FAO-56 formula.

    Returns Ra in mm/day equivalent (i.e. MJ/m²/day × 0.408), which is the
    form needed by the Hargreaves-Samani equation as we write it.
    """
    phi   = math.radians(lat_deg)
    Gsc   = 0.0820                                # MJ m⁻² min⁻¹  (solar constant)
    dr    = 1.0 + 0.033 * math.cos(2 * math.pi * doy / 365.0)
    delta = 0.409 * math.sin(2 * math.pi * doy / 365.0 - 1.39)
    # Sunset hour angle ωs ; clamp argument of arccos to [-1, 1] for tropics/poles
    arg = -math.tan(phi) * math.tan(delta)
    arg = max(-1.0, min(1.0, arg))
    omega_s = math.acos(arg)
    Ra_MJ = (24.0 * 60.0 / math.pi) * Gsc * dr * (
        omega_s * math.sin(phi) * math.sin(delta)
        + math.cos(phi) * math.cos(delta) * math.sin(omega_s)
    )
    return Ra_MJ * 0.408                          # MJ/m²/day → mm/day equivalent


def hargreaves_samani_monthly_mm(tmean_C: float, tmax_C: float, tmin_C: float,
                                   lat_deg: float, month: int,
                                   days_in_month: int) -> float:
    """PET in mm/month via Hargreaves-Samani."""
    dtr = max(tmax_C - tmin_C, 0.0)
    if not all(np.isfinite([tmean_C, tmax_C, tmin_C])):
        return np.nan
    Ra        = extraterrestrial_radiation_mm_per_day(lat_deg,
                                                       _MID_MONTH_DOY[month - 1])
    pet_daily = 0.0023 * Ra * (tmean_C + 17.8) * math.sqrt(dtr)
    return pet_daily * days_in_month


# --------------------------------------------------------------------------- #
# 2. EQM (with non-negativity clip for PET)
# --------------------------------------------------------------------------- #

def eqm(obs_calib: np.ndarray, mod_calib: np.ndarray,
         mod_target: np.ndarray) -> np.ndarray:
    """Empirical quantile mapping with linear interpolation, clipped at 0
    (PET cannot be negative).

    Delta-shift at the tails preserves any out-of-range projection values."""
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

    p_tgt     = np.interp(mod_target, mod_sorted, p_mod,
                          left=p_mod[0], right=p_mod[-1])
    corrected = np.interp(p_tgt, p_obs, obs_sorted,
                          left=obs_sorted[0], right=obs_sorted[-1])

    mod_target = np.asarray(mod_target, dtype=float)
    below = mod_target < mod_sorted[0]
    above = mod_target > mod_sorted[-1]
    corrected[below] = obs_sorted[0]  + (mod_target[below] - mod_sorted[0])
    corrected[above] = obs_sorted[-1] + (mod_target[above] - mod_sorted[-1])

    return np.clip(corrected, 0.0, None)


def correct_pet_monthly(pet_obs_pm: pd.Series, pet_hs_raw: pd.Series,
                          pet_hs_target: pd.Series,
                          calib_window: Tuple[str, str]) -> pd.Series:
    """EQM per calendar month: maps HS-derived PET onto Penman-Monteith
    historical distribution."""
    o_calib = pet_obs_pm.loc[calib_window[0]: calib_window[1]]
    m_calib = pet_hs_raw.loc[calib_window[0]: calib_window[1]]
    out = pd.Series(np.nan, index=pet_hs_target.index, dtype=float)
    for month in range(1, 13):
        oc = o_calib[o_calib.index.month == month].dropna().values
        mc = m_calib[m_calib.index.month == month].dropna().values
        mask = pet_hs_target.index.month == month
        mt = pet_hs_target[mask].values
        out.loc[mask] = eqm(oc, mc, mt)
    return out


# --------------------------------------------------------------------------- #
# 3. MAIN
# --------------------------------------------------------------------------- #

def main() -> None:
    log.info("Project root:        %s", CFG.project_root)
    log.info("Basin centroid lat:  %.2f°", CFG.lat_basin_deg)
    log.info("Retained models:     %s", CFG.retained_models)
    log.info("Calibration window:  %s .. %s", CFG.calib_start, CFG.calib_end)
    log.info("Projection window:   %s .. %s", CFG.proj_start,  CFG.proj_end)

    # ---- 1. Load bias-corrected temperature (R9 output) ---------- #
    bc = pd.read_csv(CFG.bias_corr_csv, parse_dates=["date"])
    bc["date"] = pd.to_datetime(bc["date"]).dt.to_period("M").dt.to_timestamp()
    log.info("Loaded bias-corrected T table: %d rows", len(bc))

    # ---- 2. Load TerraClimate PM PET reference ---------------------------- #
    terra = pd.read_csv(CFG.terra_csv, parse_dates=["date"]).set_index("date")
    terra.index = pd.to_datetime(terra.index).to_period("M").to_timestamp()
    pet_pm_obs = (terra["pet_mm"].astype(float)
                  .loc[CFG.calib_start: CFG.calib_end].dropna())
    log.info("TerraClimate PM PET reference: n=%d  mean=%.1f mm/month",
             len(pet_pm_obs), pet_pm_obs.mean())

    # ---- 3. Compute HS PET per (model × scenario × month), then EQM ------- #
    long_rows: List[Dict] = []

    for model in CFG.retained_models:
        log.info("=== %s ===", model)

        # Build (date × variable) wide table for this model, all scenarios
        wide = (bc[bc["model"] == model]
                .pivot_table(index=["scenario", "date"], columns="variable",
                             values="corrected_C")
                .reset_index())

        # Compute HS PET per row
        wide["days_in_month"] = wide["date"].dt.days_in_month
        wide["month"]         = wide["date"].dt.month
        wide["pet_hs_raw_mm"] = wide.apply(
            lambda r: hargreaves_samani_monthly_mm(
                r.get("tas"), r.get("tasmax"), r.get("tasmin"),
                CFG.lat_basin_deg, int(r["month"]), int(r["days_in_month"])),
            axis=1)

        # Calibrate EQM on (HS_hist 1990-2014, PM_obs 1990-2014)
        pet_hs_hist = (wide[wide["scenario"] == "historical"]
                       .set_index("date")["pet_hs_raw_mm"].sort_index())
        pet_hs_hist = pet_hs_hist.loc[CFG.calib_start: CFG.calib_end]

        # Apply EQM to every scenario's HS PET
        for scen in CFG.scenarios:
            sub = wide[wide["scenario"] == scen].sort_values("date")
            pet_target = sub.set_index("date")["pet_hs_raw_mm"]
            pet_eqm    = correct_pet_monthly(pet_pm_obs, pet_hs_hist,
                                              pet_target,
                                              (CFG.calib_start, CFG.calib_end))
            for d, raw, c in zip(pet_target.index, pet_target.values, pet_eqm.values):
                long_rows.append({
                    "model":          model,
                    "scenario":       scen,
                    "date":           d,
                    "pet_hs_raw_mm":  float(raw) if np.isfinite(raw) else np.nan,
                    "pet_pm_equiv_mm": float(c)  if np.isfinite(c)   else np.nan,
                })

    long_df = pd.DataFrame(long_rows)
    long_df["date"] = pd.to_datetime(long_df["date"])
    out_csv = CFG.out_dir / "tables" / "pet_projection_long.csv"
    long_df.to_csv(out_csv, index=False, float_format="%.4f")
    log.info("PET projection long table -> %s  (%d rows)", out_csv, len(long_df))

    # ---- 4. EQM sanity summary on calibration period ---------------------- #
    sub_hist = long_df[long_df["scenario"] == "historical"].copy()
    sub_hist["month"] = sub_hist["date"].dt.month
    sum_rows: List[Dict] = []
    for (model), grp_m in sub_hist.groupby("model"):
        for m, g in grp_m.groupby("month"):
            ref = pet_pm_obs[pet_pm_obs.index.month == m].loc[CFG.calib_start: CFG.calib_end]
            sum_rows.append({
                "model":              model,
                "month":              m,
                "pm_obs_mean_mm":     float(ref.mean()),
                "hs_raw_hist_mean_mm": float(g["pet_hs_raw_mm"].mean()),
                "pet_corr_hist_mean_mm": float(g["pet_pm_equiv_mm"].mean()),
                "residual_mm":        float(g["pet_pm_equiv_mm"].mean() - ref.mean()),
            })
    summary_df = pd.DataFrame(sum_rows)
    sum_path = CFG.out_dir / "tables" / "eqm_summary_pet.csv"
    summary_df.to_csv(sum_path, index=False, float_format="%.3f")
    log.info("PET EQM summary -> %s  (max |residual|=%.4f mm)",
             sum_path, summary_df["residual_mm"].abs().max())

    # ---- 5. Plots --------------------------------------------------------- #
    # CDF diagnostic for first model
    first = CFG.retained_models[0]
    fig, ax = plt.subplots(figsize=(7, 5))
    o = np.sort(pet_pm_obs.values)
    hs_raw = long_df[(long_df["model"] == first) & (long_df["scenario"] == "historical")]["pet_hs_raw_mm"]
    hs_cor = long_df[(long_df["model"] == first) & (long_df["scenario"] == "historical")]["pet_pm_equiv_mm"]
    r = np.sort(hs_raw.dropna().values)
    c = np.sort(hs_cor.dropna().values)
    ax.plot(o, np.linspace(0, 1, len(o)), "k",  lw=2.0, label="TerraClimate PM (reference)")
    ax.plot(r, np.linspace(0, 1, len(r)), "C3", lw=1.2, label="HS raw (from bias-corrected T)")
    ax.plot(c, np.linspace(0, 1, len(c)), "C0--", lw=1.2, label="HS bias-corrected to PM scale")
    ax.set_xlabel("Monthly PET (mm/month)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title(f"PET EQM diagnostic — {first} historical 1990–2014", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CFG.out_dir / "figures" / "pet_eqm_cdfs.png", dpi=200)
    plt.close(fig)

    # Annual-mean PET trajectories per model under each scenario
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    colors_scen = {"historical": "k", "ssp2_4_5": "C0", "ssp5_8_5": "C3"}
    for ax, model in zip(axes, CFG.retained_models):
        mod_sub = long_df[long_df["model"] == model]
        for scen, color in colors_scen.items():
            s = (mod_sub[mod_sub["scenario"] == scen]
                 .set_index("date")["pet_pm_equiv_mm"]
                 .resample("YE").sum())   # annual TOTAL PET
            if not s.empty:
                ax.plot(s.index.year, s.values, color=color, lw=1.1, label=scen)
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Year")
        ax.set_ylabel("Annual total PET (mm/year)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Annual total bias-corrected PET (PM-equivalent) per model × scenario",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(CFG.out_dir / "figures" / "pet_trajectories.png", dpi=200)
    plt.close(fig)
    log.info("Figures -> %s", CFG.out_dir / "figures")

    # Console summary
    print("\n=== PET projection summary ===")
    print(f"Rows produced:         {len(long_df)}")
    print(f"Models:                {', '.join(CFG.retained_models)}")
    print(f"Scenarios:             {', '.join(CFG.scenarios)}")
    print(f"Calibration window:    {CFG.calib_start} .. {CFG.calib_end}")
    print(f"Projection window:     {CFG.proj_start} .. {CFG.proj_end}")
    print(f"Max |residual| (hist): {summary_df['residual_mm'].abs().max():.4f} mm "
          "(should be ≈ 0 by EQM construction)")
    print(f"PM-equiv mean (hist 1990-2014): {pet_pm_obs.mean():.2f} mm/month  "
          "(reference)")
    print(f"HS-raw mean (hist 1990-2014):  "
          f"{long_df[long_df['scenario']=='historical']['pet_hs_raw_mm'].mean():.2f} mm/month")
    print(f"HS-corrected mean (hist):      "
          f"{long_df[long_df['scenario']=='historical']['pet_pm_equiv_mm'].mean():.2f} mm/month")


if __name__ == "__main__":
    main()
