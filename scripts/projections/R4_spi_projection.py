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

R4 — Projected SPI from the bias-corrected CMIP6 ensemble
==================================================================

Compute SPI-1, SPI-3, SPI-6 and SPI-12 over 2015–2040 for each station, each
retained model, and each SSP scenario, using the **bias-corrected monthly
precipitation** produced by R3.  The SPI distribution parameters
are calibrated on the observed ONM record 1990–2014, exactly as in the
historical SPI chapter, so the projected SPI is on the same scale as the
SPI used elsewhere in the thesis.

Methodology (McKee et al. 1993; Edwards & McKee 1997)
-----------------------------------------------------
For each station and each accumulation window k ∈ {1, 3, 6, 12}:

  (1)  Build rolling k-month precipitation totals X_k.
  (2)  *Calibrate* on the ONM observation 1990–2014 X_k(obs):
       fit a 2-parameter gamma distribution to the positive values for each
       calendar month m, and record q_zero(m) = empirical probability mass
       at exactly zero precipitation.
  (3)  Define the mixed CDF:
              F_total(x | m) = q_zero(m)              if x = 0
                             = q_zero(m) + (1 − q_zero(m)) · F_gamma(x | m)  otherwise.
  (4)  *Apply* to the bias-corrected projections X_k(model, scenario):
              SPI(x | m) = Φ⁻¹(F_total(x | m)),  Φ⁻¹ the standard-normal quantile.
  (5)  Persist SPI to long-format CSV and produce diagnostic figures.

Why calibrate gamma on observations, not on the bias-corrected model
--------------------------------------------------------------------
Two reasons.  (a) Methodological continuity: the historical SPI computed in
notebook E.4 uses the same gamma calibration on ONM 1990–2014; using a
different calibration here would create a discontinuity at the 2014/2015
boundary that has no physical meaning.  (b) The EQM in R3 already
maps the model's empirical distribution onto the obs empirical distribution
on the calibration window, so re-calibrating gamma on the bias-corrected
model would be redundant — the resulting gamma parameters would be expected to remain very similar to the same
values as the obs-based fit, modulo numerical noise.

Inputs
------
    01_data/external/spi_ready_dataset_main_1990_2015.csv
        ONM monthly precipitation per station (gamma calibration source).

    04_outputs/R3_bias_correction/tables/bias_corrected_long.csv
        Bias-corrected monthly precipitation per (station × model × scenario).

Outputs (under 04_outputs/r4_spi_projection/)
    tables/spi_projection_long.csv      Long format: station, model, scenario,
                                        accumulation_k, date, spi_value.
    tables/gamma_calibration.csv        Per (station × k × month): gamma shape,
                                        scale, q_zero on the ONM record.
    tables/drought_class_frequencies.csv  Per (station × model × scenario × k):
                                        fraction of months in each McKee
                                        drought class over 2015–2040.
    figures/spi_trajectories_<k>.png    9 panels per k = {3, 6, 12}, showing
                                        SPI trajectories for all model × scenario
                                        combinations.
    figures/drought_class_heatmap.png   Stacked station × scenario heatmap of
                                        severe-drought (SPI ≤ −1.5) probability.
    figures/spi_distributions.png       Comparison of SPI distributions
                                        (1990–2014 obs vs 2015–2040 projection).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm

# --------------------------------------------------------------------------- #
# 0. CONFIGURATION
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    calib_start: str = "1990-01"
    calib_end:   str = "2014-12"
    proj_start:  str = "2015-01"
    proj_end:    str = "2040-12"

    # SPI accumulation windows in months
    accumulation_windows: Tuple[int, ...] = (1, 3, 6, 12)

    # McKee drought-class thresholds (SPI cut-offs).  The McKee bands are:
    #   ≥ +2.0   extremely wet
    # +1.5 ..    severely wet
    # +1.0 ..    moderately wet
    # −0.99..    near normal
    # −1.0 ..    moderately dry
    # −1.5 ..    severely dry
    # ≤ −2.0    extremely dry
    drought_class_edges: Tuple[float, ...] = (-2.0, -1.5, -1.0, 1.0, 1.5, 2.0)
    drought_class_labels: Tuple[str, ...]  = (
        "extremely_dry", "severely_dry", "moderately_dry",
        "near_normal", "moderately_wet", "severely_wet", "extremely_wet",
    )

    # Numerical guards on the CDF before transforming via Φ⁻¹
    cdf_floor: float = 1e-6
    cdf_ceil:  float = 1.0 - 1e-6

    # Minimum number of positive monthly samples needed to fit gamma robustly
    min_positive_samples: int = 5

    project_root: Path = Path(__file__).resolve().parents[3]
    onm_file:     Path = field(init=False)
    bc_file:      Path = field(init=False)
    out_dir:      Path = field(init=False)

    def __post_init__(self) -> None:
        self.onm_file = self.project_root / "01_data" / "external" / "spi_ready_dataset_main_1990_2015.csv"
        self.bc_file  = self.project_root / "04_outputs" / "R3_bias_correction" / "tables" / "bias_corrected_long.csv"
        self.out_dir  = self.project_root / "04_outputs" / "r4_spi_projection"
        (self.out_dir / "tables").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "figures").mkdir(parents=True, exist_ok=True)


CFG = Config()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("r4_spi_projection")


# --------------------------------------------------------------------------- #
# 1. SPI CORE
# --------------------------------------------------------------------------- #

def fit_gamma_per_month(series_calib: pd.Series, cfg: Config
                        ) -> Dict[int, Tuple[float, float, float]]:
    """Fit a 2-parameter gamma distribution per calendar month on the calibration window.

    Returns a dict:  month -> (shape, scale, q_zero)
    where q_zero is the empirical probability of exactly zero precipitation.
    """
    params: Dict[int, Tuple[float, float, float]] = {}
    for m in range(1, 13):
        x = series_calib[series_calib.index.month == m].dropna().values
        if x.size == 0:
            params[m] = (np.nan, np.nan, np.nan)
            continue
        x_pos  = x[x > 0]
        n_zero = int((x == 0).sum())
        q_zero = n_zero / x.size
        if x_pos.size < cfg.min_positive_samples:
            params[m] = (np.nan, np.nan, q_zero)
            continue
        # 2-parameter gamma: location forced to 0
        shape, _, scale = gamma_dist.fit(x_pos, floc=0)
        params[m] = (float(shape), float(scale), float(q_zero))
    return params


def transform_to_spi(series: pd.Series,
                     params_per_month: Dict[int, Tuple[float, float, float]],
                     cfg: Config) -> pd.Series:
    """Convert precipitation to SPI using the calibrated mixed-zero gamma."""
    out = pd.Series(np.nan, index=series.index, dtype=float)
    for m in range(1, 13):
        shape, scale, q_zero = params_per_month.get(m, (np.nan, np.nan, np.nan))
        if not np.isfinite(shape) or not np.isfinite(scale):
            continue
        mask = series.index.month == m
        x = series[mask].values
        # Mixed CDF: zero-mass + continuous gamma
        with np.errstate(invalid="ignore"):
            F = np.where(
                x <= 0,
                q_zero / 2.0,                          # mid-point on zero atom
                q_zero + (1.0 - q_zero) * gamma_dist.cdf(x, shape, scale=scale),
            )
        F = np.clip(F, cfg.cdf_floor, cfg.cdf_ceil)
        out.loc[mask] = norm.ppf(F)
    return out


def classify_drought(spi: float, cfg: Config) -> str:
    """Map an SPI value to a McKee drought class label."""
    if not np.isfinite(spi):
        return "missing"
    edges  = cfg.drought_class_edges
    labels = cfg.drought_class_labels
    for edge, label in zip(edges, labels):
        if spi < edge:
            return label
    return labels[-1]


# --------------------------------------------------------------------------- #
# 2. DATA LOADING
# --------------------------------------------------------------------------- #

def load_obs(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.onm_file, parse_dates=["date"])
    wide = (df.pivot_table(index="date", columns="master_station_id",
                           values="precip_mm", aggfunc="mean")
              .sort_index())
    wide.index = pd.to_datetime(wide.index).to_period("M").to_timestamp()
    return wide.loc[cfg.calib_start: cfg.calib_end]


def load_bias_corrected(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.bc_file, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    return df


# --------------------------------------------------------------------------- #
# 3. PLOTS
# --------------------------------------------------------------------------- #

def plot_spi_trajectories(spi_long: pd.DataFrame, k: int, stations: List[str],
                           out_path: Path) -> None:
    """3 × 3 station panels of SPI(k) under each (model × scenario) over 2015-2040."""
    n_cols = 3
    n_rows = int(np.ceil(len(stations) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.7 * n_cols, 2.7 * n_rows),
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

    sub = spi_long[(spi_long["accumulation_k"] == k)
                   & (spi_long["scenario"] != "historical")]

    for i, sid in enumerate(stations):
        ax = axes[i]
        for (model, scenario), grp in (sub[sub["station"] == sid]
                                       .groupby(["model", "scenario"])):
            color, linestyle = palette.get((model, scenario), ("k", "-"))
            ax.plot(grp["date"].values, grp["spi"].values,
                    color=color, linestyle=linestyle, lw=0.8, alpha=0.8,
                    label=f"{model} / {scenario}" if i == 0 else None)
        ax.axhline(0,    color="0.2", lw=0.5)
        ax.axhline(-1.5, color="firebrick", lw=0.5, linestyle=":")
        ax.axhline(-2.0, color="firebrick", lw=0.5, linestyle=":")
        ax.set_title(sid, fontsize=9)
        ax.set_ylim(-3.5, 3.5)
        ax.tick_params(labelsize=7)
    for k_idx in range(len(stations), len(axes)):
        axes[k_idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               frameon=False)
    fig.suptitle(f"SPI-{k} projection 2015-2040 (red dotted: severely-dry / extremely-dry)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_drought_class_heatmap(class_freq: pd.DataFrame, out_path: Path) -> None:
    """Per-station heatmap of severe drought probability (SPI <= -1.5) under each scenario.

    Aggregates across the three retained models (taking the mean) for a single
    headline figure; per-model breakdown is in the underlying CSV.
    """
    severe = (class_freq[class_freq["accumulation_k"] == 12]
              .copy())
    severe["severe_dry_freq"] = (
        severe["extremely_dry"] + severe["severely_dry"]
    )
    pivot = (severe
             .groupby(["station", "scenario"])["severe_dry_freq"]
             .mean()
             .unstack("scenario"))

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.values * 100, aspect="auto", cmap="OrRd",
                   vmin=0, vmax=40)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=0, fontsize=9)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center",
                        color=("white" if v > 0.20 else "black"), fontsize=8)
    fig.colorbar(im, ax=ax, label="Frequency (%)")
    ax.set_title("Severely / extremely dry months on SPI-12, 2015–2040\n"
                 "(ensemble mean of the 3 retained CMIP6 models)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_spi_distributions(spi_long: pd.DataFrame, k: int,
                            stations: List[str], out_path: Path) -> None:
    """Density of SPI(k) over the projection period 2015-2040, per station, per scenario."""
    n_cols = 3
    n_rows = int(np.ceil(len(stations) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.7 * n_cols, 2.7 * n_rows),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    bins = np.linspace(-3.5, 3.5, 30)
    for i, sid in enumerate(stations):
        ax = axes[i]
        sub = spi_long[(spi_long["accumulation_k"] == k)
                       & (spi_long["station"] == sid)]
        for scenario, color in [("ssp2_4_5", "C0"), ("ssp5_8_5", "C3")]:
            vals = sub.loc[sub["scenario"] == scenario, "spi"].dropna().values
            if vals.size > 0:
                ax.hist(vals, bins=bins, alpha=0.45, color=color,
                        density=True, label=scenario if i == 0 else None)
        ax.axvline(0, color="0.2", lw=0.5)
        ax.axvline(-1.5, color="firebrick", lw=0.5, linestyle=":")
        ax.set_title(sid, fontsize=9)
        ax.tick_params(labelsize=7)
    for k_idx in range(len(stations), len(axes)):
        axes[k_idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9,
               frameon=False)
    fig.suptitle(f"SPI-{k} distribution under each SSP, 2015-2040", fontsize=11)
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
    log.info("Accumulation k:   %s", list(CFG.accumulation_windows))

    obs = load_obs(CFG)
    stations = list(obs.columns)
    log.info("Stations:         %d", len(stations))

    bc = load_bias_corrected(CFG)
    log.info("Bias-corrected rows loaded: %d", len(bc))

    long_rows: List[Dict] = []
    gamma_rows: List[Dict] = []

    for sid in stations:
        obs_s = obs[sid]
        for k in CFG.accumulation_windows:
            obs_acc = obs_s.rolling(k).sum()
            params = fit_gamma_per_month(obs_acc, CFG)
            for m, (shape, scale, q_zero) in params.items():
                gamma_rows.append({
                    "station":          sid,
                    "accumulation_k":   k,
                    "month":            m,
                    "gamma_shape":      shape,
                    "gamma_scale":      scale,
                    "q_zero":           q_zero,
                })

            # Also compute SPI on observations themselves for the calibration period
            spi_obs = transform_to_spi(obs_acc, params, CFG)
            for d, v in zip(spi_obs.index, spi_obs.values):
                if np.isfinite(v):
                    long_rows.append({
                        "station":         sid,
                        "model":           "OBS",
                        "scenario":        "historical_obs",
                        "accumulation_k":  k,
                        "date":            d,
                        "spi":             float(v),
                    })

            # Project SPI from each (model, scenario)
            for (model, scenario), grp in (bc[bc["station"] == sid]
                                           .groupby(["model", "scenario"])):
                pr_series = (grp.set_index("date")["pr_corrected_mm"]
                                .sort_index())
                pr_series.index = pd.to_datetime(pr_series.index).to_period("M").to_timestamp()
                pr_acc = pr_series.rolling(k).sum()
                spi_proj = transform_to_spi(pr_acc, params, CFG)
                for d, v in zip(spi_proj.index, spi_proj.values):
                    if np.isfinite(v):
                        long_rows.append({
                            "station":         sid,
                            "model":           model,
                            "scenario":        scenario,
                            "accumulation_k":  k,
                            "date":            d,
                            "spi":             float(v),
                        })

    spi_long = pd.DataFrame(long_rows)
    spi_long["date"] = pd.to_datetime(spi_long["date"])

    spi_path = CFG.out_dir / "tables" / "spi_projection_long.csv"
    spi_long.to_csv(spi_path, index=False, float_format="%.4f")
    log.info("SPI long table -> %s  (%d rows)", spi_path, len(spi_long))

    gamma_df   = pd.DataFrame(gamma_rows)
    gamma_path = CFG.out_dir / "tables" / "gamma_calibration.csv"
    gamma_df.to_csv(gamma_path, index=False, float_format="%.5f")
    log.info("Gamma calibration  -> %s", gamma_path)

    # Drought-class frequency table per (station × model × scenario × k)
    spi_proj_only = spi_long[spi_long["scenario"].isin(("ssp2_4_5", "ssp5_8_5"))].copy()
    spi_proj_only["class"] = spi_proj_only["spi"].apply(
        lambda v: classify_drought(v, CFG))
    class_freq = (spi_proj_only.groupby(
                      ["station", "model", "scenario", "accumulation_k", "class"])
                                .size()
                                .unstack("class", fill_value=0))
    class_freq = class_freq.div(class_freq.sum(axis=1), axis=0).reset_index()
    for label in CFG.drought_class_labels:
        if label not in class_freq.columns:
            class_freq[label] = 0.0
    class_path = CFG.out_dir / "tables" / "drought_class_frequencies.csv"
    class_freq.to_csv(class_path, index=False, float_format="%.4f")
    log.info("Drought class freq -> %s", class_path)

    # Figures
    for k in (3, 6, 12):
        plot_spi_trajectories(spi_long, k, stations,
                              CFG.out_dir / "figures" / f"spi_trajectories_{k}.png")

    plot_drought_class_heatmap(class_freq,
                                CFG.out_dir / "figures" / "drought_class_heatmap.png")

    plot_spi_distributions(spi_long, 12, stations,
                            CFG.out_dir / "figures" / "spi_distributions.png")
    log.info("Figures            -> %s", CFG.out_dir / "figures")

    print("\n=== Projected drought summary (SPI-12, 2015-2040) ===")
    headline = (class_freq[class_freq["accumulation_k"] == 12]
                .assign(severe_or_worse=lambda d: d["extremely_dry"] + d["severely_dry"])
                .groupby(["station", "scenario"])["severe_or_worse"]
                .mean()
                .unstack("scenario") * 100)
    print(headline.round(1).to_string())


if __name__ == "__main__":
    main()
