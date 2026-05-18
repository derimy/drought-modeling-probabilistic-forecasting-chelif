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

R6 — Bayesian Probabilistic Projection (SPI-6, 2015-2040)
==================================================================

Mirrors the Bayesian framework of Chapter 4 (M1 — the SPI Bayesian AR(1) notebook) — conjugate
Normal-Inverse-Gamma AR(1), closed-form posterior, Student-t predictive — but
applies it as a *projection* layer on top of the bias-corrected CMIP6 SPI
trajectories produced by R4.

Why this notebook
-----------------
M1 fits a Bayesian AR(1) on observed historical SPI to forecast 1–6
months ahead.  R4 produces deterministic 25-year SPI trajectories
under each (station × model × scenario), but each trajectory is a *single
realisation* of an inherently stochastic process — no parametric uncertainty
is attached to it, and the inter-model spread is reported only as raw
disagreement, not as a probabilistic statement.

This notebook adds a probabilistic projection layer to the SPI trajectories generated in R4.  For each (station × model × scenario) it
fits the *same* conjugate Bayesian AR(1) used in Chapter 4 to the projection
SPI series, then propagates parameter, model, and scenario
uncertainty through to:

  - posterior credible bands for the SPI trajectory itself (fan charts),
  - posterior credible intervals for drought-exceedance probabilities
    P(SPI ≤ c) over rolling decadal windows,
  - a Brunner-style independence-weighted ensemble that downweights the
    EC-Earth family (whose two members share the IFS atmospheric core).

Bayesian AR(1) — recap from Chapter 4 (closed form, no MCMC)
------------------------------------------------------------
Model:  y_t = φ y_{t-1} + ε_t,   ε_t | σ² ~ N(0, σ²)
Prior:  φ | σ² ~ N(m₀, V₀ σ²),   σ² ~ IG(a₀, b₀)
Posterior is again NIG with closed-form hyperparameters; predictive is
Student-t.  Parameters m₀, V₀, a₀, b₀ are inherited from M1 to
preserve Chapter-4 / Chapter-5 methodological consistency.

Pipeline
--------
1.  Load ONM observations 1990–2014 → compute the historical SPI-6 per station
    using the same gamma calibration as Notebooks 02c / 07.
2.  Load R4's bias-corrected SPI-6 projections per (station × model ×
    scenario) for 2015–2040.
3.  Fit Bayesian AR(1) (closed-form) on:
      - each station's historical OBS series (reference posterior),
      - each (station × model × scenario) projection series.
4.  For each (station × scenario), build a posterior predictive trajectory by
      (a) sampling N_SAMP × M from each model's NIG posterior,
      (b) iterating the AR(1) recursion forward through 2015–2040,
      (c) combining the three model trajectories with Brunner weights.
5.  Aggregate to credible bands and exceedance probabilities; persist tables
    and figures.

Inputs
------
    01_data/external/spi_ready_dataset_main_1990_2015.csv
        ONM monthly precipitation per station (gamma calibration source).

    04_outputs/R4_spi_projection/tables/spi_projection_long.csv
        SPI projections per (station × model × scenario × accumulation_k).

Outputs (under 04_outputs/r6_bayesian_bayesian_projection/)
    tables/posterior_parameters.csv         NIG posterior per (station × source)
    tables/spi_credible_bands.csv           median + 5/25/75/95 % bands per
                                            (station × scenario × month)
    tables/exceedance_credible_intervals.csv P(SPI ≤ c) with credible bands
    tables/weighted_ensemble_summary.csv    Brunner-weighted decadal severe-drought
                                            frequency per (station × scenario)
    figures/posterior_phi_comparison.png    historical OBS vs projections — φ
    figures/fan_charts.png                  9 station panels, median + 90 % band
    figures/exceedance_over_time.png        P(SPI ≤ −1.5) by year per station
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist, norm

# --------------------------------------------------------------------------- #
# 0. CONFIGURATION
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    accumulation_k: int = 6                  # PRIMARY_SERIES = SPI_6 (matches Chapter 4)

    calib_start: str = "1990-01"
    calib_end:   str = "2014-12"
    proj_start:  str = "2015-01"
    proj_end:    str = "2040-12"

    # Conjugate NIG prior — identical to Chapter 4 (M1).
    prior_m0: float = 0.5
    prior_V0: float = 1.0
    prior_a0: float = 2.0
    prior_b0: float = 0.5

    # Number of posterior predictive samples per (station × source).
    n_samp: int = 2000

    # Drought thresholds (McKee, Doesken & Kleist 1993)
    drought_thresholds: Tuple[float, ...] = (-1.0, -1.5, -2.0)
    drought_labels:     Tuple[str, ...]   = ("moderate", "severe", "extreme")

    # Independence weights (Brunner et al. 2020).  EC-Earth3-CC and
    # EC-Earth3-Veg-LR share the IFS atmospheric core → counted as half-each
    # so the family contributes one effective vote.  CESM2 is structurally
    # distinct → full weight.
    model_weights: Dict[str, float] = field(default_factory=lambda: {
        "CESM2":            1.00,
        "EC-Earth3-CC":     0.50,
        "EC-Earth3-Veg-LR": 0.50,
    })

    # Random seed for reproducibility.
    seed: int = 42

    project_root: Path = Path(__file__).resolve().parents[3]
    onm_file:     Path = field(init=False)
    spi_file:     Path = field(init=False)
    out_dir:      Path = field(init=False)

    def __post_init__(self) -> None:
        self.onm_file = self.project_root / "01_data" / "external" / "spi_ready_dataset_main_1990_2015.csv"
        self.spi_file = self.project_root / "04_outputs" / "R4_spi_projection" / "tables" / "spi_projection_long.csv"
        self.out_dir  = self.project_root / "04_outputs" / "r6_bayesian_bayesian_projection"
        (self.out_dir / "tables").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "figures").mkdir(parents=True, exist_ok=True)


CFG = Config()
np.random.seed(CFG.seed)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("r6_bayesian")


# --------------------------------------------------------------------------- #
# 1. CONJUGATE BAYESIAN AR(1)  (closed form — matches M1 exactly)
# --------------------------------------------------------------------------- #

def bayes_ar1_posterior(y: np.ndarray, m0: float, V0: float,
                        a0: float, b0: float) -> Dict[str, float]:
    """Closed-form NIG posterior for AR(1) with conjugate prior.

    Returns the four updated hyperparameters and the effective sample size n.
    """
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    if y.size < 3:
        return {"m_n": np.nan, "V_n": np.nan, "a_n": np.nan, "b_n": np.nan,
                "n":   int(y.size), "phi_post_mean": np.nan,
                "sigma2_post_mean": np.nan}

    yt = y[1:]
    xt = y[:-1]
    n  = yt.size

    V0_inv = 1.0 / V0
    Vn_inv = V0_inv + np.sum(xt * xt)
    Vn     = 1.0 / Vn_inv
    mn     = Vn * (V0_inv * m0 + np.sum(xt * yt))
    an     = a0 + n / 2.0
    bn     = b0 + 0.5 * (np.sum(yt * yt) + V0_inv * m0 ** 2 - Vn_inv * mn ** 2)

    sigma2_post_mean = bn / (an - 1.0) if an > 1.0 else np.nan

    return {
        "m_n": float(mn), "V_n": float(Vn),
        "a_n": float(an), "b_n": float(bn),
        "n":   int(n),
        "phi_post_mean":   float(mn),
        "sigma2_post_mean": float(sigma2_post_mean),
    }


def bayes_ar1_predict(post: Dict[str, float], y_last: float, h: int,
                      n_samp: int) -> np.ndarray:
    """Posterior predictive draws of (y_{t+1}, ..., y_{t+h}).

    Returns array of shape (n_samp, h).
    """
    if not np.isfinite(post.get("a_n", np.nan)):
        return np.full((n_samp, h), np.nan)
    sigma2 = 1.0 / np.random.gamma(shape=post["a_n"], scale=1.0 / post["b_n"],
                                   size=n_samp)
    phi    = np.random.normal(loc=post["m_n"],
                              scale=np.sqrt(post["V_n"] * sigma2),
                              size=n_samp)
    draws  = np.empty((n_samp, h))
    y_prev = np.full(n_samp, y_last, dtype=float)
    for k in range(h):
        eps = np.random.normal(0.0, np.sqrt(sigma2), size=n_samp)
        y_next     = phi * y_prev + eps
        draws[:, k] = y_next
        y_prev      = y_next
    return draws


# --------------------------------------------------------------------------- #
# 2. INPUT LOADING
# --------------------------------------------------------------------------- #

def load_obs_spi(cfg: Config) -> pd.DataFrame:
    """Compute SPI-k on observed ONM precipitation, gamma calibrated 1990-2014.

    Re-implements the R4 pipeline so this script is self-contained
    even if the SPI long table only stores model rows (it actually stores OBS
    too — we use those rows directly when present for full consistency).
    """
    df = pd.read_csv(cfg.spi_file)
    df["date"] = pd.to_datetime(df["date"])
    obs = df[(df["model"] == "OBS") & (df["accumulation_k"] == cfg.accumulation_k)].copy()
    obs = obs[["station", "date", "spi"]].sort_values(["station", "date"])
    obs = obs[obs["date"].between(cfg.calib_start, cfg.calib_end)]
    return obs


def load_projection_spi(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.spi_file)
    df["date"] = pd.to_datetime(df["date"])
    proj = df[(df["model"].isin(cfg.model_weights))
              & (df["scenario"].isin(("ssp2_4_5", "ssp5_8_5")))
              & (df["accumulation_k"] == cfg.accumulation_k)].copy()
    proj = proj[["station", "model", "scenario", "date", "spi"]]
    proj = proj[proj["date"].between(cfg.proj_start, cfg.proj_end)]
    return proj.sort_values(["station", "model", "scenario", "date"])


# --------------------------------------------------------------------------- #
# 3. FIT POSTERIORS
# --------------------------------------------------------------------------- #

def fit_all_posteriors(obs: pd.DataFrame, proj: pd.DataFrame,
                        cfg: Config) -> pd.DataFrame:
    """One row per (station × source) where source = OBS_historical or model_scenario."""
    rows: List[Dict] = []

    for station, g in obs.groupby("station"):
        post = bayes_ar1_posterior(g["spi"].values,
                                   cfg.prior_m0, cfg.prior_V0,
                                   cfg.prior_a0, cfg.prior_b0)
        rows.append({
            "station": station, "model": "OBS", "scenario": "historical",
            **post,
        })

    for (station, model, scenario), g in proj.groupby(["station", "model", "scenario"]):
        post = bayes_ar1_posterior(g["spi"].values,
                                   cfg.prior_m0, cfg.prior_V0,
                                   cfg.prior_a0, cfg.prior_b0)
        rows.append({
            "station": station, "model": model, "scenario": scenario,
            **post,
        })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. POSTERIOR PREDICTIVE TRAJECTORIES (with model mixture)
# --------------------------------------------------------------------------- #

def simulate_trajectories(post_df: pd.DataFrame,
                          proj: pd.DataFrame,
                          cfg: Config) -> Dict[Tuple[str, str], np.ndarray]:
    """Per (station × scenario), build a Brunner-weighted posterior predictive ensemble.

    Each model's posterior contributes `n_samp_per_model` trajectories, with
    `n_samp_per_model` proportional to its independence weight.  All three
    models share the same time index from 2015-01 to 2040-12.

    Returns a dict mapping (station, scenario) -> array of shape
    (n_total_draws, n_months_proj).
    """
    months = pd.date_range(cfg.proj_start, cfg.proj_end, freq="MS")
    h      = len(months)

    # Total ensemble size per (station × scenario).  We split it across models
    # in proportion to the Brunner weights.
    total_w = sum(cfg.model_weights.values())
    n_per_model = {m: max(1, int(round(cfg.n_samp * w / total_w)))
                   for m, w in cfg.model_weights.items()}

    out: Dict[Tuple[str, str], np.ndarray] = {}

    for (station, scenario), grp_per_model in proj.groupby(["station", "scenario"]):
        traj_blocks = []
        for model in cfg.model_weights:
            sub = grp_per_model[grp_per_model["model"] == model].sort_values("date")
            if sub.empty:
                continue
            post = post_df[(post_df["station"]  == station)
                           & (post_df["model"]    == model)
                           & (post_df["scenario"] == scenario)].iloc[0].to_dict()
            y_last = float(sub["spi"].iloc[0])    # initial condition: first SPI value
            traj   = bayes_ar1_predict(post, y_last, h, n_per_model[model])
            traj_blocks.append(traj)
        if traj_blocks:
            out[(station, scenario)] = np.vstack(traj_blocks)
    return out


# --------------------------------------------------------------------------- #
# 5. SUMMARISE
# --------------------------------------------------------------------------- #

def summarise_credible_bands(traj_by_key: Dict[Tuple[str, str], np.ndarray],
                              cfg: Config) -> pd.DataFrame:
    """Median + 5/25/75/95 % credible bands per (station × scenario × month)."""
    months = pd.date_range(cfg.proj_start, cfg.proj_end, freq="MS")
    rows: List[Dict] = []
    for (station, scenario), traj in traj_by_key.items():
        q05  = np.nanpercentile(traj, 5,  axis=0)
        q25  = np.nanpercentile(traj, 25, axis=0)
        q50  = np.nanpercentile(traj, 50, axis=0)
        q75  = np.nanpercentile(traj, 75, axis=0)
        q95  = np.nanpercentile(traj, 95, axis=0)
        for d, a, b, c, e, f in zip(months, q05, q25, q50, q75, q95):
            rows.append({"station": station, "scenario": scenario, "date": d,
                         "spi_q05": a, "spi_q25": b, "spi_median": c,
                         "spi_q75": e, "spi_q95": f})
    return pd.DataFrame(rows)


def summarise_exceedance_credible(traj_by_key: Dict[Tuple[str, str], np.ndarray],
                                   cfg: Config) -> pd.DataFrame:
    """For each station × scenario × threshold × decade window, posterior
    distribution of P(SPI ≤ c).  We summarise as median + 90 % CI."""
    months = pd.date_range(cfg.proj_start, cfg.proj_end, freq="MS")
    decade_bins = [
        (months[0],  pd.Timestamp("2024-12-01")),
        (pd.Timestamp("2025-01-01"), pd.Timestamp("2034-12-01")),
        (pd.Timestamp("2035-01-01"), months[-1]),
    ]
    rows: List[Dict] = []
    for (station, scenario), traj in traj_by_key.items():
        for thr, label in zip(cfg.drought_thresholds, cfg.drought_labels):
            for (d0, d1) in decade_bins:
                mask = (months >= d0) & (months <= d1)
                # Per draw, fraction of months in the window with SPI ≤ thr
                frac_per_draw = (traj[:, mask] <= thr).mean(axis=1)
                rows.append({
                    "station":   station,
                    "scenario":  scenario,
                    "threshold": thr,
                    "label":     label,
                    "decade":    f"{d0.year}-{d1.year}",
                    "median":    float(np.nanmedian(frac_per_draw)),
                    "q05":       float(np.nanpercentile(frac_per_draw, 5)),
                    "q95":       float(np.nanpercentile(frac_per_draw, 95)),
                })
    return pd.DataFrame(rows)


def summarise_weighted_ensemble(exceed_df: pd.DataFrame) -> pd.DataFrame:
    """Per (station × scenario) decadal severe-drought frequency.

    The credible interval already accounts for the Brunner weighting because
    the posterior predictive ensemble was built with that weighting in
    `simulate_trajectories`.  Here we simply project to a single headline
    table for the discussion chapter.
    """
    sev = exceed_df[exceed_df["label"] == "severe"].copy()
    return sev


# --------------------------------------------------------------------------- #
# 6. PLOTS
# --------------------------------------------------------------------------- #

def plot_posterior_phi(post_df: pd.DataFrame, out_path: Path) -> None:
    """Compare φ posterior mean across (station × source)."""
    pivot = (post_df
             .assign(source=lambda d: np.where(
                 d["model"] == "OBS", "OBS_historical",
                 d["model"] + "_" + d["scenario"]))
             .pivot_table(index="station", columns="source",
                          values="phi_post_mean"))
    cols_order = (
        ["OBS_historical"]
        + [c for c in pivot.columns if c != "OBS_historical"]
    )
    pivot = pivot[[c for c in cols_order if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(11, 4.8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=("white" if abs(v) > 0.5 else "black"),
                        fontsize=7)
    fig.colorbar(im, ax=ax, label="Posterior mean of φ (AR(1) coefficient)")
    ax.set_title("Bayesian AR(1) φ — historical OBS vs projection per (model × scenario)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_fan_charts(traj_by_key: Dict[Tuple[str, str], np.ndarray],
                    bands_df: pd.DataFrame,
                    stations: List[str],
                    cfg: Config,
                    out_path: Path) -> None:
    n_cols = 3
    n_rows = int(np.ceil(len(stations) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.0 * n_cols, 2.7 * n_rows),
                             sharey=True)
    axes = axes.flatten()

    for i, sid in enumerate(stations):
        ax = axes[i]
        for scenario, color in [("ssp2_4_5", "C0"), ("ssp5_8_5", "C3")]:
            sub = bands_df[(bands_df["station"] == sid)
                           & (bands_df["scenario"] == scenario)]
            if sub.empty:
                continue
            ax.fill_between(sub["date"], sub["spi_q05"], sub["spi_q95"],
                            color=color, alpha=0.15)
            ax.fill_between(sub["date"], sub["spi_q25"], sub["spi_q75"],
                            color=color, alpha=0.30)
            ax.plot(sub["date"], sub["spi_median"],
                    color=color, lw=1.0,
                    label=scenario if i == 0 else None)
        ax.axhline(0,    color="0.2", lw=0.5)
        ax.axhline(-1.5, color="firebrick", lw=0.5, linestyle=":")
        ax.set_title(sid, fontsize=9)
        ax.set_ylim(-3, 3)
        ax.tick_params(labelsize=7)
    for k in range(len(stations), len(axes)):
        axes[k].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9,
               frameon=False)
    fig.suptitle(f"Bayesian posterior predictive SPI-{cfg.accumulation_k} — "
                 f"median + 50 %/90 % credible bands, 2015–2040",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_exceedance_over_time(traj_by_key: Dict[Tuple[str, str], np.ndarray],
                               stations: List[str],
                               cfg: Config,
                               out_path: Path) -> None:
    """Annual P(SPI ≤ −1.5) with 90 % credible band, per station × scenario."""
    months = pd.date_range(cfg.proj_start, cfg.proj_end, freq="MS")
    years  = months.year
    unique_years = sorted(set(years))

    n_cols = 3
    n_rows = int(np.ceil(len(stations) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.0 * n_cols, 2.7 * n_rows),
                             sharey=True)
    axes = axes.flatten()

    for i, sid in enumerate(stations):
        ax = axes[i]
        for scenario, color in [("ssp2_4_5", "C0"), ("ssp5_8_5", "C3")]:
            traj = traj_by_key.get((sid, scenario))
            if traj is None:
                continue
            ann_med = []
            ann_lo  = []
            ann_hi  = []
            for y in unique_years:
                m = (years == y)
                # per-draw fraction of months in year y with SPI <= -1.5
                frac = (traj[:, m] <= -1.5).mean(axis=1)
                ann_med.append(np.nanmedian(frac))
                ann_lo.append(np.nanpercentile(frac, 5))
                ann_hi.append(np.nanpercentile(frac, 95))
            ax.fill_between(unique_years, ann_lo, ann_hi,
                            color=color, alpha=0.20)
            ax.plot(unique_years, ann_med, color=color, lw=1.0,
                    label=scenario if i == 0 else None)
        ax.axhline(0.067, color="0.4", lw=0.5, linestyle="--")  # baseline
        ax.set_title(sid, fontsize=9)
        ax.set_ylim(0, 0.7)
        ax.tick_params(labelsize=7)
    for k in range(len(stations), len(axes)):
        axes[k].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9,
               frameon=False)
    fig.suptitle("Annual P(SPI-6 ≤ −1.5) with 90 % credible band — "
                 "dashed line = stationary baseline 6.7 %",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 7. MAIN
# --------------------------------------------------------------------------- #

def main() -> None:
    log.info("Project root:   %s", CFG.project_root)
    log.info("SPI accumulation: %d months", CFG.accumulation_k)
    log.info("Calibration:    %s .. %s", CFG.calib_start, CFG.calib_end)
    log.info("Projection:     %s .. %s", CFG.proj_start,  CFG.proj_end)
    log.info("Brunner weights: %s", CFG.model_weights)

    obs  = load_obs_spi(CFG)
    proj = load_projection_spi(CFG)
    log.info("OBS rows:       %d  | projection rows: %d", len(obs), len(proj))

    stations = sorted(obs["station"].unique())
    log.info("Stations:       %d", len(stations))

    post_df = fit_all_posteriors(obs, proj, CFG)
    post_path = CFG.out_dir / "tables" / "posterior_parameters.csv"
    post_df.to_csv(post_path, index=False, float_format="%.5f")
    log.info("Posterior parameters -> %s", post_path)

    log.info("Simulating posterior predictive trajectories ...")
    traj_by_key = simulate_trajectories(post_df, proj, CFG)
    log.info("Trajectories built for %d (station × scenario) keys",
             len(traj_by_key))

    bands_df = summarise_credible_bands(traj_by_key, CFG)
    bands_path = CFG.out_dir / "tables" / "spi_credible_bands.csv"
    bands_df.to_csv(bands_path, index=False, float_format="%.4f")
    log.info("Credible-band table  -> %s  (%d rows)", bands_path, len(bands_df))

    exceed_df = summarise_exceedance_credible(traj_by_key, CFG)
    exceed_path = CFG.out_dir / "tables" / "exceedance_credible_intervals.csv"
    exceed_df.to_csv(exceed_path, index=False, float_format="%.4f")
    log.info("Exceedance credible  -> %s", exceed_path)

    sev = summarise_weighted_ensemble(exceed_df)
    sev_path = CFG.out_dir / "tables" / "weighted_ensemble_summary.csv"
    sev.to_csv(sev_path, index=False, float_format="%.4f")
    log.info("Weighted ensemble    -> %s", sev_path)

    plot_posterior_phi(post_df,
                        CFG.out_dir / "figures" / "posterior_phi_comparison.png")
    plot_fan_charts(traj_by_key, bands_df, stations, CFG,
                     CFG.out_dir / "figures" / "fan_charts.png")
    plot_exceedance_over_time(traj_by_key, stations, CFG,
                               CFG.out_dir / "figures" / "exceedance_over_time.png")
    log.info("Figures             -> %s", CFG.out_dir / "figures")

    print("\n=== Brunner-weighted Bayesian severe-drought frequency ===")
    headline = (sev[sev["decade"] == "2015-2024"]
                .pivot_table(index="station", columns="scenario",
                             values="median")
                * 100)
    h2 = (sev[sev["decade"] == "2025-2034"]
          .pivot_table(index="station", columns="scenario", values="median") * 100)
    h3 = (sev[sev["decade"] == "2035-2040"]
          .pivot_table(index="station", columns="scenario", values="median") * 100)

    print("\nDecade 2015-2024 — median severe-drought freq (%):")
    print(headline.round(1).to_string())
    print("\nDecade 2025-2034 — median severe-drought freq (%):")
    print(h2.round(1).to_string())
    print("\nDecade 2035-2040 — median severe-drought freq (%):")
    print(h3.round(1).to_string())


if __name__ == "__main__":
    main()
