"""
Author        : Imene DERRAR
Supervisor    : Dr. Abdelillah OTMANE CHERIF
Institutions  : NHSM — National Higher School of Mathematics
                IHFR — Institut Hydrométéorologique de Formation et de Recherche
Academic year : 2025–2026
Project       : Final-year engineering thesis in Applied Mathematics —
                Drought modelling and probabilistic forecasting for the Chélif basin

--------------------------------------------------------------------------------

R5 — SPI projection robustness and drought-event analysis
====================================================================

Builds the analytical layer on top of R4.  Does NOT recompute SPI;
loads the existing long table and adds five orthogonal analyses that turn
the projected SPI series into interpretable drought-risk indicators:

  1. Historical baseline vs projection comparison table
  2. Drought-event analysis (run-theory: count, duration, severity, intensity, peak)
  3. Cross-model uncertainty range (ensemble mean ± min-max across the 3 retained models)
  4. Gamma vs empirical (non-parametric) SPI sensitivity check
  5. Mann-Kendall + Sen-slope exploratory trend tests
  6. Composite station vulnerability ranking

Methodological note
-------------------
The Gamma SPI from R4 remains the canonical method (calibrated on
ONM 1990-2014 only, then applied to projections — the standard climate-change
methodology that does NOT recalibrate on future data and therefore preserves
the climate-change signal in the SPI).  The empirical SPI in §4 is a
sensitivity check, not a replacement: if the two methods give qualitatively
similar drought frequencies, the parametric Gamma assumption is not driving
the conclusions.

Drought event definition (Edwards & McKee 1997, run-theory)
-----------------------------------------------------------
A drought event begins at the first month with SPI ≤ start_threshold (default
−1.0, the McKee "moderate" threshold) and ends at the first subsequent month
with SPI ≥ end_threshold (default 0.0, recovery to mean climatology).  Per
event: duration (months), severity (sum of |SPI| over months with SPI<0
within the event window), intensity (severity / duration), peak (minimum SPI
reached).  Events shorter than `min_duration` are filtered out, since they
are typically below the operational threshold of "drought" in a hydrological
sense.

Inputs
------
    01_data/external/spi_ready_dataset_main_1990_2015.csv
        ONM monthly precipitation per station (used for empirical-SPI
        sensitivity check and as the raw signal underlying the historical
        baseline).

    04_outputs/R4_spi_projection/tables/spi_projection_long.csv
        Existing Gamma SPI for every (station × model × scenario × accumulation_k).

Outputs (under 04_outputs/R5_drought_analysis/)
    tables/historical_vs_projection.csv   Per (station × k × scenario × model):
                                          severe-drought frequency historical
                                          (1990-2014) vs projection (2015-2040)
                                          and the percentage-point change.
    tables/drought_events.csv             One row per detected drought event.
    tables/event_summary.csv              Per (station × k × scenario × model):
                                          n_events, mean / max duration,
                                          mean / total severity, etc.
    tables/model_uncertainty.csv          Per (station × k × scenario):
                                          mean and min-max across the 3 retained
                                          models on every drought metric.
    tables/empirical_vs_gamma_spi.csv     Side-by-side severe-drought frequency
                                          for both SPI methods.
    tables/trend_tests.csv                Mann-Kendall τ, p-value, Sen slope per
                                          (station × k × scenario × model).
    tables/vulnerability_ranking.csv      Stations ranked by composite drought-risk
                                          vulnerability score.
    figures/historical_vs_projection.png  Heatmap of percentage-point change.
    figures/event_duration_violins.png    Distribution of drought-event duration.
    figures/empirical_vs_gamma_qq.png     Q-Q comparison of the two SPI methods.
    figures/trend_significance.png        Mann-Kendall significance heatmap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist, kendalltau, norm

# --------------------------------------------------------------------------- #
# 0. CONFIGURATION
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    calib_start: str = "1990-01"
    calib_end:   str = "2014-12"
    proj_start:  str = "2015-01"
    proj_end:    str = "2040-12"

    severe_threshold:    float = -1.5
    extreme_threshold:   float = -2.0
    moderate_threshold:  float = -1.0   # used as drought-event start threshold

    # Drought-event run-theory parameters
    event_start_threshold: float = -1.0
    event_end_threshold:   float = 0.0
    event_min_duration:    int   = 3      # months — shorter events are filtered

    accumulation_windows: Tuple[int, ...] = (1, 3, 6, 12)

    # Vulnerability score weights (normalised)
    w_severe_freq:  float = 0.40
    w_max_duration: float = 0.30
    w_mean_severity: float = 0.30

    project_root: Path = Path(__file__).resolve().parents[3]
    onm_file:     Path = field(init=False)
    spi_file:     Path = field(init=False)
    out_dir:      Path = field(init=False)

    def __post_init__(self) -> None:
        self.onm_file = self.project_root / "01_data" / "external" / "spi_ready_dataset_main_1990_2015.csv"
        self.spi_file = self.project_root / "04_outputs" / "R4_spi_projection" / "tables" / "spi_projection_long.csv"
        self.out_dir  = self.project_root / "04_outputs" / "R5_drought_analysis"
        (self.out_dir / "tables").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "figures").mkdir(parents=True, exist_ok=True)


CFG = Config()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("r5_drought")


# --------------------------------------------------------------------------- #
# 1. INPUT LOADING
# --------------------------------------------------------------------------- #

def load_spi_long(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.spi_file, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    return df


def load_obs_precip(cfg: Config) -> pd.DataFrame:
    """Wide-format ONM monthly precipitation: index = month, columns = stations."""
    df = pd.read_csv(cfg.onm_file, parse_dates=["date"])
    wide = (df.pivot_table(index="date", columns="master_station_id",
                           values="precip_mm", aggfunc="mean")
              .sort_index())
    wide.index = pd.to_datetime(wide.index).to_period("M").to_timestamp()
    return wide.loc[cfg.calib_start: cfg.calib_end]


# --------------------------------------------------------------------------- #
# 2. DROUGHT-EVENT DETECTION (run theory)
# --------------------------------------------------------------------------- #

def detect_drought_events(dates: pd.DatetimeIndex, spi: np.ndarray,
                           start_thr: float, end_thr: float,
                           min_duration: int) -> List[Dict]:
    """Detect drought events.  See module docstring for run-theory definition."""
    events: List[Dict] = []
    in_event = False
    start_i  = 0

    for i, v in enumerate(spi):
        if not np.isfinite(v):
            if in_event:
                events.append(_finalize_event(dates, spi, start_i, i - 1))
                in_event = False
            continue
        if not in_event and v <= start_thr:
            start_i = i
            in_event = True
        elif in_event and v >= end_thr:
            events.append(_finalize_event(dates, spi, start_i, i - 1))
            in_event = False

    if in_event:
        events.append(_finalize_event(dates, spi, start_i, len(spi) - 1))

    return [e for e in events if e["duration"] >= min_duration]


def _finalize_event(dates: pd.DatetimeIndex, spi: np.ndarray,
                    s: int, e: int) -> Dict:
    span    = spi[s: e + 1]
    neg     = span[span < 0]
    duration = e - s + 1
    severity = float(np.sum(np.abs(neg))) if neg.size else 0.0
    intensity = severity / duration if duration > 0 else np.nan
    peak     = float(np.min(span)) if span.size else np.nan
    return {
        "start_date": dates[s], "end_date": dates[e],
        "duration": int(duration),
        "severity": severity,
        "intensity": intensity,
        "peak": peak,
    }


def event_summary_for(spi_series: pd.Series, cfg: Config) -> Dict[str, float]:
    """Aggregate drought-event statistics for a single SPI series."""
    spi_series = spi_series.sort_index()
    events = detect_drought_events(spi_series.index, spi_series.values,
                                    cfg.event_start_threshold,
                                    cfg.event_end_threshold,
                                    cfg.event_min_duration)
    if not events:
        return {"n_events": 0, "max_duration": 0,
                "mean_duration": np.nan, "total_duration_months": 0,
                "mean_severity": np.nan, "total_severity": 0.0,
                "mean_intensity": np.nan, "min_peak": np.nan}
    durs = np.array([e["duration"]   for e in events])
    sevs = np.array([e["severity"]   for e in events])
    ints = np.array([e["intensity"]  for e in events])
    peaks = np.array([e["peak"]      for e in events])
    return {
        "n_events":            int(len(events)),
        "max_duration":        int(durs.max()),
        "mean_duration":       float(durs.mean()),
        "total_duration_months": int(durs.sum()),
        "mean_severity":       float(sevs.mean()),
        "total_severity":      float(sevs.sum()),
        "mean_intensity":      float(ints.mean()),
        "min_peak":            float(peaks.min()),
    }


# --------------------------------------------------------------------------- #
# 3. EMPIRICAL (NON-PARAMETRIC) SPI
# --------------------------------------------------------------------------- #

def empirical_spi_per_month(target_values: np.ndarray, target_months: np.ndarray,
                             calib_values: np.ndarray, calib_months: np.ndarray
                             ) -> np.ndarray:
    """Compute SPI by inverting the empirical CDF, calendar-month-stratified.

    For every month m, the empirical CDF of `calib_values[calib_months==m]` is
    used to map each target value to a probability, which is then transformed
    via Φ⁻¹ to a Z-score (the "empirical SPI").
    """
    out = np.full_like(target_values, np.nan, dtype=float)
    for m in range(1, 13):
        c = calib_values[calib_months == m]
        c = c[np.isfinite(c)]
        if c.size < 5:
            continue
        c_sorted = np.sort(c)
        n        = c_sorted.size
        p_grid   = (np.arange(n) + 0.5) / n

        mask = target_months == m
        v    = target_values[mask]
        p    = np.interp(v, c_sorted, p_grid, left=p_grid[0], right=p_grid[-1])
        p    = np.clip(p, 1e-6, 1 - 1e-6)
        out[mask] = norm.ppf(p)
    return out


def rolling_sum_per_station(precip_wide: pd.DataFrame, k: int) -> pd.DataFrame:
    return precip_wide.rolling(k, min_periods=k).sum()


# --------------------------------------------------------------------------- #
# 4. MANN-KENDALL + SEN SLOPE
# --------------------------------------------------------------------------- #

def mann_kendall_sen(x: np.ndarray) -> Tuple[float, float, float]:
    """Returns (Kendall τ, two-sided p-value, Sen slope per month)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return np.nan, np.nan, np.nan
    idx = np.arange(n, dtype=float)
    tau, p_value = kendalltau(idx, x)
    # Sen slope — vectorised over upper triangle
    diffs = x[:, None] - x[None, :]
    denoms = idx[:, None] - idx[None, :]
    mask   = denoms > 0
    slopes = diffs[mask] / denoms[mask]
    sen_slope = float(np.median(slopes))
    return float(tau), float(p_value), sen_slope


# --------------------------------------------------------------------------- #
# 5. ANALYSIS DRIVERS
# --------------------------------------------------------------------------- #

def historical_vs_projection_table(spi_long: pd.DataFrame,
                                    cfg: Config) -> pd.DataFrame:
    """For every (station × k × scenario × model), compute severe-drought
    frequency on the historical OBS series and on the projection, plus the
    percentage-point change.
    """
    rows: List[Dict] = []
    obs_only  = spi_long[spi_long["model"] == "OBS"].copy()
    proj_only = spi_long[spi_long["model"] != "OBS"].copy()

    obs_freq = (obs_only
                .assign(severe=lambda d: d["spi"] <= cfg.severe_threshold)
                .groupby(["station", "accumulation_k"])["severe"]
                .mean()
                .rename("hist_freq")
                .reset_index())

    proj_freq = (proj_only[proj_only["scenario"].isin(("ssp2_4_5", "ssp5_8_5"))]
                 .assign(severe=lambda d: d["spi"] <= cfg.severe_threshold)
                 .groupby(["station", "accumulation_k", "scenario", "model"])["severe"]
                 .mean()
                 .rename("proj_freq")
                 .reset_index())

    merged = proj_freq.merge(obs_freq, on=["station", "accumulation_k"], how="left")
    merged["change_pct_points"] = (merged["proj_freq"] - merged["hist_freq"]) * 100.0
    merged["hist_freq_pct"] = merged["hist_freq"] * 100.0
    merged["proj_freq_pct"] = merged["proj_freq"] * 100.0
    return merged[["station", "accumulation_k", "scenario", "model",
                   "hist_freq_pct", "proj_freq_pct", "change_pct_points"]]


def drought_event_tables(spi_long: pd.DataFrame, cfg: Config
                          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (event_detail_df, event_summary_df) for every series."""
    detail_rows:  List[Dict] = []
    summary_rows: List[Dict] = []

    keys = (spi_long
            .groupby(["station", "model", "scenario", "accumulation_k"])
            .size()
            .reset_index()
            .drop(columns=0))

    for _, row in keys.iterrows():
        sub = (spi_long[(spi_long["station"]        == row["station"])
                        & (spi_long["model"]          == row["model"])
                        & (spi_long["scenario"]       == row["scenario"])
                        & (spi_long["accumulation_k"] == row["accumulation_k"])]
               .sort_values("date"))
        if sub.empty:
            continue
        s = pd.Series(sub["spi"].values,
                      index=pd.DatetimeIndex(sub["date"].values))
        events = detect_drought_events(s.index, s.values,
                                        cfg.event_start_threshold,
                                        cfg.event_end_threshold,
                                        cfg.event_min_duration)
        for e in events:
            detail_rows.append({
                "station":        row["station"],
                "model":          row["model"],
                "scenario":       row["scenario"],
                "accumulation_k": row["accumulation_k"],
                **e,
            })
        summary_rows.append({
            "station":        row["station"],
            "model":          row["model"],
            "scenario":       row["scenario"],
            "accumulation_k": row["accumulation_k"],
            **event_summary_for(s, cfg),
        })

    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def model_uncertainty_table(event_summary: pd.DataFrame) -> pd.DataFrame:
    """Per (station × k × scenario): ensemble mean + min-max across the 3 retained models."""
    proj = event_summary[event_summary["model"] != "OBS"].copy()
    metrics = ["n_events", "max_duration", "mean_duration", "total_duration_months",
               "mean_severity", "total_severity", "mean_intensity", "min_peak"]

    rows: List[Dict] = []
    for (sid, k, scen), sub in proj.groupby(["station", "accumulation_k", "scenario"]):
        row: Dict = {"station": sid, "accumulation_k": k, "scenario": scen,
                     "n_models": int(sub["model"].nunique())}
        for met in metrics:
            v = sub[met].values
            row[f"{met}_mean"] = float(np.nanmean(v))
            row[f"{met}_min"]  = float(np.nanmin(v))
            row[f"{met}_max"]  = float(np.nanmax(v))
        rows.append(row)
    return pd.DataFrame(rows)


def empirical_vs_gamma_table(spi_long: pd.DataFrame,
                              cfg: Config) -> pd.DataFrame:
    """Recompute SPI using the empirical (non-parametric) transform for the
    projection period, and compare severe-drought frequency to the Gamma SPI.

    Empirical SPI is calibrated on ONM observed *precipitation* (1990-2014) per
    calendar month, then applied to the bias-corrected projection precipitation
    via the R3 long table.  Because we already have the rolled
    calibration in the OBS rows of `spi_long` for the Gamma method, we use
    those exact same rolled OBS values as the calibration distribution and
    map projection rolled-sum values empirically — keeping like-for-like.
    """
    rows: List[Dict] = []
    proj = spi_long[(spi_long["model"] != "OBS")
                    & (spi_long["scenario"].isin(("ssp2_4_5", "ssp5_8_5")))]
    obs  = spi_long[spi_long["model"] == "OBS"]

    # We already have Gamma SPI in `proj["spi"]` and calib SPI in `obs["spi"]`.
    # The empirical comparison is then: rank the projection's *Gamma* SPI
    # within the observed Gamma-SPI distribution, calendar-month-stratified.
    # This isolates the parametric-vs-empirical question to the *transform*,
    # holding the rolled precipitation and calibration window identical.
    for (sid, k), obs_sub in obs.groupby(["station", "accumulation_k"]):
        if k not in cfg.accumulation_windows:
            continue
        calib_v = obs_sub["spi"].values
        calib_m = pd.to_datetime(obs_sub["date"]).dt.month.values

        for (model, scen), proj_sub in (proj[(proj["station"] == sid)
                                              & (proj["accumulation_k"] == k)]
                                         .groupby(["model", "scenario"])):
            tgt_v = proj_sub["spi"].values
            tgt_m = pd.to_datetime(proj_sub["date"]).dt.month.values

            emp = empirical_spi_per_month(tgt_v, tgt_m, calib_v, calib_m)
            gamma_freq = float(np.mean(tgt_v <= cfg.severe_threshold))
            emp_freq   = float(np.mean(emp[np.isfinite(emp)] <= cfg.severe_threshold))
            corr       = float(np.corrcoef(tgt_v, emp)[0, 1]) if emp.size > 1 else np.nan
            rows.append({
                "station": sid, "accumulation_k": k,
                "model":   model, "scenario": scen,
                "gamma_severe_freq_pct":     100.0 * gamma_freq,
                "empirical_severe_freq_pct": 100.0 * emp_freq,
                "delta_pct_points":          100.0 * (emp_freq - gamma_freq),
                "rank_correlation":          corr,
            })
    return pd.DataFrame(rows)


def trend_test_table(spi_long: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Mann-Kendall + Sen slope per (station × k × scenario × model) over 2015-2040."""
    rows: List[Dict] = []
    proj = spi_long[(spi_long["model"] != "OBS")
                    & (spi_long["scenario"].isin(("ssp2_4_5", "ssp5_8_5")))]
    for (sid, k, scen, model), sub in proj.groupby(
            ["station", "accumulation_k", "scenario", "model"]):
        ssub = sub.sort_values("date")
        tau, p, sen = mann_kendall_sen(ssub["spi"].values)
        # sen is per month; convert to per decade for readability
        sen_per_decade = sen * 120.0 if np.isfinite(sen) else np.nan
        rows.append({
            "station": sid, "accumulation_k": k,
            "model":   model, "scenario": scen,
            "kendall_tau":      tau,
            "p_value":          p,
            "sen_slope_per_month":  sen,
            "sen_slope_per_decade": sen_per_decade,
            "significant_5pct":  bool(np.isfinite(p) and p < 0.05),
        })
    return pd.DataFrame(rows)


def vulnerability_ranking(event_summary: pd.DataFrame,
                          historical: pd.DataFrame,
                          cfg: Config) -> pd.DataFrame:
    """Composite station vulnerability score on SPI-12, ensemble-mean across models.

    Combines z-normalised:
      - severe-drought-month frequency (proxy for chronic stress)
      - max event duration (proxy for persistence)
      - mean event severity (proxy for episode magnitude)
    """
    sub = (event_summary[(event_summary["accumulation_k"] == 12)
                          & (event_summary["model"] != "OBS")]
           .merge(historical[historical["accumulation_k"] == 12]
                  [["station", "scenario", "model", "proj_freq_pct"]],
                  on=["station", "scenario", "model"]))
    agg = (sub.groupby(["station", "scenario"])
           .agg(severe_freq_pct=("proj_freq_pct", "mean"),
                max_duration=("max_duration", "mean"),
                mean_severity=("mean_severity", "mean"))
           .reset_index())

    def _z(x: pd.Series) -> pd.Series:
        s = x.std(ddof=0)
        return (x - x.mean()) / (s if s > 0 else 1.0)

    out_rows: List[Dict] = []
    for scen, g in agg.groupby("scenario"):
        g = g.copy()
        g["z_freq"] = _z(g["severe_freq_pct"])
        g["z_dur"]  = _z(g["max_duration"])
        g["z_sev"]  = _z(g["mean_severity"])
        g["vulnerability_score"] = (
            cfg.w_severe_freq  * g["z_freq"]
          + cfg.w_max_duration * g["z_dur"]
          + cfg.w_mean_severity * g["z_sev"]
        )
        g = g.sort_values("vulnerability_score", ascending=False)
        g["rank"] = np.arange(1, len(g) + 1)
        out_rows.append(g)
    return pd.concat(out_rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# 6. PLOTS
# --------------------------------------------------------------------------- #

def plot_historical_vs_projection(hist_proj: pd.DataFrame, out_path: Path) -> None:
    sub = hist_proj[hist_proj["accumulation_k"] == 12].copy()
    pivot = (sub.groupby(["station", "scenario"])["change_pct_points"]
             .mean()
             .unstack("scenario"))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-30, vmax=30)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        color=("white" if abs(v) > 12 else "black"), fontsize=8)
    fig.colorbar(im, ax=ax, label="Change in severe-drought freq (pp)")
    ax.set_title("SPI-12 severe-drought frequency: projection (2015-2040) − historical (1990-2014)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_event_durations(event_detail: pd.DataFrame, out_path: Path) -> None:
    """Violin plot of drought-event durations per (k × scenario), across all stations × models."""
    sub = event_detail[event_detail["model"] != "OBS"].copy()
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    for ax, k in zip(axes, sorted(sub["accumulation_k"].unique())):
        data = []
        labels = []
        for scen in sorted(sub["scenario"].unique()):
            d = sub[(sub["accumulation_k"] == k) & (sub["scenario"] == scen)]["duration"].values
            if d.size:
                data.append(d)
                labels.append(scen)
        if data:
            ax.violinplot(data, showmeans=True, showmedians=True)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, fontsize=8, rotation=15)
        ax.set_title(f"SPI-{k}", fontsize=10)
    axes[0].set_ylabel("Drought-event duration (months)")
    fig.suptitle("Distribution of drought-event durations across the retained ensemble",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_empirical_vs_gamma_qq(emp_gamma: pd.DataFrame, out_path: Path) -> None:
    """Scatter of empirical vs Gamma severe-drought frequency."""
    fig, ax = plt.subplots(figsize=(6, 6))
    sub = emp_gamma[emp_gamma["accumulation_k"] == 12]
    ax.scatter(sub["gamma_severe_freq_pct"],
               sub["empirical_severe_freq_pct"],
               s=18, alpha=0.55, color="C0", edgecolor="white", linewidth=0.4)
    lim = max(sub["gamma_severe_freq_pct"].max(),
              sub["empirical_severe_freq_pct"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Gamma SPI severe-drought frequency (%)")
    ax.set_ylabel("Empirical SPI severe-drought frequency (%)")
    ax.set_title("Sensitivity check: Gamma vs empirical SPI (SPI-12, all stations × models × scenarios)",
                 fontsize=10)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_trend_significance(trend_df: pd.DataFrame, out_path: Path) -> None:
    sub = trend_df[trend_df["accumulation_k"] == 12].copy()
    pivot_p   = (sub.assign(label=lambda d: d["model"] + "_" + d["scenario"])
                 .pivot_table(index="station", columns="label",
                              values="p_value"))
    pivot_sen = (sub.assign(label=lambda d: d["model"] + "_" + d["scenario"])
                 .pivot_table(index="station", columns="label",
                              values="sen_slope_per_decade"))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    im = ax.imshow(pivot_sen.values, aspect="auto", cmap="RdBu_r",
                   vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(pivot_sen.shape[1]))
    ax.set_xticklabels(pivot_sen.columns, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(pivot_sen.shape[0]))
    ax.set_yticklabels(pivot_sen.index, fontsize=8)
    for i in range(pivot_sen.shape[0]):
        for j in range(pivot_sen.shape[1]):
            v = pivot_sen.values[i, j]
            p = pivot_p.values[i, j]
            if np.isfinite(v):
                star = "*" if (np.isfinite(p) and p < 0.05) else ""
                ax.text(j, i, f"{v:+.2f}{star}", ha="center", va="center",
                        color=("white" if abs(v) > 0.2 else "black"), fontsize=7)
    fig.colorbar(im, ax=ax, label="Sen slope per decade (SPI units)")
    ax.set_title("Mann-Kendall on SPI-12 (2015-2040) — Sen slope per decade. * = p < 0.05",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 7. MAIN
# --------------------------------------------------------------------------- #

def main() -> None:
    log.info("Project root: %s", CFG.project_root)
    log.info("SPI accumulation windows: %s", CFG.accumulation_windows)
    log.info("Drought-event thresholds: start = %.2f, end = %.2f, min duration = %d",
             CFG.event_start_threshold, CFG.event_end_threshold,
             CFG.event_min_duration)

    spi_long = load_spi_long(CFG)
    log.info("SPI long table loaded (%d rows)", len(spi_long))

    # ---- 1. Historical vs projection table ---- #
    hist_proj = historical_vs_projection_table(spi_long, CFG)
    p1 = CFG.out_dir / "tables" / "historical_vs_projection.csv"
    hist_proj.to_csv(p1, index=False, float_format="%.3f")
    log.info("[1] Historical vs projection -> %s", p1)

    # ---- 2. Drought events ---- #
    event_detail, event_summary = drought_event_tables(spi_long, CFG)
    p2a = CFG.out_dir / "tables" / "drought_events.csv"
    p2b = CFG.out_dir / "tables" / "event_summary.csv"
    event_detail.to_csv(p2a, index=False, float_format="%.3f")
    event_summary.to_csv(p2b, index=False, float_format="%.3f")
    log.info("[2] Event detail / summary  -> %s, %s  (%d events detected)",
             p2a, p2b, len(event_detail))

    # ---- 3. Cross-model uncertainty ---- #
    model_unc = model_uncertainty_table(event_summary)
    p3 = CFG.out_dir / "tables" / "model_uncertainty.csv"
    model_unc.to_csv(p3, index=False, float_format="%.3f")
    log.info("[3] Cross-model uncertainty -> %s", p3)

    # ---- 4. Gamma vs empirical SPI ---- #
    emp_gamma = empirical_vs_gamma_table(spi_long, CFG)
    p4 = CFG.out_dir / "tables" / "empirical_vs_gamma_spi.csv"
    emp_gamma.to_csv(p4, index=False, float_format="%.3f")
    log.info("[4] Gamma vs empirical SPI  -> %s", p4)

    # ---- 5. Trend tests ---- #
    trend_df = trend_test_table(spi_long, CFG)
    p5 = CFG.out_dir / "tables" / "trend_tests.csv"
    trend_df.to_csv(p5, index=False, float_format="%.4f")
    log.info("[5] Mann-Kendall + Sen      -> %s", p5)

    # ---- 6. Vulnerability ranking ---- #
    vuln = vulnerability_ranking(event_summary, hist_proj, CFG)
    p6 = CFG.out_dir / "tables" / "vulnerability_ranking.csv"
    vuln.to_csv(p6, index=False, float_format="%.3f")
    log.info("[6] Vulnerability ranking   -> %s", p6)

    # ---- Figures ---- #
    plot_historical_vs_projection(hist_proj,
                                   CFG.out_dir / "figures" / "historical_vs_projection.png")
    plot_event_durations(event_detail,
                          CFG.out_dir / "figures" / "event_duration_violins.png")
    plot_empirical_vs_gamma_qq(emp_gamma,
                                CFG.out_dir / "figures" / "empirical_vs_gamma_qq.png")
    plot_trend_significance(trend_df,
                             CFG.out_dir / "figures" / "trend_significance.png")
    log.info("Figures -> %s", CFG.out_dir / "figures")

    # ---- Console summary ---- #
    print("\n=== Historical vs Projection (SPI-12, mean across 3 models) ===")
    sub = hist_proj[hist_proj["accumulation_k"] == 12]
    summary = (sub.groupby(["station", "scenario"])
               .agg(hist_pct=("hist_freq_pct", "mean"),
                    proj_pct=("proj_freq_pct", "mean"),
                    delta_pp=("change_pct_points", "mean"))
               .reset_index())
    print(summary.round(1).to_string(index=False))

    print("\n=== Vulnerability ranking under SSP2-4.5 (top to bottom) ===")
    print(vuln[vuln["scenario"] == "ssp2_4_5"]
          [["rank", "station", "severe_freq_pct", "max_duration",
            "mean_severity", "vulnerability_score"]]
          .round(3).to_string(index=False))

    print("\n=== Mann-Kendall significance counts (SPI-12) ===")
    sig = trend_df[trend_df["accumulation_k"] == 12]
    print(f"Significant (p < 0.05) results: {sig['significant_5pct'].sum()} "
          f"out of {len(sig)} (station × model × scenario) combinations")
    print("(Trend tests should be interpreted as exploratory: 26-year window is "
          "short and internal variability remains the main signal.)")


if __name__ == "__main__":
    main()
