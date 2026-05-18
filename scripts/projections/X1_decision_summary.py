"""
Author        : Imene DERRAR
Supervisor    : Dr. Abdelillah OTMANE CHERIF
Institutions  : NHSM — National Higher School of Mathematics
                IHFR — Institut Hydrométéorologique de Formation et de Recherche
Academic year : 2025–2026
Project       : Final-year engineering thesis in Applied Mathematics —
                Drought modelling and probabilistic forecasting for the Chélif basin

--------------------------------------------------------------------------------

X1 — Chéliff Drought-Risk Decision Summary
=====================================================

Single decision-oriented synthesis.  Consumes every R4 / R5 output
and produces:

  -  tables/chelif_drought_risk_summary.csv
        Long format, one row per (station × scenario), 13 columns.  This is
        the table to paste into the discussion chapter and the basin-authority
        briefing.

  -  figures/chelif_drought_scorecard.png
        Publication-ready scorecard.  Stations as rows (sorted by
        vulnerability rank under SSP2-4.5), risk metrics as columns,
        cells colour-coded by severity.  Annotated values inside each cell.
        One figure replaces ten tables on a slide.

  -  figures/chelif_risk_dashboard.png
        Three-panel dashboard for quick visual reading: historical-vs-
        projection delta, max-event-duration ranking, vulnerability score.

Inputs (all from 04_outputs/R5_drought_analysis/tables/)
    historical_vs_projection.csv
    event_summary.csv
    model_uncertainty.csv
    trend_tests.csv
    vulnerability_ranking.csv
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

# --------------------------------------------------------------------------- #
# 0. CONFIGURATION
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    accumulation_k: int = 12

    project_root: Path = Path(__file__).resolve().parents[3]
    in_dir:       Path = field(init=False)
    out_dir:      Path = field(init=False)

    def __post_init__(self) -> None:
        self.in_dir  = self.project_root / "04_outputs" / "R5_drought_analysis" / "tables"
        self.out_dir = self.project_root / "04_outputs" / "x1_decision_decision_summary"
        (self.out_dir / "tables").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "figures").mkdir(parents=True, exist_ok=True)


CFG = Config()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("x1_decision")


# --------------------------------------------------------------------------- #
# 1. BUILD THE DECISION SUMMARY TABLE
# --------------------------------------------------------------------------- #

def build_summary(cfg: Config) -> pd.DataFrame:
    hist_proj = pd.read_csv(cfg.in_dir / "historical_vs_projection.csv")
    events    = pd.read_csv(cfg.in_dir / "event_summary.csv")
    model_unc = pd.read_csv(cfg.in_dir / "model_uncertainty.csv")
    trends    = pd.read_csv(cfg.in_dir / "trend_tests.csv")
    vuln      = pd.read_csv(cfg.in_dir / "vulnerability_ranking.csv")

    k = cfg.accumulation_k
    hp = hist_proj[hist_proj["accumulation_k"] == k]
    ev = events[(events["accumulation_k"] == k) & (events["model"] != "OBS")]
    mu = model_unc[model_unc["accumulation_k"] == k]
    tr = trends[trends["accumulation_k"] == k]
    vr = vuln.copy()

    # (1) Historical: collapse over scenarios/models — historical is the same
    hist_per_station = (hp.groupby("station")["hist_freq_pct"]
                        .mean()
                        .rename("hist_severe_pct")
                        .reset_index())

    # (2) Projection mean ± model range, per (station × scenario)
    proj_stats = (hp.groupby(["station", "scenario"])
                  .agg(proj_mean_pct=("proj_freq_pct", "mean"),
                       proj_min_pct=("proj_freq_pct", "min"),
                       proj_max_pct=("proj_freq_pct", "max"))
                  .reset_index())

    # (3) Delta in percentage points, ensemble mean
    delta = (hp.groupby(["station", "scenario"])["change_pct_points"]
             .mean()
             .rename("delta_pp")
             .reset_index())

    # (4) Drought-event metrics — ensemble mean across the 3 retained models
    ev_agg = (ev.groupby(["station", "scenario"])
              .agg(max_duration_months=("max_duration", "mean"),
                   mean_duration_months=("mean_duration", "mean"),
                   mean_severity=("mean_severity", "mean"),
                   mean_intensity=("mean_intensity", "mean"),
                   n_events=("n_events", "mean"))
              .reset_index())
    # Range across models for the headline metric (max event duration)
    ev_range = (ev.groupby(["station", "scenario"])
                .agg(max_duration_min=("max_duration", "min"),
                     max_duration_max=("max_duration", "max"))
                .reset_index())

    # (5) Trend sign / significance — collapse across the 3 retained models
    def _trend_summary(g: pd.DataFrame) -> pd.Series:
        n_models = len(g)
        n_sig    = int((g["p_value"] < 0.05).sum())
        # Direction: main Sen-slope sign across models
        signs = np.sign(g["sen_slope_per_decade"]).fillna(0)
        if (signs > 0).sum() > (signs < 0).sum():
            direction = "wetter (↑)"
        elif (signs < 0).sum() > (signs > 0).sum():
            direction = "drier (↓)"
        else:
            direction = "mixed"
        # Median Sen slope per decade
        sen_med  = float(np.nanmedian(g["sen_slope_per_decade"]))
        return pd.Series({
            "trend_direction":    direction,
            "trend_n_significant": n_sig,
            "trend_n_models":      n_models,
            "trend_sen_per_decade_median": sen_med,
        })

    trend_agg = (tr.groupby(["station", "scenario"])
                 .apply(_trend_summary, include_groups=False)
                 .reset_index())

    # (6) Vulnerability rank
    rank_agg = (vr[["station", "scenario", "rank", "vulnerability_score"]]
                .rename(columns={"rank": "vulnerability_rank"}))

    # ---------------- Merge everything into one table --------------------- #
    summary = (proj_stats
               .merge(hist_per_station, on="station", how="left")
               .merge(delta,             on=["station", "scenario"], how="left")
               .merge(ev_agg,            on=["station", "scenario"], how="left")
               .merge(ev_range,          on=["station", "scenario"], how="left")
               .merge(trend_agg,         on=["station", "scenario"], how="left")
               .merge(rank_agg,          on=["station", "scenario"], how="left"))

    cols = [
        "vulnerability_rank", "station", "scenario",
        "hist_severe_pct",
        "proj_mean_pct", "proj_min_pct", "proj_max_pct",
        "delta_pp",
        "max_duration_months", "max_duration_min", "max_duration_max",
        "mean_duration_months", "mean_severity", "mean_intensity", "n_events",
        "trend_direction", "trend_n_significant", "trend_n_models",
        "trend_sen_per_decade_median",
        "vulnerability_score",
    ]
    summary = summary[cols].sort_values(["scenario", "vulnerability_rank"])
    return summary


# --------------------------------------------------------------------------- #
# 2. SCORECARD FIGURE (one panel, multi-metric, colour-coded cells)
# --------------------------------------------------------------------------- #

def plot_scorecard(summary: pd.DataFrame, scenario: str, out_path: Path) -> None:
    df = summary[summary["scenario"] == scenario].sort_values("vulnerability_rank")
    stations = df["station"].tolist()

    # Define the columns we want to display, with display-friendly headers
    display_cols = [
        ("hist_severe_pct",           "Hist. severe (%)",       "Reds"),
        ("proj_mean_pct",             "Proj. mean (%)",         "Reds"),
        ("delta_pp",                  "Δ (pp)",                 "diverging"),
        ("max_duration_months",       "Max event (mo)",         "Purples"),
        ("mean_severity",             "Mean severity",          "YlOrBr"),
        ("mean_intensity",            "Mean intensity",         "YlOrBr"),
        ("n_events",                  "N events",               "Greys"),
        ("vulnerability_score",       "Vuln. score",            "diverging"),
    ]

    n_rows = len(stations)
    n_cols = len(display_cols)

    fig, ax = plt.subplots(figsize=(1.45 * n_cols + 2.5, 0.55 * n_rows + 2.5))

    # Build a coloured cell for each (station, column) combination
    for j, (col_name, header, cmap_kind) in enumerate(display_cols):
        values = df[col_name].values.astype(float)
        if cmap_kind == "diverging":
            vmax = max(abs(np.nanmin(values)), abs(np.nanmax(values)), 1e-6)
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            cmap = plt.get_cmap("RdBu_r")
        else:
            vmin = float(np.nanmin(values))
            vmax = float(np.nanmax(values))
            if vmax == vmin: vmax = vmin + 1.0
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap(cmap_kind)
        for i, val in enumerate(values):
            color = cmap(norm(val)) if np.isfinite(val) else (0.9, 0.9, 0.9, 1.0)
            ax.add_patch(plt.Rectangle((j, n_rows - 1 - i), 1, 1,
                                        facecolor=color, edgecolor="white",
                                        linewidth=1.5))
            txt = f"{val:.1f}" if np.isfinite(val) else "—"
            # darker text on light cells, white text on dark cells
            lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            text_col = "white" if lum < 0.55 else "black"
            ax.text(j + 0.5, n_rows - 1 - i + 0.5, txt,
                    ha="center", va="center",
                    color=text_col, fontsize=9, fontweight="bold")

    # Trend direction column (text-only with arrows)
    j_trend = n_cols
    ax.text(j_trend + 0.5, n_rows + 0.4, "Trend\n(direction · #sig)",
            ha="center", va="center", fontsize=9, fontweight="bold")
    for i, (_, row) in enumerate(df.iterrows()):
        d = row["trend_direction"]
        n_sig = int(row["trend_n_significant"])
        n_mod = int(row["trend_n_models"])
        if "drier" in d:
            symbol, col = "▼", "firebrick"
        elif "wetter" in d:
            symbol, col = "▲", "steelblue"
        else:
            symbol, col = "■", "grey"
        ax.add_patch(plt.Rectangle((j_trend, n_rows - 1 - i), 1, 1,
                                    facecolor="white", edgecolor="lightgrey",
                                    linewidth=1.0))
        ax.text(j_trend + 0.5, n_rows - 1 - i + 0.5,
                f"{symbol}  {n_sig}/{n_mod}",
                ha="center", va="center",
                color=col, fontsize=10, fontweight="bold")

    # Headers
    for j, (_, header, _) in enumerate(display_cols):
        ax.text(j + 0.5, n_rows + 0.4, header,
                ha="center", va="center", fontsize=9, fontweight="bold")

    # Station names (with rank prefix)
    for i, (_, row) in enumerate(df.iterrows()):
        rank = int(row["vulnerability_rank"])
        ax.text(-0.2, n_rows - 1 - i + 0.5, f"#{rank}  {row['station']}",
                ha="right", va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(-3.0, n_cols + 1)
    ax.set_ylim(0, n_rows + 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.suptitle(
        f"Chéliff basin — drought-risk scorecard, SPI-12, 2015–2040 ({scenario})\n"
        f"stations sorted by vulnerability (most → least), trend column shows "
        f"direction and #models with p < 0.05",
        fontsize=11, y=0.99)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. THREE-PANEL DASHBOARD FIGURE
# --------------------------------------------------------------------------- #

def plot_dashboard(summary: pd.DataFrame, out_path: Path) -> None:
    """Three panels: (a) Δ severe-drought freq;  (b) max-event duration;
                     (c) vulnerability score.  Both scenarios on each panel."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))

    df = summary.copy()
    df["scenario_disp"] = df["scenario"].map({"ssp2_4_5": "SSP2-4.5",
                                                "ssp5_8_5": "SSP5-8.5"})

    # Sort stations by vulnerability rank under SSP2-4.5 (most → least)
    order = (df[df["scenario"] == "ssp2_4_5"]
             .sort_values("vulnerability_rank")["station"]
             .tolist())
    df["station_order"] = df["station"].map({s: i for i, s in enumerate(order)})

    # Panel (a) — Δ severe-drought freq
    ax = axes[0]
    width = 0.36
    for i, (scen, color) in enumerate([("ssp2_4_5", "firebrick"),
                                        ("ssp5_8_5", "darkorange")]):
        sub = df[df["scenario"] == scen].sort_values("station_order")
        x = np.arange(len(sub)) + (i - 0.5) * width
        ax.bar(x, sub["delta_pp"], width=width,
               color=color, alpha=0.85,
               label=sub["scenario_disp"].iloc[0])
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Δ severe-drought freq (pp)\nproj − hist on SPI-12")
    ax.set_title("(a) Change in severe-drought frequency", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    # Panel (b) — max event duration
    ax = axes[1]
    for i, (scen, color) in enumerate([("ssp2_4_5", "firebrick"),
                                        ("ssp5_8_5", "darkorange")]):
        sub = df[df["scenario"] == scen].sort_values("station_order")
        x = np.arange(len(sub)) + (i - 0.5) * width
        ax.bar(x, sub["max_duration_months"], width=width,
               color=color, alpha=0.85,
               label=sub["scenario_disp"].iloc[0],
               yerr=[sub["max_duration_months"] - sub["max_duration_min"],
                     sub["max_duration_max"] - sub["max_duration_months"]],
               capsize=3, ecolor="0.3", error_kw=dict(lw=0.7))
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Max drought-event duration (months)")
    ax.set_title("(b) Longest projected drought episode\n(error bars: model min–max)",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    # Panel (c) — vulnerability score (diverging)
    ax = axes[2]
    for i, (scen, color) in enumerate([("ssp2_4_5", "firebrick"),
                                        ("ssp5_8_5", "darkorange")]):
        sub = df[df["scenario"] == scen].sort_values("station_order")
        x = np.arange(len(sub)) + (i - 0.5) * width
        ax.bar(x, sub["vulnerability_score"], width=width,
               color=color, alpha=0.85,
               label=sub["scenario_disp"].iloc[0])
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Composite vulnerability z-score")
    ax.set_title("(c) Composite vulnerability\n(higher = more at-risk)", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle("Chéliff basin — climate-risk dashboard, SPI-12, 2015–2040",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. MAIN
# --------------------------------------------------------------------------- #

def main() -> None:
    log.info("Project root: %s", CFG.project_root)
    log.info("Reading 07/07b tables from %s", CFG.in_dir)

    summary = build_summary(CFG)
    out_csv = CFG.out_dir / "tables" / "chelif_drought_risk_summary.csv"
    summary.to_csv(out_csv, index=False, float_format="%.3f")
    log.info("Decision summary -> %s  (%d rows)", out_csv, len(summary))

    plot_scorecard(summary, "ssp2_4_5",
                    CFG.out_dir / "figures" / "chelif_drought_scorecard_ssp245.png")
    plot_scorecard(summary, "ssp5_8_5",
                    CFG.out_dir / "figures" / "chelif_drought_scorecard_ssp585.png")
    plot_dashboard(summary,
                    CFG.out_dir / "figures" / "chelif_risk_dashboard.png")
    log.info("Figures          -> %s", CFG.out_dir / "figures")

    # ---- Console headline ---- #
    print("\n=== Chéliff Drought-Risk Decision Summary (SPI-12, 2015–2040) ===")
    pretty = (summary[["vulnerability_rank", "station", "scenario",
                       "hist_severe_pct", "proj_mean_pct", "delta_pp",
                       "max_duration_months",
                       "trend_direction", "trend_n_significant",
                       "vulnerability_score"]]
              .rename(columns={
                  "vulnerability_rank":     "Rank",
                  "station":                "Station",
                  "scenario":               "Scenario",
                  "hist_severe_pct":        "Hist (%)",
                  "proj_mean_pct":          "Proj (%)",
                  "delta_pp":               "Δ (pp)",
                  "max_duration_months":    "MaxEv (mo)",
                  "trend_direction":        "Trend",
                  "trend_n_significant":    "#sig",
                  "vulnerability_score":    "VulnScore",
              }))
    print(pretty.to_string(index=False))


if __name__ == "__main__":
    main()
