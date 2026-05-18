"""
Author        : Imene DERRAR
Supervisor    : Dr. Abdelillah OTMANE CHERIF
Institutions  : NHSM — National Higher School of Mathematics
                IHFR — Institut Hydrométéorologique de Formation et de Recherche
Academic year : 2025–2026
Project       : Final-year engineering thesis in Applied Mathematics —
                Drought modelling and probabilistic forecasting for the Chélif basin

--------------------------------------------------------------------------------

"""

#!/usr/bin/env python3
# =====================================================================
# R16_chapter5_figures.py
# ---------------------------------------------------------------------
# Genere les 5 figures principales du Chapitre 5 du rapport PFE.
#
# Inputs :
#   02_processed/drought_indices/projection/scpdsi_projection_long.csv
#       (continuous, output from R12)
#   02_processed/drought_indices/projection/scpdsi_projection_frozen_long.csv
#       (frozen, sortie Notebook 16b)
#
# Outputs (sauvegardes en .pdf ET .png) :
#   04_outputs/R16_chapter5_figures/
#       fig_01_timeseries_frozen.{pdf,png}     <- LA figure principale
#       fig_02_frozen_vs_continuous.{pdf,png}  <- justifie le pivot methodo
#       fig_03_decadal_severe_freq.{pdf,png}   <- chiffres cles 2035-2040
#       fig_04_climate_signal_A5.{pdf,png}     <- forced vs AR(1) naive
#       fig_05_drought_class_shift.{pdf,png}   <- distribution shift
# =====================================================================

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
LEGACY_ROOT     = Path(os.environ.get("CHELIF_PROJECT_ROOT", Path.cwd()))
STRUCTURED_ROOT = LEGACY_ROOT / "_structured_project"
PROJ_DIR        = STRUCTURED_ROOT / "02_processed" / "drought_indices" / "projection"

CONTINUOUS_CSV = PROJ_DIR / "scpdsi_projection_long.csv"
FROZEN_CSV     = PROJ_DIR / "scpdsi_projection_frozen_long.csv"

OUT_DIR = STRUCTURED_ROOT / "04_outputs" / "R16_chapter5_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS    = ["ACCESS-CM2", "CMCC-ESM2", "GFDL-ESM4"]
SCENARIOS = ["ssp2_4_5", "ssp5_8_5"]
SCEN_PRETTY = {"ssp2_4_5": "SSP2-4.5", "ssp5_8_5": "SSP5-8.5"}

# Palette : un modele = une couleur, deux scenarios = ligne pleine vs pointillee
COLORS = {
    "ACCESS-CM2": "#1F78B4",   # bleu
    "CMCC-ESM2":  "#33A02C",   # vert
    "GFDL-ESM4":  "#E31A1C",   # rouge
}
SCEN_LS = {"ssp2_4_5": "-", "ssp5_8_5": "--"}

# Style commun publication-quality
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.linewidth":   0.7,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linewidth":   0.4,
    "legend.frameon":   False,
    "figure.dpi":       100,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})


def save(fig, name):
    """Sauvegarde fig en .pdf + .png."""
    for ext in (".pdf", ".png"):
        path = OUT_DIR / f"{name}{ext}"
        fig.savefig(path)
        print(f"  -> {path.name}")


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
print("[load] continuous + frozen long-format")
df_cont   = pd.read_csv(CONTINUOUS_CSV, parse_dates=["date"])
df_frozen = pd.read_csv(FROZEN_CSV,     parse_dates=["date"])
print(f"  continuous : {len(df_cont)} rows")
print(f"  frozen     : {len(df_frozen)} rows")


# =====================================================================
# Figure 1 -- Timeseries scPDSI frozen, 6 panneaux
# =====================================================================
print("\n[fig 1] timeseries frozen, 6 panneaux")
fig, axes = plt.subplots(3, 2, figsize=(11, 8.5), sharex=True, sharey=True)

for i, m in enumerate(MODELS):
    for j, s in enumerate(SCENARIOS):
        ax = axes[i, j]
        sub = df_frozen[(df_frozen["model"] == m) &
                        (df_frozen["scenario"] == s)].copy()

        # Periode historique en gris
        hist = sub[sub["date"] <= "2014-12-01"]
        proj = sub[sub["date"] >= "2015-01-01"]

        ax.plot(hist["date"], hist["scPDSI"], color="grey",
                lw=0.6, label="Historique 1990-2014")
        ax.plot(proj["date"], proj["scPDSI"], color=COLORS[m],
                lw=0.6, label=f"Projection {SCEN_PRETTY[s]}")

        # Lignes de reference Palmer
        ax.axhline(0,  color="black", lw=0.3, ls="-")
        for thresh, label in [(-2, "severe"), (-3, "extreme"), (-4, "extreme intense")]:
            ax.axhline(thresh, color="firebrick", lw=0.3, ls=":", alpha=0.5)

        # Bande 2035-2040 (la fenetre focale)
        ax.axvspan(pd.Timestamp("2035-01-01"), pd.Timestamp("2040-12-01"),
                   color="gold", alpha=0.12, zorder=0)

        # Annotations
        ax.set_title(f"{m} -- {SCEN_PRETTY[s]}", fontsize=10, fontweight="bold",
                     pad=4)
        ax.set_ylim(-7.5, 7.5)
        if i == 2:
            ax.set_xlabel("Annee")
        if j == 0:
            ax.set_ylabel("scPDSI")
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Legende globale en bas
handles = [
    plt.Line2D([0], [0], color="grey", lw=1, label="Historique 1990-2014"),
    plt.Line2D([0], [0], color=COLORS["ACCESS-CM2"], lw=1, label="ACCESS-CM2"),
    plt.Line2D([0], [0], color=COLORS["CMCC-ESM2"],  lw=1, label="CMCC-ESM2"),
    plt.Line2D([0], [0], color=COLORS["GFDL-ESM4"],  lw=1, label="GFDL-ESM4"),
    plt.Line2D([0], [0], color="firebrick", lw=0.7, ls=":", label="Seuils Palmer (-2, -3, -4)"),
    Patch(facecolor="gold", alpha=0.3, label="Fenetre focale 2035-2040"),
]
fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle("scPDSI projete -- calibration figee 1990-2014 -- bassin du Chelif",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0.04, 1, 0.96])
save(fig, "fig_01_timeseries_frozen")
plt.close(fig)


# =====================================================================
# Figure 2 -- Frozen vs continuous (artefact methodologique)
# =====================================================================
print("\n[fig 2] frozen vs continuous (CMCC SSP5-8.5, illustration de l'artefact)")
fig, ax = plt.subplots(figsize=(10, 5))

m, s = "CMCC-ESM2", "ssp5_8_5"
sub_c = df_cont[(df_cont["model"] == m) & (df_cont["scenario"] == s)]
sub_f = df_frozen[(df_frozen["model"] == m) & (df_frozen["scenario"] == s)]

ax.plot(sub_c["date"], sub_c["scPDSI"], color="#888888", lw=0.8,
        label="Calibration continue 1990-2040 (artefact)")
ax.plot(sub_f["date"], sub_f["scPDSI"], color="#C03020", lw=0.8,
        label="Calibration figee 1990-2014 (corrigee)")

ax.axhline(0,  color="black", lw=0.3)
ax.axhline(-2, color="firebrick", lw=0.4, ls=":", alpha=0.6)
ax.axhline(-3, color="firebrick", lw=0.4, ls=":", alpha=0.4)
ax.axhline(-4, color="firebrick", lw=0.4, ls=":", alpha=0.3)

# Annotation periode focale
ax.axvspan(pd.Timestamp("2035-01-01"), pd.Timestamp("2040-12-01"),
           color="gold", alpha=0.12)

# Calculer les frequences <=-2 sur 2035-2040 pour annoter
def freq_severe(df_, start, end):
    s_ = df_[(df_["date"] >= start) & (df_["date"] <= end)]
    return (s_["scPDSI"] <= -2).mean() * 100

f_c = freq_severe(sub_c, "2035-01-01", "2040-12-01")
f_f = freq_severe(sub_f, "2035-01-01", "2040-12-01")

ax.annotate(
    f"2035-2040 freq(scPDSI <= -2):\n"
    f"  continuous = {f_c:.1f} %\n"
    f"  frozen     = {f_f:.1f} %\n"
    f"  artefact   = {f_c - f_f:+.1f} pp",
    xy=(pd.Timestamp("2037-06-01"), -6.5), xycoords="data",
    ha="center", va="bottom", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.4", fc="#FDF6EC", ec="#C96F4A", lw=0.8),
)

ax.set_xlabel("Annee")
ax.set_ylabel("scPDSI")
ax.set_title(
    f"Artefact de la calibration sc=TRUE sur 1990-2040 "
    f"(modele = {m}, scenario = {SCEN_PRETTY[s]})",
    fontsize=11, fontweight="bold"
)
ax.set_ylim(-9, 9)
ax.legend(loc="upper left", fontsize=9)
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

fig.tight_layout()
save(fig, "fig_02_frozen_vs_continuous")
plt.close(fig)


# =====================================================================
# Figure 3 -- Frequence decennale des mois en secheresse severe (<= -2)
# =====================================================================
print("\n[fig 3] frequence decennale severe drought")

def decadal_freq(df_, models, scenarios):
    rows = []
    df_ = df_.copy()
    df_["decade"] = pd.cut(df_["date"],
        bins=[pd.Timestamp("1990-01-01"), pd.Timestamp("2015-01-01"),
              pd.Timestamp("2025-01-01"), pd.Timestamp("2035-01-01"),
              pd.Timestamp("2041-01-01")],
        labels=["1990-2014", "2015-2024", "2025-2034", "2035-2040"],
        right=False)
    for m in models:
        for s in scenarios:
            for d, g in df_[(df_["model"] == m) &
                            (df_["scenario"] == s)].groupby("decade", observed=True):
                if len(g) == 0:
                    continue
                rows.append({
                    "model": m, "scenario": s, "decade": str(d),
                    "freq": (g["scPDSI"] <= -2).mean() * 100,
                })
    return pd.DataFrame(rows)

freq_f = decadal_freq(df_frozen, MODELS, SCENARIOS)
baseline_obs = 28.7  # ensemble historical mean fallback (a remplacer par 22.7 observed)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

DECADES = ["1990-2014", "2015-2024", "2025-2034", "2035-2040"]
X = np.arange(len(DECADES))
width = 0.25

for ax, s in zip(axes, SCENARIOS):
    for k, m in enumerate(MODELS):
        sub = freq_f[(freq_f["model"] == m) & (freq_f["scenario"] == s)]
        vals = [sub[sub["decade"] == d]["freq"].iloc[0] if (sub["decade"] == d).any()
                else 0 for d in DECADES]
        ax.bar(X + (k - 1) * width, vals, width=width,
               color=COLORS[m], label=m, edgecolor="white", lw=0.5)

    ax.axhline(baseline_obs, color="black", lw=0.7, ls="--",
               label=f"Baseline historique = {baseline_obs:.1f} %")
    ax.set_xticks(X)
    ax.set_xticklabels(DECADES, rotation=15)
    ax.set_title(SCEN_PRETTY[s], fontsize=11, fontweight="bold")
    if s == "ssp2_4_5":
        ax.set_ylabel("Fraction des mois (%)\nscPDSI <= -2")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="upper left")

fig.suptitle("Frequence decennale des mois en secheresse severe "
             "(calibration figee 1990-2014)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "fig_03_decadal_severe_freq")
plt.close(fig)


# =====================================================================
# Figure 4 -- Climate signal A.5 : forced vs AR(1) naive vs baseline
# =====================================================================
print("\n[fig 4] climate signal A.5")

# Recalcule rapide des chiffres (deja dans la sortie patch A.5)
hist_mean = (df_frozen[df_frozen["period"] == "historical"]
             .groupby("date")["scPDSI"].mean())
baseline_freq = (hist_mean <= -2).mean() * 100

# AR(1) Monte-Carlo (identique au patch A.5)
hist_vals = hist_mean.values
phi = np.corrcoef(hist_vals[:-1], hist_vals[1:])[0, 1]
mu = hist_vals.mean()
resid = hist_vals[1:] - mu - phi * (hist_vals[:-1] - mu)
sigma = resid.std(ddof=1)
np.random.seed(42)
N_SIM, N_STEPS = 1000, (2040 - 2015 + 1) * 12
sim = np.zeros((N_SIM, N_STEPS))
sim[:, 0] = hist_vals[-1]
for t in range(1, N_STEPS):
    sim[:, t] = mu + phi * (sim[:, t-1] - mu) + np.random.normal(0, sigma, N_SIM)
ar1_freqs = (sim[:, -72:] <= -2).mean(axis=1)
ar1_mean = ar1_freqs.mean() * 100
ar1_p05  = np.quantile(ar1_freqs, 0.05) * 100
ar1_p95  = np.quantile(ar1_freqs, 0.95) * 100

# Forced ensemble
forced = {}
for s in SCENARIOS:
    per_model = [
        (df_frozen[(df_frozen["model"] == m) &
                   (df_frozen["scenario"] == s) &
                   (df_frozen["date"] >= "2035-01-01") &
                   (df_frozen["date"] <= "2040-12-01")]["scPDSI"] <= -2).mean()
        for m in MODELS
    ]
    forced[s] = np.mean(per_model) * 100

fig, ax = plt.subplots(figsize=(8, 5))
labels = ["Baseline\n1990-2014", "AR(1) naif\n2035-2040", "Forced\nSSP2-4.5", "Forced\nSSP5-8.5"]
vals = [baseline_freq, ar1_mean, forced["ssp2_4_5"], forced["ssp5_8_5"]]
colors = ["#888888", "#FFAA33", "#1F78B4", "#E31A1C"]
errs = [[0, ar1_mean - ar1_p05, 0, 0], [0, ar1_p95 - ar1_mean, 0, 0]]

bars = ax.bar(labels, vals, color=colors, edgecolor="white", lw=0.7)
ax.errorbar([1], [ar1_mean], yerr=[[ar1_mean - ar1_p05], [ar1_p95 - ar1_mean]],
            fmt="none", ecolor="black", capsize=4, lw=0.8)

for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f} %",
            ha="center", fontsize=10, fontweight="bold")

# Annotations climate signal
ax.annotate("", xy=(2, forced["ssp2_4_5"]), xytext=(1, ar1_mean),
            arrowprops=dict(arrowstyle="->", color="#1F78B4", lw=1.2))
ax.text(1.5, (ar1_mean + forced["ssp2_4_5"]) / 2 + 4,
        f"+{forced['ssp2_4_5'] - ar1_mean:.1f} pp",
        color="#1F78B4", fontweight="bold", ha="center", fontsize=10)

ax.annotate("", xy=(3, forced["ssp5_8_5"]), xytext=(1, ar1_mean),
            arrowprops=dict(arrowstyle="->", color="#E31A1C", lw=1.2))
ax.text(2.5, (ar1_mean + forced["ssp5_8_5"]) / 2 + 4,
        f"+{forced['ssp5_8_5'] - ar1_mean:.1f} pp",
        color="#E31A1C", fontweight="bold", ha="center", fontsize=10)

ax.set_ylabel("Fraction des mois (%)  --  scPDSI <= -2")
ax.set_title("Signal climatique : ensemble force vs AR(1) naif "
             "(fenetre 2035-2040)", fontsize=11, fontweight="bold")
ax.set_ylim(0, max(vals) * 1.25)
fig.tight_layout()
save(fig, "fig_04_climate_signal_A5")
plt.close(fig)


# =====================================================================
# Figure 5 -- Drought class shift (distribution des classes Palmer)
# =====================================================================
print("\n[fig 5] drought class shift")

CLASS_ORDER = ["Extreme drought", "Severe drought", "Moderate drought",
               "Mild drought", "Near normal", "Mild wet",
               "Moderate wet", "Severe wet", "Extreme wet"]
CLASS_COLORS = ["#67000D", "#A50F15", "#CB181D", "#EF3B2C",
                "#FCBBA1", "#9ECAE1", "#4292C6", "#08519C", "#08306B"]

def class_distribution(df_, start, end):
    sub = df_[(df_["date"] >= start) & (df_["date"] <= end)]
    if len(sub) == 0:
        return {c: 0 for c in CLASS_ORDER}
    counts = sub["drought_class"].value_counts(normalize=True) * 100
    return {c: counts.get(c, 0) for c in CLASS_ORDER}

dists = {
    "Historique\n1990-2014": class_distribution(
        df_frozen[df_frozen["period"] == "historical"], "1990-01-01", "2014-12-01"),
    "SSP2-4.5\n2035-2040": class_distribution(
        df_frozen[df_frozen["scenario"] == "ssp2_4_5"], "2035-01-01", "2040-12-01"),
    "SSP5-8.5\n2035-2040": class_distribution(
        df_frozen[df_frozen["scenario"] == "ssp5_8_5"], "2035-01-01", "2040-12-01"),
}

fig, ax = plt.subplots(figsize=(10, 5))

bottom = np.zeros(len(dists))
labels = list(dists.keys())
x = np.arange(len(labels))

for c, col in zip(CLASS_ORDER, CLASS_COLORS):
    vals = [dists[lab][c] for lab in labels]
    bars = ax.bar(x, vals, bottom=bottom, color=col, edgecolor="white",
                  lw=0.5, label=c)
    # Etiquette si la classe > 5%
    for bar, v in zip(bars, vals):
        if v >= 5:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_y() + v/2, f"{v:.0f}%",
                    ha="center", va="center", fontsize=8,
                    color="white" if c in CLASS_ORDER[:3] + CLASS_ORDER[-2:] else "black",
                    fontweight="bold")
    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Fraction des mois (%)")
ax.set_title("Decalage de la distribution des classes Palmer "
             "(calibration figee 1990-2014)",
             fontsize=11, fontweight="bold")
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8,
          title="Classe Palmer")
ax.set_ylim(0, 100)
ax.set_yticks([0, 25, 50, 75, 100])

fig.tight_layout()
save(fig, "fig_05_drought_class_shift")
plt.close(fig)


print(f"\n[done] 5 figures dans : {OUT_DIR}")
print("       .pdf pour LaTeX  +  .png pour previsualisation")
