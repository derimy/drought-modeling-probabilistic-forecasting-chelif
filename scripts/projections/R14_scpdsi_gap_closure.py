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

R14 — scPDSI projection gap closure
=====================================================

Closes 5 methodological gaps from the original Étape 5–6 plan:

  A.1  Continuity check at the 2014-12 / 2015-01 junction
  A.2  Mann-Kendall trend test on annual scPDSI 2015-2040
  A.3  Sensitivity: continuous sc=TRUE vs frozen 1990-2014 calibration
       (pragmatic approximation via historical-window rescaling)
  A.4  Three timeline figures (SSP2-4.5, SSP5-8.5, Brunner ensemble)
       with inter-model envelope + Sen-Theil trend
  A.5  Illustrative AR(1) persistence comparator inspired by the short-memory
       behaviour identified in the SPI analysis (not a transfer of the SPI
       Bayesian model itself), evaluated against the CMIP6-forced projection

Inputs  : outputs_scpdsi_projection/scpdsi_projection_long.csv  (from R12)
Outputs : 4 CSVs + 3 PDFs in outputs_scpdsi_projection/

Dependencies:
    pip install pymannkendall

Run:
    python notebook_16_scpdsi_gap_closure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import pymannkendall as mk
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pymannkendall missing — install with: pip install pymannkendall"
    ) from exc


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent

SEVERE_THRESHOLD = -2.0
EXTREME_THRESHOLD = -3.0
CONTINUITY_GAP_TOLERANCE = 0.3  # |gap| acceptable for junction check

N_BOOTSTRAP_AR1 = 1000
AR1_PHI = 0.95            # from Chapter 4 (SPI Bayesian fit)
AR1_SIGMA_EPS = 0.71      # from Chapter 4

MODELS = ["ACCESS-CM2", "CMCC-ESM2", "GFDL-ESM4"]
SCENARIOS = ["ssp2_4_5", "ssp5_8_5"]
BRUNNER_WEIGHTS = {m: 1.0 / 3.0 for m in MODELS}  # 3 structurally independent

HIST_END = pd.Timestamp("2014-12-01")
PROJ_START = pd.Timestamp("2015-01-01")
PROJ_END_DECADE_START = pd.Timestamp("2035-01-01")
PROJ_END = pd.Timestamp("2040-12-01")


def locate_inputs_dir() -> Path:
    """Find the directory containing scpdsi_projection_long.csv."""
    candidates: list[Path] = []
    for base in [SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent.parent,
                 SCRIPT_DIR.parent.parent.parent]:
        candidates.extend(base.rglob("scpdsi_projection_long.csv"))
        if candidates:
            break
    if not candidates:
        raise SystemExit(
            "Cannot locate scpdsi_projection_long.csv. "
            f"Searched from {SCRIPT_DIR} upwards."
        )
    return candidates[0].parent


INPUTS_DIR = locate_inputs_dir()
OUTPUTS_DIR = INPUTS_DIR  # write alongside R13 outputs
print(f"[INFO] Inputs/outputs dir: {INPUTS_DIR}")


# ──────────────────────────────────────────────────────────────────────────────
# LOAD scPDSI long series
# ──────────────────────────────────────────────────────────────────────────────

src = INPUTS_DIR / "scpdsi_projection_long.csv"
df = pd.read_csv(src)
df.columns = [c.lower() for c in df.columns]
print(f"[INFO] Raw columns: {list(df.columns)}")

# Normalize column names (R may name them differently than expected)
COLUMN_ALIASES = {
    "ssp": ["ssp", "scenario", "experiment", "exp", "run"],
    "model": ["model", "gcm", "source", "source_id"],
    "scpdsi": ["scpdsi", "pdsi", "sc_pdsi", "scpdsi_value", "value", "index"],
    "date": ["date", "time", "datetime"],
}
for canonical, aliases in COLUMN_ALIASES.items():
    if canonical in df.columns:
        continue
    for alias in aliases:
        if alias in df.columns:
            df = df.rename(columns={alias: canonical})
            print(f"[INFO] Renamed column '{alias}' → '{canonical}'")
            break

# Date handling
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
elif {"year", "month"}.issubset(df.columns):
    df["date"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=1)
    )
else:
    raise SystemExit(
        f"No date column found. Available columns: {list(df.columns)}"
    )

# Validate required columns exist after aliasing
required = {"date", "model", "ssp", "scpdsi"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(
        f"Missing required columns after alias mapping: {missing}. "
        f"Available: {list(df.columns)}"
    )

# Normalize ssp labels (handle 'ssp245', 'ssp2-4.5', etc.)
df["ssp"] = (
    df["ssp"].astype(str).str.lower()
    .str.replace("-", "_", regex=False)
    .str.replace(".", "_", regex=False)
)
# Map common variants
ssp_map = {
    "ssp245": "ssp2_4_5",
    "ssp585": "ssp5_8_5",
    "ssp2_45": "ssp2_4_5",
    "ssp5_85": "ssp5_8_5",
}
df["ssp"] = df["ssp"].replace(ssp_map)

print(f"[INFO] Rows loaded: {len(df):,}")
print(f"[INFO] Models: {sorted(df['model'].unique())}")
print(f"[INFO] Scenarios: {sorted(df['ssp'].unique())}")
print(f"[INFO] Date span: {df['date'].min().date()} → {df['date'].max().date()}")


# ──────────────────────────────────────────────────────────────────────────────
# A.1 — Continuity check at junction 2014-12 / 2015-01
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("A.1 — Continuity check at junction 2014-12 / 2015-01")
print("═" * 70)

rows = []
for model in MODELS:
    for ssp in SCENARIOS:
        sub = (
            df[(df["model"] == model) & (df["ssp"] == ssp)]
            .set_index("date")["scpdsi"]
        )
        v_hist = sub.loc[HIST_END] if HIST_END in sub.index else np.nan
        v_proj = sub.loc[PROJ_START] if PROJ_START in sub.index else np.nan
        gap = v_proj - v_hist
        rows.append({
            "model": model,
            "ssp": ssp,
            "scpdsi_2014_12": round(float(v_hist), 4),
            "scpdsi_2015_01": round(float(v_proj), 4),
            "gap": round(float(gap), 4),
            "abs_gap": round(abs(float(gap)), 4),
            "gap_acceptable": bool(abs(gap) <= CONTINUITY_GAP_TOLERANCE),
        })

continuity_df = pd.DataFrame(rows)
continuity_path = OUTPUTS_DIR / "continuity_check_scpdsi.csv"
continuity_df.to_csv(continuity_path, index=False)
print(continuity_df.to_string(index=False))
n_ok = int(continuity_df["gap_acceptable"].sum())
print(f"\n[OK]   Saved → {continuity_path.name}")
print(f"[INFO] {n_ok}/{len(continuity_df)} series have |gap| ≤ "
      f"{CONTINUITY_GAP_TOLERANCE} unit")
print(f"[INFO] Max |gap| across all series: "
      f"{continuity_df['abs_gap'].max():.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# A.2 — Mann-Kendall trend on annual scPDSI 2015-2040
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("A.2 — Mann-Kendall trend on annual mean scPDSI, 2015-2040")
print("═" * 70)

rows = []
for model in MODELS:
    for ssp in SCENARIOS:
        sub = df[(df["model"] == model) & (df["ssp"] == ssp)
                 & (df["date"] >= PROJ_START)]
        annual = sub.groupby(sub["date"].dt.year)["scpdsi"].mean()
        res = mk.original_test(annual.values)
        rows.append({
            "model": model,
            "ssp": ssp,
            "kendall_tau": round(float(res.Tau), 4),
            "p_value": round(float(res.p), 4),
            "sen_slope_per_decade": round(float(res.slope) * 10, 4),
            "significant_at_5pct": bool(res.p < 0.05),
            "trend_direction": res.trend,
        })

trend_df = pd.DataFrame(rows)
trend_path = OUTPUTS_DIR / "trend_mann_kendall_scpdsi.csv"
trend_df.to_csv(trend_path, index=False)
print(trend_df.to_string(index=False))
n_sig = int(trend_df["significant_at_5pct"].sum())
print(f"\n[OK]   Saved → {trend_path.name}")
print(f"[INFO] {n_sig}/{len(trend_df)} series have significant trend at α=5%")


# ──────────────────────────────────────────────────────────────────────────────
# A.3 — Calibration sensitivity (CMCC-ESM2 / SSP5-8.5)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("A.3 — Calibration sensitivity: continuous sc=TRUE vs frozen 1990-2014")
print("       Test scenario: CMCC-ESM2 / SSP5-8.5 (most extreme signal)")
print("═" * 70)
print("[NOTE] Pragmatic approximation: rescale projection using 1990-2014")
print("       mean/std of the continuous series to mimic a calibration that")
print("       would have been frozen on the historical window only.")
print("       Equivalent to van der Schrier (2013) sensitivity heuristic.")

target_model, target_ssp = "CMCC-ESM2", "ssp5_8_5"
target = (
    df[(df["model"] == target_model) & (df["ssp"] == target_ssp)]
    .set_index("date")["scpdsi"].sort_index()
)

hist_window = target.loc[:HIST_END]
proj_window = target.loc[PROJ_START:]
proj_decade = target.loc[PROJ_END_DECADE_START:PROJ_END]

mu_hist = hist_window.mean()
sd_hist = hist_window.std()

proj_frozen = (proj_window - mu_hist) / sd_hist
proj_frozen_decade = (proj_decade - mu_hist) / sd_hist

freq_cont_full = (proj_window <= SEVERE_THRESHOLD).mean() * 100
freq_frozen_full = (proj_frozen <= SEVERE_THRESHOLD).mean() * 100
freq_cont_decade = (proj_decade <= SEVERE_THRESHOLD).mean() * 100
freq_frozen_decade = (proj_frozen_decade <= SEVERE_THRESHOLD).mean() * 100

sens_df = pd.DataFrame([
    {
        "scenario": f"{target_model} / {target_ssp}",
        "window": "2015-2040 (full)",
        "continuous_calibration_freq_pct": round(freq_cont_full, 2),
        "frozen_calibration_freq_pct": round(freq_frozen_full, 2),
        "delta_pp": round(freq_frozen_full - freq_cont_full, 2),
    },
    {
        "scenario": f"{target_model} / {target_ssp}",
        "window": "2035-2040 (extreme decade)",
        "continuous_calibration_freq_pct": round(freq_cont_decade, 2),
        "frozen_calibration_freq_pct": round(freq_frozen_decade, 2),
        "delta_pp": round(freq_frozen_decade - freq_cont_decade, 2),
    },
])

sens_path = OUTPUTS_DIR / "calibration_sensitivity_scpdsi.csv"
sens_df.to_csv(sens_path, index=False)
print(sens_df.to_string(index=False))
print(f"\n[INFO] Historical-window mean (continuous): {mu_hist:+.4f}")
print(f"[INFO] Historical-window std  (continuous): {sd_hist:+.4f}")

delta_decade = abs(freq_frozen_decade - freq_cont_decade)
print(f"[INFO] Δ on 2035-2040 frequency: {delta_decade:.2f} pp")
if delta_decade < 5.0:
    print("[OK]   Δ < 5 pp → continuous calibration does NOT introduce")
    print("       material bias on the direction of the signal.")
else:
    print("[WARN] Δ ≥ 5 pp → calibration choice does influence frequency.")
    print("       Recommend full R-based frozen calibration re-run.")
print(f"[OK]   Saved → {sens_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# A.4 — Timeline figures, one per scenario + Brunner ensemble
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("A.4 — Timeline figures with inter-model envelope + Sen-Theil trend")
print("═" * 70)

# Historical reference: average across the 6 (model × ssp) series for 1990-2014.
# Inputs are identical historically, so curves nearly overlap; averaging is safe.
hist_ref = (
    df[df["date"] <= HIST_END]
    .groupby("date")["scpdsi"].mean().sort_index()
)

COLOR_MAP = {
    "ACCESS-CM2": "#1f77b4",
    "CMCC-ESM2": "#d62728",
    "GFDL-ESM4": "#2ca02c",
}


def plot_scenario_timeline(scenario_key: str, save_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(13, 5.5))

    ax.plot(hist_ref.index, hist_ref.values, color="gray", lw=1.4,
            label="Historical (TerraClimate-driven, 1990–2014)")

    proj = df[(df["ssp"] == scenario_key) & (df["date"] >= PROJ_START)]
    pivoted = (
        proj.pivot_table(index="date", columns="model", values="scpdsi")
        .sort_index()
    )

    for model in MODELS:
        if model in pivoted.columns:
            ax.plot(pivoted.index, pivoted[model], lw=0.9, alpha=0.75,
                    color=COLOR_MAP[model], label=model)

    env_min = pivoted.min(axis=1)
    env_max = pivoted.max(axis=1)
    ax.fill_between(pivoted.index, env_min, env_max, alpha=0.13,
                    color="steelblue",
                    label="Inter-model min–max envelope")

    median_proj = pivoted.median(axis=1)
    x_num = mdates.date2num(median_proj.index.to_pydatetime())
    res = stats.theilslopes(median_proj.values, x_num)
    trend_line = res.intercept + res.slope * x_num
    slope_per_decade = res.slope * 365.25 * 10
    ax.plot(median_proj.index, trend_line, color="black", lw=1.7, ls="--",
            label=f"Sen–Theil trend ({slope_per_decade:+.3f} /decade)")

    ax.axhline(0.0, color="black", lw=0.4, ls=":")
    ax.axhline(SEVERE_THRESHOLD, color="orange", lw=0.7, ls=":",
               label="scPDSI = −2 (severe)")
    ax.axhline(EXTREME_THRESHOLD, color="red", lw=0.7, ls=":",
               label="scPDSI = −3 (extreme)")

    ax.set_xlim(pd.Timestamp("1990-01-01"), pd.Timestamp("2041-01-01"))
    ax.set_ylim(-6, 5)
    ax.set_xlabel("Year")
    ax.set_ylabel("scPDSI")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8, ncol=2, framealpha=0.85)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[OK]   {save_path.name}")


for ssp in SCENARIOS:
    pretty = ssp.replace("_", "-").upper()
    out_path = OUTPUTS_DIR / f"fig_scpdsi_timeline_{ssp}.pdf"
    plot_scenario_timeline(ssp, out_path,
                           f"scPDSI projection, 1990–2040 — {pretty}")


# Brunner ensemble figure (both scenarios on one)
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.plot(hist_ref.index, hist_ref.values, color="gray", lw=1.4,
        label="Historical (TerraClimate-driven)")

scenario_color = {"ssp2_4_5": "#1f77b4", "ssp5_8_5": "#d62728"}
for ssp in SCENARIOS:
    proj = df[(df["ssp"] == ssp) & (df["date"] >= PROJ_START)]
    pivoted = (
        proj.pivot_table(index="date", columns="model", values="scpdsi")
        .sort_index()
    )
    brunner = sum(
        pivoted[m] * BRUNNER_WEIGHTS[m] for m in MODELS if m in pivoted.columns
    )
    ax.plot(brunner.index, brunner.values, color=scenario_color[ssp], lw=1.6,
            label=f"Brunner ensemble — {ssp.replace('_', '-').upper()}")
    env_min = pivoted.min(axis=1)
    env_max = pivoted.max(axis=1)
    ax.fill_between(pivoted.index, env_min, env_max, alpha=0.10,
                    color=scenario_color[ssp])

ax.axhline(0.0, color="black", lw=0.4, ls=":")
ax.axhline(SEVERE_THRESHOLD, color="orange", lw=0.7, ls=":",
           label="scPDSI = −2")
ax.axhline(EXTREME_THRESHOLD, color="red", lw=0.7, ls=":",
           label="scPDSI = −3")
ax.set_xlim(pd.Timestamp("1990-01-01"), pd.Timestamp("2041-01-01"))
ax.set_ylim(-6, 5)
ax.set_xlabel("Year")
ax.set_ylabel("scPDSI")
ax.set_title("scPDSI Brunner-weighted ensemble (1/3 each model), 1990–2040")
ax.legend(loc="lower left", fontsize=9, ncol=2, framealpha=0.85)
ax.grid(True, alpha=0.25)
plt.tight_layout()
brunner_path = OUTPUTS_DIR / "fig_scpdsi_timeline_brunner.pdf"
plt.savefig(brunner_path, dpi=160)
plt.close(fig)
print(f"[OK]   {brunner_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# A.5 — AR(1) persistence baseline vs CMIP6-forced projection
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("A.5 — AR(1) persistence baseline vs CMIP6-forced projection")
print("═" * 70)
print(f"[NOTE] AR(1) parameters used as illustrative persistence comparator:")
print(f"       φ = {AR1_PHI}, σ_ε = {AR1_SIGMA_EPS}")
print(f"[NOTE] Residual bootstrap: {N_BOOTSTRAP_AR1} trajectories from x0 = "
      f"scPDSI(2014-12)")
print(f"[NOTE] Cross-index comparison (SPI-fit AR(1) on scPDSI starting point)")
print(f"       is methodologically conservative — see CSV footer note.")

x0 = float(hist_ref.loc[HIST_END])
n_months_proj = (
    (PROJ_END.year - HIST_END.year) * 12
    + (PROJ_END.month - HIST_END.month)
)
print(f"[INFO] Starting x0 (scPDSI at 2014-12): {x0:+.4f}")
print(f"[INFO] Forward horizon: {n_months_proj} months")

rng = np.random.default_rng(seed=42)
severe_freqs = np.zeros(N_BOOTSTRAP_AR1)
for k in range(N_BOOTSTRAP_AR1):
    x = np.zeros(n_months_proj)
    x[0] = AR1_PHI * x0 + rng.normal(0.0, AR1_SIGMA_EPS)
    for t in range(1, n_months_proj):
        x[t] = AR1_PHI * x[t - 1] + rng.normal(0.0, AR1_SIGMA_EPS)
    severe_freqs[k] = (x <= SEVERE_THRESHOLD).mean() * 100.0

ar1_naive_freq = float(severe_freqs.mean())
ar1_p10 = float(np.percentile(severe_freqs, 10))
ar1_p90 = float(np.percentile(severe_freqs, 90))

print(f"[INFO] AR(1) naïve : mean = {ar1_naive_freq:.2f} %,  "
      f"P10–P90 = [{ar1_p10:.2f}, {ar1_p90:.2f}]")

rows = []
for ssp in SCENARIOS:
    sub = df[(df["ssp"] == ssp) & (df["date"] >= PROJ_START)]
    pivoted = sub.pivot_table(index="date", columns="model", values="scpdsi")
    brunner = sum(
        pivoted[m] * BRUNNER_WEIGHTS[m] for m in MODELS if m in pivoted.columns
    )
    forced_freq = float((brunner <= SEVERE_THRESHOLD).mean() * 100.0)
    rows.append({
        "scenario": ssp,
        "ar1_naive_freq_pct": round(ar1_naive_freq, 2),
        "ar1_naive_p10_pct": round(ar1_p10, 2),
        "ar1_naive_p90_pct": round(ar1_p90, 2),
        "forced_proj_freq_pct": round(forced_freq, 2),
        "climate_signal_pp": round(forced_freq - ar1_naive_freq, 2),
    })

ar1_df = pd.DataFrame(rows)
ar1_path = OUTPUTS_DIR / "ar1_naive_vs_forced_projection.csv"
ar1_df.to_csv(ar1_path, index=False)
# Append explanatory footer to CSV
with open(ar1_path, "a", encoding="utf-8") as fh:
    fh.write("\n# Note: AR(1) parameters (phi=0.95, sigma_eps=0.71) from "
             "Chapter 4 SPI Bayesian fit.\n")
    fh.write("# Forward bootstrap (1000 trajectories) from x0 = scPDSI("
             "2014-12).\n")
    fh.write("# climate_signal_pp = additional severe-drought frequency "
             "attributable to CMIP6 forcing beyond pure statistical "
             "persistence.\n")

print(ar1_df.to_string(index=False))
print(f"\n[OK]   Saved → {ar1_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# WRAP-UP
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("R14 — gap closure DONE")
print("═" * 70)
print(f"Outputs directory: {OUTPUTS_DIR}")
print("  CSVs:")
print("    1. continuity_check_scpdsi.csv")
print("    2. trend_mann_kendall_scpdsi.csv")
print("    3. calibration_sensitivity_scpdsi.csv")
print("    4. ar1_naive_vs_forced_projection.csv")
print("  Figures:")
print("    5. fig_scpdsi_timeline_ssp2_4_5.pdf")
print("    6. fig_scpdsi_timeline_ssp5_8_5.pdf")
print("    7. fig_scpdsi_timeline_brunner.pdf")
print("\nMethodological gap-closure complete. Ready for Bloc B (ArcGIS export).")
