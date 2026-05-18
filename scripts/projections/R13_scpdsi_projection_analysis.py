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

"""

#!/usr/bin/env python3
# =====================================================================
# R13_scpdsi_projection_analysis.py
# ---------------------------------------------------------------------
# Expected input schema: model, scenario, date, scPDSI,
#                            drought_class, period, AWC_mm
# Valeurs scenario : ssp2_4_5, ssp5_8_5
#
#   --source continuous   ->  02_processed/drought_indices/projection/
#                             scpdsi_projection_long.csv
#   --source frozen       ->  02_processed/drought_indices/projection/
#                             scpdsi_projection_frozen_long.csv
#
# Output: 02_processed/drought_indices/projection/{source}/
# =====================================================================

import argparse
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LEGACY_ROOT     = Path(os.environ.get("CHELIF_PROJECT_ROOT", Path.cwd()))
STRUCTURED_ROOT = LEGACY_ROOT / "_structured_project"

PROJ_DIR        = STRUCTURED_ROOT / "02_processed" / "drought_indices" / "projection"
HIST_GLOB_PATTERNS = [
    "**/scpdsi_chelif_historical.csv",
    "**/scpdsi_historical*.csv",
    "**/scpdsi*chelif*historical*.csv",
    "**/scpdsi*/**/scpdsi*.csv",
]

MODELS    = ["ACCESS-CM2", "CMCC-ESM2", "GFDL-ESM4"]
SCENARIOS = ["ssp2_4_5", "ssp5_8_5"]

SEVERE_DROUGHT   = -2.0
MODERATE_DROUGHT = -1.0

BASELINE_START = "1990-01-01"
BASELINE_END   = "2014-12-31"


def load_source(source: str) -> pd.DataFrame:
    if source == "continuous":
        path = PROJ_DIR / "scpdsi_projection_long.csv"
    elif source == "frozen":
        path = PROJ_DIR / "scpdsi_projection_frozen_long.csv"
    else:
        raise ValueError(f"source must be continuous|frozen, got {source!r}")

    if not path.exists():
        raise FileNotFoundError(
            f"Source file not found: {path}\n"
            f"  -> for 'frozen', run R15 (frozen calibration) first"
        )

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["model", "scenario", "date"]).reset_index(drop=True)

    # Safety check: drop any unexpected scenario
    df = df[df["scenario"].isin(SCENARIOS) | (df["scenario"].isna())].copy()
    print(f"[load_source] {source}: {len(df)} lignes depuis {path.name}")
    return df


def load_historical(df_proj_fallback: pd.DataFrame = None) -> pd.DataFrame:
    """Load the observed historical scPDSI series.

    1) Glob recursif sur _structured_project/ pour des fichiers candidats
    2) Si rien trouve : fallback sur la moyenne des modeles, periode 'historical'
       extraite du long-format projection (CMIP6 historique bias-corrige).
       Note : ce fallback n'est PAS strictement la baseline observee,
       mais une approximation utilisable en attendant.
    """
    candidates = []
    for pattern in HIST_GLOB_PATTERNS:
        candidates.extend(STRUCTURED_ROOT.glob(pattern))
    # filtrer les doublons et ceux dans 'projection' (eviter de reboucler)
    seen = set()
    uniq = []
    for c in candidates:
        rp = c.resolve()
        if rp in seen: continue
        seen.add(rp)
        if "projection" in c.parts: continue
        uniq.append(c)
    print(f"[load_historical] {len(uniq)} candidat(s) trouve(s) :")
    for c in uniq:
        print(f"   - {c}")

    for c in uniq:
        try:
            df = pd.read_csv(c, parse_dates=["date"])
            if "scPDSI" not in df.columns:
                continue
            df = df.sort_values("date").reset_index(drop=True)
            df = df[(df["date"] >= BASELINE_START) &
                    (df["date"] <= BASELINE_END)].copy()
            if len(df) >= 200:
                print(f"[load_historical] OK {len(df)} lignes depuis {c.name}")
                return df
        except Exception as e:
            print(f"   skip {c.name} : {e}")

    # Fallback : moyenne ensemble historique du long projection
    if df_proj_fallback is not None and "period" in df_proj_fallback.columns:
        hist = df_proj_fallback[df_proj_fallback["period"] == "historical"].copy()
        if len(hist) > 0:
            hist_avg = (hist.groupby("date")["scPDSI"].mean()
                            .reset_index())
            hist_avg = hist_avg[(hist_avg["date"] >= BASELINE_START) &
                                (hist_avg["date"] <= BASELINE_END)]
            print(f"[load_historical] FALLBACK ensemble historical mean "
                  f"({len(hist_avg)} mois) depuis long-format projection")
            return hist_avg

    raise FileNotFoundError(
        "Aucun CSV scPDSI historique trouve par glob, et pas de fallback "
        "usable. Check that the scPDSI compute step (notebook E.7) has run and that its output "
        "is in _structured_project/."
    )


def ensure_out(source: str) -> Path:
    out = PROJ_DIR / source
    (out / "figures").mkdir(parents=True, exist_ok=True)
    return out


def build_hist_vs_proj(df_proj, df_hist, out_dir):
    hist = df_hist[["date", "scPDSI"]].assign(
        model="HISTORICAL", scenario="historical", phase="historical"
    )
    proj = df_proj.copy()
    # Si 'period' existe, l'utiliser ; sinon fallback sur date
    if "period" in proj.columns:
        proj["phase"] = proj["period"]
    else:
        proj["phase"] = np.where(proj["date"] <= "2014-12-01", "historical", "projection")

    combined = pd.concat([
        hist[["date", "scPDSI", "model", "scenario", "phase"]],
        proj[["date", "scPDSI", "model", "scenario", "phase"]]
    ], ignore_index=True)

    out = out_dir / "historical_vs_projection_scpdsi.csv"
    combined.to_csv(out, index=False)
    print(f"[bloc1] wrote {out.name}  ({len(combined)} rows)")


def _months_between(a, b):
    """Inclusive month count between two pd.Timestamp."""
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    return (b.year - a.year) * 12 + (b.month - a.month) + 1


def drought_events(df_proj, out_dir):
    rows = []
    for (model, scen), g in df_proj.groupby(["model", "scenario"]):
        g = g.sort_values("date").reset_index(drop=True)
        in_event = False
        start = peak = None
        peak_val = None
        for _, r in g.iterrows():
            below = r["scPDSI"] <= SEVERE_DROUGHT
            if below and not in_event:
                in_event = True
                start = r["date"]
                peak, peak_val = r["date"], r["scPDSI"]
            elif below and in_event:
                if r["scPDSI"] < peak_val:
                    peak, peak_val = r["date"], r["scPDSI"]
            elif not below and in_event:
                end = r["date"]
                rows.append({
                    "model": model, "scenario": scen,
                    "start": start, "end": end,
                    "duration_months": _months_between(start, end),
                    "peak_date": peak, "peak_scPDSI": peak_val,
                })
                in_event = False
                start = peak = None
                peak_val = None
        if in_event:
            rows.append({
                "model": model, "scenario": scen,
                "start": start, "end": g["date"].iloc[-1],
                "duration_months": _months_between(start, g["date"].iloc[-1]),
                "peak_date": peak, "peak_scPDSI": peak_val,
            })

    events = pd.DataFrame(rows)
    out = out_dir / "drought_events_scpdsi.csv"
    events.to_csv(out, index=False)
    print(f"[bloc2] wrote {out.name}  ({len(events)} events)")


def decadal_summary(df_proj, df_hist, out_dir):
    def dec(d): return f"{(d.year // 10) * 10}s"

    proj = df_proj.copy()
    proj["decade"] = proj["date"].apply(dec)
    agg_proj = proj.groupby(["model", "scenario", "decade"]).agg(
        mean_scPDSI=("scPDSI", "mean"),
        freq_severe=("scPDSI", lambda s: (s <= SEVERE_DROUGHT).mean()),
        freq_moderate=("scPDSI", lambda s: (s <= MODERATE_DROUGHT).mean()),
        n_months=("scPDSI", "count"),
    ).reset_index()

    hist = df_hist.copy()
    hist["decade"] = hist["date"].apply(dec)
    agg_hist = hist.groupby("decade").agg(
        mean_scPDSI=("scPDSI", "mean"),
        freq_severe=("scPDSI", lambda s: (s <= SEVERE_DROUGHT).mean()),
        freq_moderate=("scPDSI", lambda s: (s <= MODERATE_DROUGHT).mean()),
        n_months=("scPDSI", "count"),
    ).reset_index()
    agg_hist["model"] = "HISTORICAL"
    agg_hist["scenario"] = "historical"

    full = pd.concat([agg_proj, agg_hist], ignore_index=True, sort=False)
    out = out_dir / "decadal_summary_with_baseline.csv"
    full.to_csv(out, index=False)
    print(f"[bloc3] wrote {out.name}  ({len(full)} rows)")


def brunner_ensemble(df_proj, out_dir):
    pivot = (df_proj
             .pivot_table(index=["date", "scenario"], columns="model",
                          values="scPDSI", aggfunc="first")
             .reset_index())
    avail = [m for m in MODELS if m in pivot.columns]
    pivot["ensemble_median"] = pivot[avail].median(axis=1)
    pivot["ensemble_p10"]    = pivot[avail].quantile(0.10, axis=1)
    pivot["ensemble_p90"]    = pivot[avail].quantile(0.90, axis=1)

    out = out_dir / "brunner_ensemble_scpdsi.csv"
    pivot.to_csv(out, index=False)
    print(f"[bloc4] wrote {out.name}  ({len(pivot)} rows)")


def decision_summary(df_proj, df_hist, out_dir):
    base_freq = (df_hist["scPDSI"] <= SEVERE_DROUGHT).mean()

    rows = []
    for scen in SCENARIOS:
        sub = df_proj[(df_proj["scenario"] == scen) &
                      (df_proj["date"] >= "2035-01-01") &
                      (df_proj["date"] <= "2040-12-01")]
        per_model_freq = [
            (sub[sub["model"] == m]["scPDSI"] <= SEVERE_DROUGHT).mean()
            for m in MODELS
        ]
        forced_freq = float(np.mean(per_model_freq))
        rows.append({
            "scenario": scen,
            "baseline_1990_2014_freq_severe": base_freq,
            "forced_freq_severe_2035_2040": forced_freq,
            "climate_signal_pp": (forced_freq - base_freq) * 100,
            "per_model_freqs": ";".join(f"{m}={f:.3f}" for m, f in zip(MODELS, per_model_freq)),
        })

    pd.DataFrame(rows).to_csv(out_dir / "scpdsi_decision_summary.csv", index=False)
    print(f"[bloc5] baseline = {base_freq:.1%}")


def figures(df_proj, df_hist, out_dir):
    fig_dir = out_dir / "figures"
    for scen in SCENARIOS:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(df_hist["date"], df_hist["scPDSI"],
                color="black", lw=0.9, label="Observed 1990-2014")
        for m in MODELS:
            sub = df_proj[(df_proj["model"] == m) & (df_proj["scenario"] == scen)]
            ax.plot(sub["date"], sub["scPDSI"], lw=0.7, alpha=0.85, label=m)
        ax.axhline(-2, color="red", lw=0.6, linestyle="--", alpha=0.5)
        ax.axhline(0,  color="grey", lw=0.4)
        ax.set_title(f"scPDSI -- {scen.upper()} ({out_dir.name})")
        ax.set_ylabel("scPDSI")
        ax.legend(fontsize=8, ncol=2, loc="lower left")
        fig.tight_layout()
        fig.savefig(fig_dir / f"timeseries_{scen}.png", dpi=150)
        plt.close(fig)
    print(f"[bloc6] figures dans {fig_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["continuous", "frozen"], required=True)
    args = p.parse_args()

    out_dir = ensure_out(args.source)
    df_proj = load_source(args.source)
    df_hist = load_historical(df_proj_fallback=df_proj)

    build_hist_vs_proj(df_proj, df_hist, out_dir)
    drought_events(df_proj, out_dir)
    decadal_summary(df_proj, df_hist, out_dir)
    brunner_ensemble(df_proj, out_dir)
    decision_summary(df_proj, df_hist, out_dir)
    figures(df_proj, df_hist, out_dir)

    print(f"\n[done] livrables dans : {out_dir}")
    print(f"       source = {args.source}")


if __name__ == "__main__":
    main()
