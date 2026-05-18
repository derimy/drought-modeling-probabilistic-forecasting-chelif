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
# R15_scpdsi_frozen_calibration.py
#
# Role: frozen-historical-calibration sensitivity branch for scPDSI.
# Methodological comparison against R12 (the continuous-calibration
# reference run). The thesis Chapter 5 headline rests on R12; this
# script reports the calibration artefact differential as a robustness check.
# ---------------------------------------------------------------------
# scPDSI projection avec calibration FROZEN sur 1990-2014.
#
# Why Python rather than R:
#   Le package R `scPDSI` n'expose pas la separation calibration/calcul :
#   passer start=1990, end=2014 sur une serie 612 mois renvoie seulement
#   300 valeurs.  Le package Python `climate_indices` (NIDIS/NOAA, Wells
#   2004) expose explicitement calibration_start_year /
#   calibration_end_year separes de data_start_year, et c'est
#   l'implementation standard pour les analyses CMIP6 (Stagge et al. 2017,
#   Hobbins et al. 2016).
#
# Inputs (same as R12, the continuous-calibration reference run):
#   02_processed/scpdsi_inputs/projection/{model_key}__{scenario}.csv
#   colonnes : date, ppt_mm, pet_mm
#
# Outputs :
#   02_processed/drought_indices/projection/scpdsi_projection_frozen_long.csv
#   02_processed/drought_indices/projection/scpdsi_<model_key>__<scenario>_frozen.csv (x6)
#   02_processed/drought_indices/projection/R15_run_log.txt
# =====================================================================

from pathlib import Path
import os
from datetime import datetime
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# 0. Dependances -- climate_indices v2.x (Wells 2004 self-calibrating PDSI)
# ---------------------------------------------------------------------
# v2.4.0 expose : palmer.pdsi(precips, pet, awc, data_start_year,
#                             calibration_year_initial, calibration_year_final,
#                             fitting_params=None)
# Renvoie (PDSI, PHDI, PMDI, ZINDEX, fitting_params).  Self-calibration
# activee par defaut (fitting_params=None).
try:
    from climate_indices import palmer
    _PDSI_FUNC = palmer.pdsi
    _API_PATH = "climate_indices.palmer.pdsi"
except (ImportError, AttributeError) as e:
    sys.stderr.write(
        f"ERROR: could not import climate_indices.palmer.pdsi\n"
        f"  -> {type(e).__name__}: {e}\n"
        f"Check the install: pip show climate-indices  (>= 2.0 expected)\n"
    )
    sys.exit(1)

# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------
LEGACY_ROOT     = Path(os.environ.get("CHELIF_PROJECT_ROOT", Path.cwd()))
STRUCTURED_ROOT = LEGACY_ROOT / "_structured_project"

IN_DIR  = STRUCTURED_ROOT / "02_processed" / "scpdsi_inputs" / "projection"
OUT_DIR = STRUCTURED_ROOT / "02_processed" / "drought_indices" / "projection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AWC          = 120                 # mm
DATA_START   = 1990
DATA_END     = 2040
CAL_START    = 1990
CAL_END      = 2014                # <-- FROZEN

JOBS = [
    ("ACCESS-CM2", "access_cm2", "ssp2_4_5"),
    ("ACCESS-CM2", "access_cm2", "ssp5_8_5"),
    ("CMCC-ESM2",  "cmcc_esm2",  "ssp2_4_5"),
    ("CMCC-ESM2",  "cmcc_esm2",  "ssp5_8_5"),
    ("GFDL-ESM4",  "gfdl_esm4",  "ssp2_4_5"),
    ("GFDL-ESM4",  "gfdl_esm4",  "ssp5_8_5"),
]

OUTPUT_LONG = OUT_DIR / "scpdsi_projection_frozen_long.csv"
LOG_FILE    = OUT_DIR / "R15_run_log.txt"

# ---------------------------------------------------------------------
# 2. Logging
# ---------------------------------------------------------------------
log_buf = []

def log(msg: str):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    log_buf.append(line)
    print(line, flush=True)

log(f"=== R15_scpdsi_frozen_calibration.py START ===")
log(f"climate_indices entry point : {_API_PATH}")
log(f"AWC={AWC} mm  data={DATA_START}-{DATA_END}  calibration=FROZEN {CAL_START}-{CAL_END}")
log(f"IN_DIR  = {IN_DIR}")
log(f"OUT_DIR = {OUT_DIR}")

# ---------------------------------------------------------------------
# 3. Palmer drought class (same thresholds as R12)
# ---------------------------------------------------------------------
PALMER_BREAKS = [-np.inf, -4, -3, -2, -1, 1, 2, 3, 4, np.inf]
PALMER_LABELS = ["Extreme drought", "Severe drought", "Moderate drought",
                 "Mild drought", "Near normal",
                 "Mild wet", "Moderate wet", "Severe wet", "Extreme wet"]

def palmer_class(x):
    return pd.cut(x, bins=PALMER_BREAKS, labels=PALMER_LABELS, right=False)


# ---------------------------------------------------------------------
# 4. Core scPDSI computation with frozen calibration
# ---------------------------------------------------------------------
def call_pdsi_frozen(ppt: np.ndarray, pet: np.ndarray) -> np.ndarray:
    """Appel a climate_indices.palmer.pdsi avec calibration ancree 1990-2014.

    Signature v2.4.0 :
        pdsi(precips, pet, awc, data_start_year,
             calibration_year_initial, calibration_year_final,
             fitting_params=None) -> (PDSI, PHDI, PMDI, ZINDEX, fit_params)
    """
    result = _PDSI_FUNC(
        precips                  = np.asarray(ppt, dtype=np.float64),
        pet                      = np.asarray(pet, dtype=np.float64),
        awc                      = float(AWC),
        data_start_year          = DATA_START,
        calibration_year_initial = CAL_START,
        calibration_year_final   = CAL_END,
        fitting_params           = None,    # None => self-calibration enabled
    )
    # 5-tuple : (PDSI, PHDI, PMDI, ZINDEX, fit_params)
    if isinstance(result, tuple):
        pdsi = np.asarray(result[0], dtype=np.float64)
    else:
        pdsi = np.asarray(result, dtype=np.float64)

    n_expected = (DATA_END - DATA_START + 1) * 12
    if len(pdsi) != n_expected:
        raise RuntimeError(
            f"climate_indices.palmer.pdsi returned {len(pdsi)} values, "
            f"expected {n_expected}"
        )
    return pdsi


def compute_one_job(model: str, model_key: str, scenario: str) -> pd.DataFrame:
    log(f"--- {model} / {scenario} ---")
    in_file = IN_DIR / f"{model_key}__{scenario}.csv"
    if not in_file.exists():
        raise FileNotFoundError(f"Input missing: {in_file}")

    df = pd.read_csv(in_file, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    n_expected = (DATA_END - DATA_START + 1) * 12
    if len(df) != n_expected:
        raise RuntimeError(
            f"[{model}/{scenario}] expected {n_expected} months, got {len(df)}")

    if df["ppt_mm"].isna().any() or df["pet_mm"].isna().any():
        raise RuntimeError(f"[{model}/{scenario}] NA in ppt/pet")

    log(f"  series length = {len(df)} months ({df['date'].min().date()} -> {df['date'].max().date()})")

    pdsi = call_pdsi_frozen(df["ppt_mm"].values, df["pet_mm"].values)

    out = pd.DataFrame({
        "model":              model,
        "scenario":           scenario,
        "date":               df["date"].values,
        "scPDSI":             pdsi,
        "drought_class":      palmer_class(pdsi),
        "period":             np.where(df["date"] <= "2014-12-01",
                                       "historical", "projection"),
        "AWC_mm":             AWC,
        "calibration_method": "frozen_1990_2014",
    })

    log(f"  scPDSI range : [{pdsi.min():.2f}, {pdsi.max():.2f}]   mean = {pdsi.mean():.2f}")

    v_dec = out.loc[out["date"] == "2014-12-01", "scPDSI"].iloc[0]
    v_jan = out.loc[out["date"] == "2015-01-01", "scPDSI"].iloc[0]
    log(f"  junction : Dec2014={v_dec:.4f}  Jan2015={v_jan:.4f}  |gap|={abs(v_jan-v_dec):.4f}")

    return out


# ---------------------------------------------------------------------
# 5. Run all 6 jobs
# ---------------------------------------------------------------------
results = []
for (model, model_key, scenario) in JOBS:
    try:
        out = compute_one_job(model, model_key, scenario)
    except Exception as e:
        log(f"  !! FAILED : {e}")
        raise
    results.append(out)

    per_csv = OUT_DIR / f"scpdsi_{model_key}__{scenario}_frozen.csv"
    out.to_csv(per_csv, index=False)
    log(f"  wrote {per_csv.name}")

scpdsi_long = pd.concat(results, ignore_index=True)
scpdsi_long.to_csv(OUTPUT_LONG, index=False)
log(f"Wrote combined long table : {OUTPUT_LONG.name}  ({len(scpdsi_long)} rows)")

# ---------------------------------------------------------------------
# 6. VERIFICATIONS
# ---------------------------------------------------------------------
log("=" * 30 + " VERIFICATIONS " + "=" * 30)

# V1 : scPDSI(2014-12) doit etre identique sur 6 jobs
v1 = (scpdsi_long
      .loc[scpdsi_long["date"] == "2014-12-01", ["model","scenario","scPDSI"]]
      .sort_values(["model","scenario"]))
log("V1  scPDSI(2014-12) per job :")
for _, r in v1.iterrows():
    log(f"      {r['model']:<12} {r['scenario']:<10}  {r['scPDSI']:.6f}")
v1_spread = float(v1["scPDSI"].max() - v1["scPDSI"].min())
v1_pass = v1_spread < 1e-6
log(f"V1  spread = {v1_spread:.2e}   tol = 1e-6   {'PASS' if v1_pass else 'FAIL'}")

# V2 : continuite |gap| 2014-12 -> 2015-01 < 0.3
v2_rows = []
for (m, s), g in scpdsi_long.groupby(["model","scenario"]):
    a = g.loc[g["date"] == "2014-12-01", "scPDSI"].iloc[0]
    b = g.loc[g["date"] == "2015-01-01", "scPDSI"].iloc[0]
    v2_rows.append((m, s, abs(b - a)))
log("V2  continuity gaps :")
for (m, s, gap) in v2_rows:
    log(f"      {m:<12} {s:<10}  |gap| = {gap:.4f}")
v2_pass = all(g < 0.3 for (_, _, g) in v2_rows)
v2_max = max(g for (_, _, g) in v2_rows)
log(f"V2  max gap = {v2_max:.4f}   threshold = 0.3   {'PASS' if v2_pass else 'FAIL'}")

# V3 : 6 x 612 = 3672 lignes
v3_expected = len(JOBS) * (DATA_END - DATA_START + 1) * 12
v3_pass = len(scpdsi_long) == v3_expected
log(f"V3  rows = {len(scpdsi_long)} / expected {v3_expected}   {'PASS' if v3_pass else 'FAIL'}")

# V4 : sanity bracket CMCC ssp5_8_5 2035-2040 freq(<= -2) in [25, 40] %
late = scpdsi_long[(scpdsi_long["model"] == "CMCC-ESM2") &
                   (scpdsi_long["scenario"] == "ssp5_8_5") &
                   (scpdsi_long["date"] >= "2035-01-01") &
                   (scpdsi_long["date"] <= "2040-12-01")]
freq_late = float((late["scPDSI"] <= -2).mean()) if len(late) > 0 else float("nan")
v4_ok = (not np.isnan(freq_late)) and (0.25 <= freq_late <= 0.40)
log(f"V4  CMCC-ESM2/ssp5_8_5/2035-2040 freq(<= -2) = {freq_late*100:.1f} %   "
    f"bracket [25, 40] %   {'PASS' if v4_ok else 'CHECK'}")

log("=" * 30 + " SUMMARY " + "=" * 30)
log(f"  V1 calibration frozen      : {'PASS' if v1_pass else 'FAIL'}")
log(f"  V2 continuity gap < 0.3    : {'PASS' if v2_pass else 'FAIL'}")
log(f"  V3 row count == {v3_expected}     : {'PASS' if v3_pass else 'FAIL'}")
log(f"  V4 sanity bracket [25,40]% : {'PASS' if v4_ok else 'CHECK'}")

LOG_FILE.write_text("\n".join(log_buf), encoding="utf-8")
print(f"\nLog written to : {LOG_FILE}")
print(f"Combined CSV   : {OUTPUT_LONG}")

if not v1_pass:
    sys.exit("V1 FAIL -- calibration not properly frozen.  See log.")
