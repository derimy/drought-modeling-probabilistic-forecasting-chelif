"""
Author        : Imene DERRAR
Supervisor    : Dr. Abdelillah OTMANE CHERIF
Institutions  : NHSM — National Higher School of Mathematics
                IHFR — Institut Hydrométéorologique de Formation et de Recherche
Academic year : 2025–2026
Project       : Final-year engineering thesis in Applied Mathematics —
                Drought modelling and probabilistic forecasting for the Chélif basin

--------------------------------------------------------------------------------

X4 — Drought-year frequency per station (per-model corrected)
========================================================================

Replaces the earlier delta in X3.  The X3 delta
compared the historical OBS drought-year count (one realisation) against
the Brunner-weighted ENSEMBLE-MEAN drought-year count (a smoothed average
of three model realisations).  Ensemble averaging artificially reduces
year-to-year variability, so the delta was systematically negative — an
partly influenced by ensemble smoothing effects rather than reflecting only the climate signal itself.

This notebook fixes the comparison by:

  1.  Counting drought years for each retained CMIP6 model INDIVIDUALLY
      (CESM2, EC-Earth3-CC, EC-Earth3-Veg-LR — each is one realisation
      with realistic year-to-year variability).
  2.  Per (station × scenario), reporting the mean, min and max
      drought-year count ACROSS the three models.
  3.  Computing the delta against the historical observed rate using the
      per-model MEAN.  This is the comparison practitioners and the IPCC
      AR6 use for projected drought frequencies.

Inputs
------
    01_data/external/master_station_metadata_final_plus_pluvio.csv
    04_outputs/R4_spi_projection/tables/spi_projection_long.csv

Outputs
-------
    04_outputs/X2_arcgis_export/chelif_drought_frequency_per_model.csv
    04_outputs/X2_arcgis_export/chelif_drought_frequency_per_model.xlsx
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 0. CONFIGURATION
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    accumulation_k: int = 6
    severe_threshold: float = -1.5

    hist_start: str = "1990-01"
    hist_end:   str = "2014-12"
    proj_start: str = "2015-01"
    proj_end:   str = "2040-12"

    hist_window_years: int = 25
    proj_window_years: int = 26

    retained_models: tuple = ("CESM2", "EC-Earth3-CC", "EC-Earth3-Veg-LR")
    scenarios:       tuple = ("ssp2_4_5", "ssp5_8_5")

    project_root: Path = Path(__file__).resolve().parents[3]
    meta_file:    Path = field(init=False)
    spi_file:     Path = field(init=False)
    out_dir:      Path = field(init=False)

    def __post_init__(self) -> None:
        self.meta_file = (self.project_root / "01_data" / "external"
                          / "master_station_metadata_final_plus_pluvio.csv")
        self.spi_file  = (self.project_root / "04_outputs"
                          / "R4_spi_projection" / "tables"
                          / "spi_projection_long.csv")
        self.out_dir   = (self.project_root / "04_outputs" / "X2_arcgis_export")
        self.out_dir.mkdir(parents=True, exist_ok=True)


CFG = Config()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("x4_freq_model")


# --------------------------------------------------------------------------- #
# 1. STATION METADATA
# --------------------------------------------------------------------------- #

def load_station_metadata(cfg: Config, station_ids: List[str]) -> pd.DataFrame:
    df = pd.read_csv(cfg.meta_file)
    keep = df[df["master_station_id"].isin(station_ids)].copy()
    meta = pd.DataFrame({
        "station_id":   keep["master_station_id"].astype(str).values,
        "station_name": keep["station_name"].astype(str).values,
        "x_lambert_km": pd.to_numeric(keep["x"], errors="coerce").values,
        "y_lambert_km": pd.to_numeric(keep["y"], errors="coerce").values,
        "z_elev_m":     pd.to_numeric(keep["z"], errors="coerce").values,
    })
    try:
        import pyproj
        tr = pyproj.Transformer.from_crs("EPSG:30791", "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(meta["x_lambert_km"].values * 1000.0,
                                meta["y_lambert_km"].values * 1000.0)
        meta["lat_wgs84"] = lat
        meta["lon_wgs84"] = lon
    except ImportError:
        log.warning("pyproj missing — lat/lon columns skipped.")
    return meta


# --------------------------------------------------------------------------- #
# 2. DROUGHT-YEAR COUNTING (one realisation at a time)
# --------------------------------------------------------------------------- #

def count_drought_years(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """For each group key, return the number of years that contained at least
    one month with SPI <= severe_threshold.

    df must contain columns: station, model, scenario, year, spi.
    """
    df = df.copy()
    df["is_severe"] = df["spi"] <= cfg.severe_threshold

    yearly = (df.groupby(["station", "model", "scenario", "year"])
              .agg(n_severe=("is_severe", "sum"),
                   spi_year_min=("spi", "min"))
              .reset_index())
    yearly["is_drought_year"] = yearly["n_severe"] >= 1

    per_realisation = (yearly.groupby(["station", "model", "scenario"])
                       .agg(n_drought_years=("is_drought_year", "sum"),
                            total_severe_months=("n_severe", "sum"),
                            min_spi_in_period=("spi_year_min", "min"))
                       .reset_index())
    return per_realisation


# --------------------------------------------------------------------------- #
# 3. MAIN
# --------------------------------------------------------------------------- #

def main() -> None:
    log.info("Loading SPI long table: %s", CFG.spi_file)
    spi = pd.read_csv(CFG.spi_file, parse_dates=["date"])
    spi = spi[spi["accumulation_k"] == CFG.accumulation_k].copy()
    spi["year"] = spi["date"].dt.year
    log.info("SPI-%d rows: %d", CFG.accumulation_k, len(spi))

    # ---- Historical (OBS, one realisation) -------------------------------- #
    obs = spi[(spi["model"] == "OBS")
              & (spi["date"].between(CFG.hist_start, CFG.hist_end))].copy()
    hist_per_real = count_drought_years(obs, CFG)
    # Collapse the single OBS realisation to one row per station
    hist = (hist_per_real
            .groupby("station")
            .agg(hist_n_drought_years=("n_drought_years", "sum"),
                 hist_total_severe_months=("total_severe_months", "sum"),
                 hist_min_spi=("min_spi_in_period", "min"))
            .reset_index())
    hist["hist_rate_per_decade"] = (10.0 * hist["hist_n_drought_years"]
                                    / CFG.hist_window_years)

    # ---- Projection (per model, then aggregate across models) ------------- #
    proj = spi[(spi["model"].isin(CFG.retained_models))
               & (spi["scenario"].isin(CFG.scenarios))
               & (spi["date"].between(CFG.proj_start, CFG.proj_end))].copy()
    proj_per_real = count_drought_years(proj, CFG)

    # Cross-model statistics per (station, scenario)
    cross_model = (proj_per_real.groupby(["station", "scenario"])
                   .agg(n_drought_years_mean=("n_drought_years", "mean"),
                        n_drought_years_min =("n_drought_years", "min"),
                        n_drought_years_max =("n_drought_years", "max"),
                        n_drought_years_std =("n_drought_years", "std"),
                        total_severe_months_mean=("total_severe_months", "mean"),
                        min_spi_in_period_min   =("min_spi_in_period", "min"),
                        n_models=("model", "nunique"))
                   .reset_index())
    cross_model["rate_per_decade_mean"] = (10.0 * cross_model["n_drought_years_mean"]
                                            / CFG.proj_window_years)
    cross_model["rate_per_decade_min"]  = (10.0 * cross_model["n_drought_years_min"]
                                            / CFG.proj_window_years)
    cross_model["rate_per_decade_max"]  = (10.0 * cross_model["n_drought_years_max"]
                                            / CFG.proj_window_years)

    # Wide pivot: one row per station, columns per scenario
    proj_wide = cross_model.pivot(index="station", columns="scenario")
    proj_wide.columns = [f"{c}_{s}" for c, s in proj_wide.columns]
    proj_wide = proj_wide.reset_index()

    # Merge meta + historical + projection
    meta = load_station_metadata(CFG, sorted(spi["station"].unique()))
    meta = meta.rename(columns={"station_id": "station"})

    out = (meta
           .merge(hist, on="station", how="left")
           .merge(proj_wide, on="station", how="left")
           .rename(columns={"station": "station_id"}))

    # Honest deltas (per-decade)
    if "rate_per_decade_mean_ssp2_4_5" in out.columns:
        out["delta_rate_ssp245"] = (out["rate_per_decade_mean_ssp2_4_5"]
                                    - out["hist_rate_per_decade"])
    if "rate_per_decade_mean_ssp5_8_5" in out.columns:
        out["delta_rate_ssp585"] = (out["rate_per_decade_mean_ssp5_8_5"]
                                    - out["hist_rate_per_decade"])

    # Tidy floats
    float_cols = out.select_dtypes(include="float").columns
    out[float_cols] = out[float_cols].round(3)

    # Reorder columns for ArcGIS readability
    leading = ["station_id", "station_name",
               "x_lambert_km", "y_lambert_km", "z_elev_m",
               "lat_wgs84", "lon_wgs84",
               "hist_n_drought_years", "hist_rate_per_decade",
               "hist_min_spi"]
    rest = [c for c in out.columns if c not in leading]
    out = out[leading + rest]

    csv_path  = CFG.out_dir / "chelif_drought_frequency_per_model.csv"
    xlsx_path = CFG.out_dir / "chelif_drought_frequency_per_model.xlsx"
    out.to_csv(csv_path, index=False)
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            out.to_excel(writer, sheet_name="DroughtFreq_per_station",
                          index=False)
        log.info("Excel -> %s", xlsx_path)
    except Exception as exc:
        log.warning("Excel write failed (%s) — CSV still produced.", exc)
    log.info("CSV   -> %s", csv_path)

    # ---- Console summary --------------------------------------------------- #
    print("\n=== Honest drought-year frequency comparison (per-model averaging) ===\n")
    cols = [
        "station_id", "station_name",
        "hist_n_drought_years", "hist_rate_per_decade",
        "n_drought_years_mean_ssp2_4_5", "n_drought_years_min_ssp2_4_5",
        "n_drought_years_max_ssp2_4_5", "rate_per_decade_mean_ssp2_4_5",
        "delta_rate_ssp245",
        "n_drought_years_mean_ssp5_8_5", "rate_per_decade_mean_ssp5_8_5",
        "delta_rate_ssp585",
    ]
    cols = [c for c in cols if c in out.columns]
    print(out[cols].to_string(index=False))

    print("\nInterpretation guide:")
    print("  - hist_*           : historical observed ONM (1990-2014, 25 years).")
    print("  - n_drought_years_mean_ssp* : average number of drought years across")
    print("    the 3 retained CMIP6 models, for that scenario, over 2015-2040.")
    print("  - n_drought_years_min/max   : envelope across models (lowest- and")
    print("    highest-vulnerability model for that station × scenario).")
    print("  - delta_rate_ssp* : projected change per decade (per-model mean rate")
    print("                       MINUS historical observed rate). Positive = more")
    print("                       drought years projected; negative = fewer.")
    print("\n*** This is the per-model comparison used in the thesis. ***")
    print("*** Use these deltas (not those of X3) for the cartography. ***")


if __name__ == "__main__":
    main()
