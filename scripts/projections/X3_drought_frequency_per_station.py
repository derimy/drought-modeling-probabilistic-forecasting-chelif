"""
Author        : Imene DERRAR
Supervisor    : Dr. Abdelillah OTMANE CHERIF
Institutions  : NHSM — National Higher School of Mathematics
                IHFR — Institut Hydrométéorologique de Formation et de Recherche
Academic year : 2025–2026
Project       : Final-year engineering thesis in Applied Mathematics —
                Drought modelling and probabilistic forecasting for the Chélif basin

--------------------------------------------------------------------------------

X3 — Drought-frequency-per-station (chronic vulnerability map)
========================================================================

Builds the companion table to X2 that ArcGIS uses for the
"chronic-vulnerability" map: one row per station, with the cumulative
drought statistics over the historical period and over each SSP scenario
in the projection period.

Reads the two drought-year tables produced by X2 and aggregates
them by station.  Output is a single CSV with 9 rows (one per station)
and a parallel sheet in an Excel file.

Output columns
--------------
    station_id, station_name, x_lambert_km, y_lambert_km, z_elev_m,
    lat_wgs84, lon_wgs84,
    hist_n_drought_years, hist_total_severe_months, hist_total_extreme_months,
    hist_worst_year, hist_worst_spi6_min, hist_mean_spi6_annual_min,
    ssp245_n_drought_years, ssp245_total_severe_months, ssp245_total_extreme_months,
    ssp245_worst_year, ssp245_worst_spi6_min, ssp245_mean_spi6_annual_min,
    ssp585_n_drought_years, ssp585_total_severe_months, ssp585_total_extreme_months,
    ssp585_worst_year, ssp585_worst_spi6_min, ssp585_mean_spi6_annual_min,
    delta_n_drought_years_ssp245, delta_n_drought_years_ssp585

The two "delta" columns express the change in the number of drought-years
between the projection (per scenario, 26 years) and the historical
(25 years), normalised per decade to make them directly comparable
despite the slightly different window lengths.

Inputs
------
    04_outputs/X2_arcgis_export/chelif_spi6_drought_years_historical.csv
    04_outputs/X2_arcgis_export/chelif_spi6_drought_years_projection.csv

Outputs
-------
    04_outputs/X2_arcgis_export/chelif_drought_frequency_per_station.csv
    04_outputs/X2_arcgis_export/chelif_drought_frequency_per_station.xlsx
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 0. CONFIGURATION
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    hist_window_years: int = 25       # 1990-2014
    proj_window_years: int = 26       # 2015-2040

    project_root: Path = Path(__file__).resolve().parents[3]
    in_dir:       Path = field(init=False)
    out_dir:      Path = field(init=False)

    def __post_init__(self) -> None:
        self.in_dir  = self.project_root / "04_outputs" / "X2_arcgis_export"
        self.out_dir = self.in_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)


CFG = Config()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("x3_freq_station")


# --------------------------------------------------------------------------- #
# 1. AGGREGATION HELPERS
# --------------------------------------------------------------------------- #

def aggregate_by_station(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Aggregate a drought-year long table by station_id, with column prefix."""
    rows = []
    for sid, sub in df.groupby("station_id"):
        worst_row = sub.loc[sub["spi6_annual_min"].idxmin()]
        rows.append({
            "station_id":                              sid,
            f"{prefix}_n_drought_years":               int(len(sub)),
            f"{prefix}_total_severe_months":           int(sub["n_severe_months"].sum()),
            f"{prefix}_total_extreme_months":          int(sub["n_extreme_months"].sum()),
            f"{prefix}_worst_year":                    int(worst_row["year"]),
            f"{prefix}_worst_spi6_min":                float(worst_row["spi6_annual_min"]),
            f"{prefix}_mean_spi6_annual_min":          float(sub["spi6_annual_min"].mean()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. MAIN
# --------------------------------------------------------------------------- #

def main() -> None:
    hist_path = CFG.in_dir / "chelif_spi6_drought_years_historical.csv"
    proj_path = CFG.in_dir / "chelif_spi6_drought_years_projection.csv"

    log.info("Reading historical drought years : %s", hist_path)
    hist = pd.read_csv(hist_path)
    log.info("Reading projection drought years : %s", proj_path)
    proj = pd.read_csv(proj_path)

    log.info("Historical rows  : %d", len(hist))
    log.info("Projection rows  : %d", len(proj))

    # Station-level metadata table (coordinates) — take it from the
    # historical file, since every station should appear there at least once
    meta_cols = ["station_id", "station_name", "x_lambert_km", "y_lambert_km",
                 "z_elev_m", "lat_wgs84", "lon_wgs84"]
    station_meta = (hist[meta_cols]
                    .drop_duplicates(subset="station_id")
                    .reset_index(drop=True))

    # Historical aggregation
    hist_agg = aggregate_by_station(hist, prefix="hist")

    # Projection: split by scenario, aggregate each
    proj_245 = aggregate_by_station(
        proj[proj["scenario"] == "ssp2_4_5"], prefix="ssp245")
    proj_585 = aggregate_by_station(
        proj[proj["scenario"] == "ssp5_8_5"], prefix="ssp585")

    # Merge everything
    out = (station_meta
           .merge(hist_agg,  on="station_id", how="left")
           .merge(proj_245, on="station_id", how="left")
           .merge(proj_585, on="station_id", how="left"))

    # Fill counts that have no rows (i.e., the station had no drought in
    # that period) with zero rather than NaN, so ArcGIS treats them as a
    # plain count and the symbology renders them correctly.
    count_cols = [c for c in out.columns
                  if "n_drought_years" in c
                  or "total_severe_months" in c
                  or "total_extreme_months" in c]
    out[count_cols] = out[count_cols].fillna(0).astype(int)

    # Drought-years-per-decade rate, for fair historical vs projection
    # comparison (windows differ by 1 year).
    out["hist_rate_per_decade"]    = 10.0 * out["hist_n_drought_years"]   / CFG.hist_window_years
    out["ssp245_rate_per_decade"]  = 10.0 * out["ssp245_n_drought_years"] / CFG.proj_window_years
    out["ssp585_rate_per_decade"]  = 10.0 * out["ssp585_n_drought_years"] / CFG.proj_window_years

    out["delta_rate_ssp245"] = out["ssp245_rate_per_decade"] - out["hist_rate_per_decade"]
    out["delta_rate_ssp585"] = out["ssp585_rate_per_decade"] - out["hist_rate_per_decade"]

    # Round floats to 3 decimals for readability
    float_cols = out.select_dtypes(include="float").columns
    out[float_cols] = out[float_cols].round(3)

    csv_path  = CFG.out_dir / "chelif_drought_frequency_per_station.csv"
    xlsx_path = CFG.out_dir / "chelif_drought_frequency_per_station.xlsx"
    out.to_csv(csv_path, index=False)
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            out.to_excel(writer, sheet_name="Frequency_per_station", index=False)
        log.info("Excel  -> %s", xlsx_path)
    except Exception as exc:
        log.warning("Excel write failed (%s) — CSV is still produced.", exc)

    log.info("CSV    -> %s", csv_path)

    # Console summary — the headline table for the discussion chapter
    print("\n=== Drought-year frequency per station ===")
    display_cols = [
        "station_id", "station_name",
        "hist_n_drought_years",
        "ssp245_n_drought_years",
        "ssp585_n_drought_years",
        "hist_rate_per_decade",
        "ssp245_rate_per_decade",
        "ssp585_rate_per_decade",
        "delta_rate_ssp245",
        "delta_rate_ssp585",
    ]
    print(out[display_cols].to_string(index=False))

    print("\nInterpretation guide:")
    print("  - hist_n_drought_years        : nb of years 1990-2014 with >= 1 month SPI-6 <= -1.5")
    print("  - ssp245/585_n_drought_years  : same for 2015-2040 under each SSP scenario")
    print("  - *_rate_per_decade           : normalised count per decade (fair comparison)")
    print("  - delta_rate_*                : projected change in drought-year rate (per decade)")
    print("                                    positive = more drought years in projection")
    print("\nArcGIS map suggestions:")
    print("  - Plot 9 points using lat_wgs84/lon_wgs84 (EPSG:4326).")
    print("  - Map A: symbolise by hist_n_drought_years (sequential reds).")
    print("  - Map B: symbolise by ssp245_n_drought_years (sequential reds, same scale).")
    print("  - Map C: symbolise by delta_rate_ssp245 (diverging RdBu, centred on 0).")
    print("  - The size or label can show hist_worst_spi6_min (the deepest drought peak).")


if __name__ == "__main__":
    main()
