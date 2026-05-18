"""
Author        : Imene DERRAR
Supervisor    : Abdelillah OTMANE CHERIF
Institutions  : NHSM — National Higher School of Mathematics
                IHFR — Institut Hydrométéorologique de Formation et de Recherche
Academic year : 2025–2026
Project       : Final-year engineering thesis in Applied Mathematics —
                Drought modelling and probabilistic forecasting for the Chélif basin

--------------------------------------------------------------------------------

X2 — ArcGIS cartography export
========================================

Builds an Excel file with two sheets, plus two stand-alone CSV mirrors, for
ArcGIS cartography of historical vs projected SPI-6.

Sheet 1 (Historical): one row per (station × year) over 1990-2014.
    Columns: year, station_id, x_lambert_km, y_lambert_km, z_elev_m,
             spi6_annual_mean, spi6_annual_min, spi6_n_months,
             drought_class_dominant

Sheet 2 (Projection): one row per (station × year × scenario) over 2015-2040.
    Columns: year, station_id, x_lambert_km, y_lambert_km, z_elev_m,
             scenario, spi6_annual_mean, spi6_annual_min, spi6_n_months,
             drought_class_dominant
    The SPI-6 is the Brunner-weighted ensemble mean across the three
    retained CMIP6 models (CESM2 = 0.50, EC-Earth3-CC = 0.25,
    EC-Earth3-Veg-LR = 0.25), so the two EC-Earth members contribute one
    effective vote between them and CESM2 contributes a full vote.

The coordinates are kept in Lambert Nord Algérie (EPSG:30791, km) — the
native convention of the ONM station metadata and the standard projection
used by Algerian topographic/hydrological work in ArcGIS.  If you also want
WGS84 lat/lon for a non-projected map, swap `_LATLON_COLUMNS` on in the
config: it appends `lat_wgs84` and `lon_wgs84` to both sheets.

Inputs
------
    01_data/external/master_station_metadata_final_plus_pluvio.csv
        Station coordinates (x, y in km; z in m).
    04_outputs/R4_spi_projection/tables/spi_projection_long.csv
        SPI long table from R4 (k = 1, 3, 6, 12).  OBS rows
        provide the historical SPI-6; model rows provide the projection.

Outputs
-------
    04_outputs/X2_arcgis_export/chelif_spi6_arcgis.xlsx     (two sheets)
    04_outputs/X2_arcgis_export/chelif_spi6_historical.csv  (sheet 1 mirror)
    04_outputs/X2_arcgis_export/chelif_spi6_projection.csv  (sheet 2 mirror)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 0. CONFIGURATION
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    accumulation_k: int = 6

    hist_start: str = "1990-01"
    hist_end:   str = "2014-12"
    proj_start: str = "2015-01"
    proj_end:   str = "2040-12"

    # Brunner-weighted ensemble — normalised so weights sum to 1.0.
    # CESM2 (structurally distinct)   weight 0.50
    # EC-Earth3-CC + EC-Earth3-Veg-LR weight 0.50 between them (0.25 each)
    model_weights: Dict[str, float] = field(default_factory=lambda: {
        "CESM2":            0.50,
        "EC-Earth3-CC":     0.25,
        "EC-Earth3-Veg-LR": 0.25,
    })

    # SPI -> McKee class threshold (the value used for the
    # `drought_class_dominant` column).
    spi_class_edges:  tuple = (-2.0, -1.5, -1.0, 1.0, 1.5, 2.0)
    spi_class_labels: tuple = ("extremely_dry", "severely_dry", "moderately_dry",
                                "near_normal", "moderately_wet",
                                "severely_wet", "extremely_wet")

    # Drought-year filter:  retain only (station × year) rows where at
    # least one calendar month of that year had SPI-6 <= severe_threshold.
    # severe_threshold = -1.5 captures both McKee classes "severely dry"
    # (-2 < SPI <= -1.5) and "extremely dry" (SPI <= -2).
    severe_threshold:        float = -1.5
    drought_year_filter_on:  bool  = True
    drought_min_n_months:    int   = 1     # minimum severe months/year to retain

    # Append WGS84 lat/lon columns alongside the Lambert coordinates?
    add_wgs84_columns: bool = True

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
        self.out_dir   = (self.project_root / "04_outputs"
                          / "X2_arcgis_export")
        self.out_dir.mkdir(parents=True, exist_ok=True)


CFG = Config()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("x2_arcgis")


# --------------------------------------------------------------------------- #
# 1. UTILITIES
# --------------------------------------------------------------------------- #

def classify(spi: float, cfg: Config) -> str:
    if not np.isfinite(spi):
        return "missing"
    for edge, label in zip(cfg.spi_class_edges, cfg.spi_class_labels):
        if spi < edge:
            return label
    return cfg.spi_class_labels[-1]


def mode_or_first(s: pd.Series) -> str:
    m = s.mode()
    return m.iloc[0] if len(m) else "missing"


# --------------------------------------------------------------------------- #
# 2. STATION METADATA (with optional Lambert -> WGS84)
# --------------------------------------------------------------------------- #

def load_station_metadata(cfg: Config, station_ids: list) -> pd.DataFrame:
    df = pd.read_csv(cfg.meta_file)
    keep = df[df["master_station_id"].isin(station_ids)].copy()

    # Standardise column names — the source file has `x`, `y`, `z` in km/m
    meta = pd.DataFrame({
        "station_id":    keep["master_station_id"].astype(str).values,
        "station_name":  keep["station_name"].astype(str).values,
        "x_lambert_km":  pd.to_numeric(keep["x"], errors="coerce").values,
        "y_lambert_km":  pd.to_numeric(keep["y"], errors="coerce").values,
        "z_elev_m":      pd.to_numeric(keep["z"], errors="coerce").values,
    })

    if cfg.add_wgs84_columns:
        try:
            import pyproj
            tr = pyproj.Transformer.from_crs("EPSG:30791", "EPSG:4326",
                                              always_xy=True)
            lon, lat = tr.transform(meta["x_lambert_km"].values * 1000.0,
                                    meta["y_lambert_km"].values * 1000.0)
            meta["lat_wgs84"] = lat
            meta["lon_wgs84"] = lon
        except ImportError:
            log.warning("pyproj not available — WGS84 columns skipped.")
    return meta


# --------------------------------------------------------------------------- #
# 3. AGGREGATIONS
# --------------------------------------------------------------------------- #

def aggregate_historical(spi_long: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Annual aggregation of OBS SPI-6 per (station × year), filtered to
    drought years only (at least one month with SPI <= severe_threshold)."""
    obs = spi_long[(spi_long["model"] == "OBS")
                   & (spi_long["accumulation_k"] == cfg.accumulation_k)].copy()
    obs = obs[obs["date"].between(cfg.hist_start, cfg.hist_end)]
    obs["year"]    = pd.to_datetime(obs["date"]).dt.year
    obs["class"]   = obs["spi"].apply(lambda v: classify(v, cfg))
    obs["is_sev"]  = obs["spi"] <= cfg.severe_threshold
    obs["is_ext"]  = obs["spi"] <= -2.0

    agg = (obs.groupby(["station", "year"])
              .agg(spi6_annual_mean=("spi", "mean"),
                   spi6_annual_min=("spi", "min"),
                   spi6_n_months=("spi", "count"),
                   n_severe_months=("is_sev", "sum"),
                   n_extreme_months=("is_ext", "sum"),
                   drought_class_dominant=("class", mode_or_first))
              .reset_index()
              .rename(columns={"station": "station_id"}))

    if cfg.drought_year_filter_on:
        agg = agg[agg["n_severe_months"] >= cfg.drought_min_n_months].copy()
    return agg


def aggregate_projection(spi_long: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Brunner-weighted annual aggregation of projected SPI-6 per
    (station × year × scenario).  Weights are applied at the *monthly* level
    before annual averaging, so the weights propagate correctly into the
    annual mean."""
    proj = spi_long[(spi_long["model"].isin(cfg.model_weights))
                    & (spi_long["scenario"].isin(("ssp2_4_5", "ssp5_8_5")))
                    & (spi_long["accumulation_k"] == cfg.accumulation_k)].copy()
    proj = proj[proj["date"].between(cfg.proj_start, cfg.proj_end)]
    proj["year"] = pd.to_datetime(proj["date"]).dt.year

    # Brunner-weighted monthly SPI per (station × scenario × date)
    proj["weight"] = proj["model"].map(cfg.model_weights)
    proj["weighted_spi"] = proj["spi"] * proj["weight"]
    monthly = (proj.groupby(["station", "scenario", "date", "year"])
               .agg(weighted_spi_sum=("weighted_spi", "sum"),
                    weight_sum=("weight", "sum"))
               .reset_index())
    monthly["spi6_brunner"] = monthly["weighted_spi_sum"] / monthly["weight_sum"]
    monthly["class"] = monthly["spi6_brunner"].apply(lambda v: classify(v, cfg))

    monthly["is_sev"] = monthly["spi6_brunner"] <= cfg.severe_threshold
    monthly["is_ext"] = monthly["spi6_brunner"] <= -2.0

    agg = (monthly.groupby(["station", "scenario", "year"])
                  .agg(spi6_annual_mean=("spi6_brunner", "mean"),
                       spi6_annual_min=("spi6_brunner", "min"),
                       spi6_n_months=("spi6_brunner", "count"),
                       n_severe_months=("is_sev", "sum"),
                       n_extreme_months=("is_ext", "sum"),
                       drought_class_dominant=("class", mode_or_first))
                  .reset_index()
                  .rename(columns={"station": "station_id"}))

    if cfg.drought_year_filter_on:
        agg = agg[agg["n_severe_months"] >= cfg.drought_min_n_months].copy()
    return agg


# --------------------------------------------------------------------------- #
# 4. MAIN
# --------------------------------------------------------------------------- #

def main() -> None:
    log.info("Loading SPI long table: %s", CFG.spi_file)
    spi_long = pd.read_csv(CFG.spi_file, parse_dates=["date"])
    spi_long["date"] = pd.to_datetime(spi_long["date"]).dt.to_period("M").dt.to_timestamp()
    log.info("SPI long table: %d rows", len(spi_long))

    stations = sorted(spi_long["station"].unique())
    meta = load_station_metadata(CFG, stations)
    log.info("Station metadata resolved for %d stations", len(meta))

    hist = aggregate_historical(spi_long, CFG)
    proj = aggregate_projection(spi_long, CFG)
    log.info("Historical rows  (station × year):              %d", len(hist))
    log.info("Projection rows  (station × year × scenario):   %d", len(proj))

    sheet1 = (meta.merge(hist, on="station_id", how="right")
                  .sort_values(["year", "station_id"]))
    sheet2 = (meta.merge(proj, on="station_id", how="right")
                  .sort_values(["scenario", "year", "station_id"]))

    desired_hist = [
        "year", "station_id", "station_name",
        "x_lambert_km", "y_lambert_km", "z_elev_m",
        "lat_wgs84", "lon_wgs84",
        "spi6_annual_mean", "spi6_annual_min",
        "n_severe_months", "n_extreme_months",
        "spi6_n_months", "drought_class_dominant",
    ]
    desired_proj = [
        "year", "station_id", "station_name",
        "x_lambert_km", "y_lambert_km", "z_elev_m",
        "lat_wgs84", "lon_wgs84",
        "scenario",
        "spi6_annual_mean", "spi6_annual_min",
        "n_severe_months", "n_extreme_months",
        "spi6_n_months", "drought_class_dominant",
    ]
    sheet1 = sheet1[[c for c in desired_hist if c in sheet1.columns]]
    sheet2 = sheet2[[c for c in desired_proj if c in sheet2.columns]]

    # Outputs — file names reflect the drought-year filter so they don't
    # collide with any earlier (unfiltered) export.
    suffix = "_drought_years" if CFG.drought_year_filter_on else ""
    xlsx_path = CFG.out_dir / f"chelif_spi6{suffix}_arcgis.xlsx"
    hist_csv  = CFG.out_dir / f"chelif_spi6{suffix}_historical.csv"
    proj_csv  = CFG.out_dir / f"chelif_spi6{suffix}_projection.csv"

    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            sheet1.to_excel(writer, sheet_name="Historical", index=False,
                            float_format="%.4f")
            sheet2.to_excel(writer, sheet_name="Projection", index=False,
                            float_format="%.4f")
        log.info("Excel file (2 sheets) -> %s", xlsx_path)
    except Exception as exc:
        log.error("Failed to write Excel file (%s).  CSV mirrors will still be written.",
                  exc)

    sheet1.to_csv(hist_csv, index=False, float_format="%.4f")
    sheet2.to_csv(proj_csv, index=False, float_format="%.4f")
    log.info("CSV mirror (Historical) -> %s", hist_csv)
    log.info("CSV mirror (Projection) -> %s", proj_csv)

    print("\n=== Historical sheet (head) ===")
    print(sheet1.head(10).to_string(index=False))
    print("\n=== Projection sheet (head) ===")
    print(sheet2.head(10).to_string(index=False))
    print(f"\nHistorical : {len(sheet1):>5d} rows  (drought years only: SPI-6 <= {CFG.severe_threshold:.1f})")
    print(f"Projection : {len(sheet2):>5d} rows  (drought years only: SPI-6 <= {CFG.severe_threshold:.1f}, "
          f"Brunner-weighted ensemble)")
    print("\nFilter definition:")
    print(f"  A 'drought year' is a (station, year) for which AT LEAST")
    print(f"  {CFG.drought_min_n_months} month(s) had SPI-6 <= {CFG.severe_threshold:.1f}")
    print(f"  (severely-dry or extremely-dry on the McKee scale).")
    print(f"  Columns n_severe_months and n_extreme_months tell you how many.")
    print("\nArcGIS hints:")
    print("  - In ArcGIS Pro, add the .xlsx via 'Add Data', then choose")
    print("    which sheet to load (Historical or Projection).")
    print("  - For 'XY Table to Point': use x_lambert_km × 1000 and")
    print("    y_lambert_km × 1000 as X-field/Y-field with the EPSG:30791")
    print("    spatial reference (Lambert Nord Algérie).  Or use the")
    print("    lat_wgs84/lon_wgs84 columns with EPSG:4326.")
    print("  - Symbolise spi6_annual_mean with a diverging RdBu palette")
    print("    centred on 0 to render wet/dry years on the map.")


if __name__ == "__main__":
    main()
