"""
Author        : Imene DERRAR
Supervisor    : Dr. Abdelillah OTMANE CHERIF
Institutions  : NHSM — National Higher School of Mathematics
                IHFR — Institut Hydrométéorologique de Formation et de Recherche
Academic year : 2025–2026
Project       : Final-year engineering thesis in Applied Mathematics —
                Drought modelling and probabilistic forecasting for the Chélif basin

--------------------------------------------------------------------------------

E5_build_basin_precip_master.py
================================================================================
Build the canonical basin-mean monthly precipitation series for the Chelif basin
from the SPI station panel.

Output:
    01_data/basin_precip_master_1990_2015.csv
    Columns:
        date                       monthly timestamp (month-start)
        precip_basin_mm            unweighted arithmetic mean of station P
                                   across the 9 retained stations (mm/month)
        n_stations_contributing    count of stations with non-NaN precipitation
                                   for that month (should be 9 throughout)
        n_stations_interpolated    count of stations whose interpolation flag
                                   is True for that month (provenance trace)

Rationale:
    Both SPI and scPDSI must rest on the same precipitation observations
    (gauge network, not gridded reanalysis). The aggregation operator here is
    the simple unweighted arithmetic mean across stations -- the same operator
    the SPI notebook applies to its per-station SPI values to form the
    regional series. This file is the canonical basin precipitation input used by both the
    SPI and scPDSI pipelines.

Source columns accepted (in precedence order):
    * `precip_filled` + `was_interpolated`     (canonical, from spi_station_series.csv)
    * `precip_mm`     + `data_flag`            (from spi_ready_dataset_main_*.csv;
                                                precip_mm there IS the gap-filled
                                                series, just renamed at publication
                                                time, and data_flag = "interpolated"
                                                or "observed_or_merged")

Usage:
    python E5_build_basin_precip_master.py
    python E5_build_basin_precip_master.py --in <station_csv> --out <basin_csv>
================================================================================
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

# Default locations: try the SPI notebook's output first, fall back to the
# published "main" file at the project root if needed.
DEFAULT_IN_CANDIDATES = [
    ROOT / "03_outputs" / "02_spi_construction_and_validation" / "spi_station_series.csv",
    ROOT / "01_data" / "spi_ready_dataset_main_1990_2015.csv.csv",
    ROOT / "01_data" / "spi_ready_dataset_main_1990_2015.csv",
]
DEFAULT_OUT = ROOT / "01_data" / "basin_precip_master_1990_2015.csv"

EXPECTED_START = pd.Timestamp("1990-01-01")
EXPECTED_END   = pd.Timestamp("2015-12-01")
EXPECTED_INDEX = pd.date_range(EXPECTED_START, EXPECTED_END, freq="MS")
EXPECTED_N     = len(EXPECTED_INDEX)  # 312

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("basin_precip")


def resolve_input(cli_in: Path | None) -> Path:
    """Return the first existing candidate, or fail with a clear message."""
    if cli_in is not None:
        if not cli_in.exists():
            raise FileNotFoundError(f"--in does not exist: {cli_in}")
        return cli_in
    for c in DEFAULT_IN_CANDIDATES:
        if c.exists():
            return c
    raise FileNotFoundError(
        "None of the default station inputs were found. Tried:\n  "
        + "\n  ".join(str(c) for c in DEFAULT_IN_CANDIDATES)
        + "\nPass --in <path> explicitly."
    )


def build_basin_precip(stations_csv: Path) -> pd.DataFrame:
    """Pivot the station panel and return a basin-mean monthly P series."""
    df = pd.read_csv(stations_csv)
    if "date" not in df.columns:
        raise KeyError(f"{stations_csv} has no 'date' column. Cols: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp(how="start")

    # ---- Pick precipitation column (precedence: precip_filled > precip_mm) ----
    if "precip_filled" in df.columns:
        p_col = "precip_filled"
    elif "precip_mm" in df.columns:
        p_col = "precip_mm"
    else:
        raise KeyError(
            f"{stations_csv} carries neither 'precip_filled' nor 'precip_mm'. "
            f"Columns found: {list(df.columns)}"
        )

    # ---- Pick interpolation flag column ----
    if "was_interpolated" in df.columns:
        flag = df["was_interpolated"].astype(bool)
    elif "data_flag" in df.columns:
        flag = df["data_flag"].astype(str).str.lower().eq("interpolated")
    else:
        log.warning("no interpolation-flag column found; "
                    "n_stations_interpolated will be 0 throughout")
        flag = pd.Series(False, index=df.index)
    df = df.assign(_interp=flag)

    if "master_station_id" not in df.columns:
        raise KeyError(f"{stations_csv} has no 'master_station_id' column. "
                       f"Cols: {list(df.columns)}")

    log.info("[in] precipitation column : %s", p_col)
    log.info("[in] stations             : %d", df["master_station_id"].nunique())
    log.info("[in] panel rows           : %d", len(df))

    # ---- Pivot to date x station ----
    p_pivot = df.pivot_table(
        index="date", columns="master_station_id", values=p_col, aggfunc="first"
    )
    i_pivot = df.pivot_table(
        index="date", columns="master_station_id", values="_interp", aggfunc="first"
    )

    # ---- Aggregate ----
    basin = pd.DataFrame({
        "precip_basin_mm":         p_pivot.mean(axis=1, skipna=True).round(4),
        "n_stations_contributing": p_pivot.notna().sum(axis=1).astype(int),
        "n_stations_interpolated": i_pivot.fillna(False).astype(bool).sum(axis=1).astype(int),
    })
    basin.index.name = "date"
    return basin.sort_index()


def validate(basin: pd.DataFrame) -> None:
    """Validation checks on schema, time axis, and data integrity."""
    expected_cols = {"precip_basin_mm", "n_stations_contributing", "n_stations_interpolated"}
    missing = expected_cols - set(basin.columns)
    if missing:
        raise AssertionError(f"missing required output columns: {missing}")

    idx = pd.DatetimeIndex(basin.index)
    if not idx.equals(EXPECTED_INDEX):
        raise AssertionError(
            f"time axis mismatch.\n"
            f"  expected: {EXPECTED_START:%Y-%m} -> {EXPECTED_END:%Y-%m}  ({EXPECTED_N} months)\n"
            f"  got     : {idx.min():%Y-%m} -> {idx.max():%Y-%m}  ({len(idx)} months)"
        )

    n_nan = int(basin["precip_basin_mm"].isna().sum())
    if n_nan > 0:
        raise AssertionError(f"precip_basin_mm has {n_nan} NaN value(s); expected 0")

    if (basin["precip_basin_mm"] < 0).any():
        raise AssertionError("precip_basin_mm has negative values")

    log.info("[validate] schema OK, %d months, no NaN, all non-negative", len(idx))


def report(basin: pd.DataFrame) -> None:
    """Print summary statistics for the operator's log."""
    s = basin["precip_basin_mm"]
    n = basin["n_stations_contributing"]
    i = basin["n_stations_interpolated"]
    annual = s.resample("YS").sum()

    log.info("[stats] precip_basin_mm  : mean=%.2f  std=%.2f  min=%.2f  max=%.2f  (mm/month)",
             s.mean(), s.std(), s.min(), s.max())
    log.info("[stats] n_stations_contr.: min=%d max=%d  always-9? %s",
             int(n.min()), int(n.max()), bool((n == 9).all()))
    log.info("[stats] interpolated cell: total=%d  (%.2f%% of panel)",
             int(i.sum()), 100.0 * i.sum() / (len(basin) * max(int(n.max()), 1)))
    log.info("[stats] annual basin P   : mean=%.0f  min=%.0f (%s)  max=%.0f (%s) mm/yr",
             annual.mean(),
             annual.min(), annual.idxmin().year,
             annual.max(), annual.idxmax().year)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build basin_precip_master_1990_2015.csv from the SPI station panel."
    )
    parser.add_argument("--in",  dest="inp",  type=Path, default=None,
                        help="station panel CSV (default: search standard locations)")
    parser.add_argument("--out", dest="outp", type=Path, default=DEFAULT_OUT,
                        help=f"basin precip master CSV (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    inp = resolve_input(args.inp)

    log.info("=" * 78)
    log.info("Basin precipitation master builder - Chelif basin (1990-2015)")
    log.info("  input  : %s", inp)
    log.info("  output : %s", args.outp)
    log.info("=" * 78)

    basin = build_basin_precip(inp)
    validate(basin)
    report(basin)

    args.outp.parent.mkdir(parents=True, exist_ok=True)
    basin.to_csv(args.outp, date_format="%Y-%m-%d")
    log.info("[out] wrote %d rows -> %s", len(basin), args.outp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
