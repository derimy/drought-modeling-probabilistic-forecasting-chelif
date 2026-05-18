"""
Author        : Imene DERRAR
Supervisor    : Dr. Abdelillah OTMANE CHERIF
Institutions  : NHSM — National Higher School of Mathematics
                IHFR — Institut Hydrométéorologique de Formation et de Recherche
Academic year : 2025–2026
Project       : Final-year engineering thesis in Applied Mathematics —
                Drought modelling and probabilistic forecasting for the Chélif basin

--------------------------------------------------------------------------------

E1_ingest_ihfr_onm.py
================================================================================
Reproducible ingestion of IHFR / ONM raw precipitation datasets into the
structured project. The ingestion is designed to be side-by-side with the
existing workflow: nothing outside the IHFR sub-tree is modified.

Design safeguards:
  * No file outside the IHFR sub-tree is read, modified, moved or deleted.
  * No existing folder is renamed.
  * The script only WRITES to:
        01_data/raw/ihfr/original/    (extracted raw, byte-identical)
        01_data/raw/ihfr/extracted/   (optional structured copies; unused here)
        01_data/raw/ihfr/metadata/    (provenance JSON + ingestion log CSV)
        01_data/external/ihfr_clean/  (one tidy CSV per station-sheet)
  * Existing notebooks, scripts, outputs and reports remain untouched.

Pipeline:
  1. Extract data.zip into 01_data/raw/ihfr/original/  (preserve sub-tree).
  2. Walk every Excel file (.xls / .xlsx) under original/, skipping junk
     (.dotx, .tmp, ~$ Excel lock files, the data/processed/ branch).
  3. For each sheet that looks like a precipitation station table:
       - extract X / Y / Z metadata header
       - detect month-order (civil  janv->dec   OR  hydrological  sept->aout)
       - reshape from wide (year x month) to long (date, precip_mm)
       - assign calendar year correctly under the hydrological convention
  4. Emit one CSV per (file, sheet) combination to ihfr_clean/.
  5. Write metadata JSON + ingestion log CSV.
  6. Append a "Raw Data Integration — IHFR / ONM" section to README.md
     (only if the section does not already exist).

No interpolation, no value alteration: missing months stay missing.

Run from the project root:
    python E1_ingest_ihfr_onm.py /path/to/data.zip

If the zip path is omitted, the script searches a couple of conventional
locations (uploads/, project root, current directory).
================================================================================
"""

from __future__ import annotations

import csv
import datetime as dt
import os
import hashlib
import json
import logging
import re
import shutil
import sys
import unicodedata
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ----------------------------------------------------------------------------
# CONFIG  (paths only; do not edit anything else)
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(
    os.environ.get("CHELIF_PROJECT_ROOT", Path.cwd())
)

IHFR_DIR        = PROJECT_ROOT / "01_data" / "raw"      / "ihfr"
ORIGINAL_DIR    = IHFR_DIR     / "original"
EXTRACTED_DIR   = IHFR_DIR     / "extracted"
METADATA_DIR    = IHFR_DIR     / "metadata"
CLEAN_DIR       = PROJECT_ROOT / "01_data" / "external" / "ihfr_clean"
README_PATH     = PROJECT_ROOT / "README.md"

# Where to look for data.zip if no argument is given
DEFAULT_ZIP_CANDIDATES = [
    PROJECT_ROOT.parent / "uploads" / "data.zip",
    PROJECT_ROOT.parent / "data.zip",
    PROJECT_ROOT / "data.zip",
    Path.cwd() / "data.zip",
]

# Extensions handled
PROCESS_EXTS = {".xlsx", ".xls"}
SKIP_PREFIXES = ("~$",)             # Excel lock files
SKIP_EXTS     = {".dotx", ".tmp"}   # Word templates and temp files

# Sub-paths inside the zip we deliberately do not re-process
SKIP_PROCESSED_DIRS = ("data/processed", "data/interim")

# Month name -> month number (lowercased prefix match)
MONTH_PREFIXES: Dict[str, int] = {
    "janv": 1, "jan": 1, "janu": 1,
    "fev":  2, "fevr": 2, "feb": 2, "fbr": 2, "fevri": 2,
    "mar":  3, "mars": 3,
    "avr":  4, "avri": 4, "april": 4, "avril": 4,
    "mai":  5, "may": 5,
    "juin": 6, "jun": 6, "june": 6,
    "juil": 7, "juill": 7, "jul": 7, "july": 7,
    "aout": 8, "aou":  8, "aug": 8, "august": 8,
    "sept": 9, "sep":  9,
    "oct": 10, "octo": 10,
    "nov": 11, "nove": 11, "novem": 11,
    "dec": 12, "deca": 12, "dece": 12, "decem": 12,
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("ihfr-ingest")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s or "unnamed"


def file_md5(p: Path) -> str:
    h = hashlib.md5()                                                # noqa: S324
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def month_from_label(s: str) -> Optional[int]:
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    if not s:
        return None
    # Try longest prefix match first to avoid 'jan' shadowing 'janv', etc.
    for prefix in sorted(MONTH_PREFIXES.keys(), key=len, reverse=True):
        if s.startswith(prefix):
            return MONTH_PREFIXES[prefix]
    return None


# ----------------------------------------------------------------------------
# Phase 1 - Extraction
# ----------------------------------------------------------------------------
def find_zip(arg: Optional[str]) -> Path:
    if arg:
        p = Path(arg).expanduser()
        if p.exists():
            return p
        sys.exit(f"[FAIL] zip not found: {p}")
    for c in DEFAULT_ZIP_CANDIDATES:
        if c.exists():
            return c
    sys.exit("[FAIL] data.zip not found - pass its path as an argument")


def extract_zip(zip_path: Path) -> int:
    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    n_extracted = 0
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            if member.endswith("/"):
                continue
            target = ORIGINAL_DIR / member
            if target.exists() and target.stat().st_size == z.getinfo(member).file_size:
                continue                                              # idempotent
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n_extracted += 1
    log.info("[extract] %d files written to %s", n_extracted, ORIGINAL_DIR)
    return n_extracted


# ----------------------------------------------------------------------------
# Phase 2 - Per-sheet station parser
# ----------------------------------------------------------------------------
def _find_header_row(df_raw: pd.DataFrame) -> Optional[int]:
    """Return the row index whose cells look like a month header
    (>= 8 cells matching a French month name)."""
    n_scan = min(20, len(df_raw))
    for i in range(n_scan):
        cells = [str(v) for v in df_raw.iloc[i].fillna("").tolist()]
        n_months = sum(1 for c in cells if month_from_label(c) is not None)
        if n_months >= 8:
            return i
    return None


def _extract_metadata(df_raw: pd.DataFrame, header_row: int) -> Dict[str, Optional[float]]:
    """Pull X / Y / Z scalars and any 'Code/Nom station' free text from the
    rows above the header."""
    meta: Dict[str, Optional[float]] = {"X": None, "Y": None, "Z": None,
                                         "station_code": None,
                                         "station_name_in_sheet": None}
    label_pat = re.compile(r"\s*([XYZ])\s*:\s*([-+]?\d+(?:[.,]\d+)?)\s*$",
                            re.IGNORECASE)
    for i in range(header_row):
        for v in df_raw.iloc[i].fillna("").tolist():
            s = str(v).strip()
            if not s:
                continue
            m = label_pat.match(s)
            if m:
                key = m.group(1).upper()
                val = m.group(2).replace(",", ".")
                try:
                    meta[key] = float(val)
                except ValueError:
                    pass
                continue
            low = s.lower()
            if low.startswith("code station") and ":" in s:
                tail = s.split(":", 1)[1].strip()
                if tail:
                    meta["station_code"] = tail
            elif low.startswith("nom station") and ":" in s:
                tail = s.split(":", 1)[1].strip()
                if tail:
                    meta["station_name_in_sheet"] = tail
    return meta


def _detect_month_columns(df_raw: pd.DataFrame, header_row: int) -> Dict[int, int]:
    """Map column index -> month number (1..12)."""
    cells = df_raw.iloc[header_row].fillna("").tolist()
    out: Dict[int, int] = {}
    for ci, c in enumerate(cells):
        m = month_from_label(c)
        if m is not None and m not in out.values():
            out[ci] = m
    return out


def parse_station_sheet(df_raw: pd.DataFrame,
                        source_file: str,
                        sheet_name: str,
                        station_label: str) -> Tuple[Optional[pd.DataFrame], Dict]:
    """Return (long_df, info)."""
    info = {
        "source_file": source_file,
        "source_sheet": sheet_name,
        "station_label": station_label,
        "n_rows_raw": int(len(df_raw)),
        "header_row": None,
        "n_months_detected": 0,
        "year_convention": None,
        "n_years": 0,
        "n_records": 0,
        "missing_records": 0,
        "min_date": None,
        "max_date": None,
        "X": None, "Y": None, "Z": None,
        "station_code": None,
        "station_name_in_sheet": None,
        "duplicate_dates": 0,
        "skip_reason": None,
    }

    header_row = _find_header_row(df_raw)
    if header_row is None:
        info["skip_reason"] = "no month-name header detected"
        return None, info
    info["header_row"] = header_row

    col_to_month = _detect_month_columns(df_raw, header_row)
    info["n_months_detected"] = len(col_to_month)
    if len(col_to_month) < 8:
        info["skip_reason"] = f"only {len(col_to_month)} month columns"
        return None, info

    meta = _extract_metadata(df_raw, header_row)
    info.update({k: meta[k] for k in ("X", "Y", "Z", "station_code",
                                       "station_name_in_sheet")})

    # Year convention
    first_col_idx = min(col_to_month.keys())
    first_month   = col_to_month[first_col_idx]
    is_hydro      = (first_month == 9)
    info["year_convention"] = "hydrological_sept_aout" if is_hydro else "civil_jan_dec"

    # Extract records
    records: List[dict] = []
    n_missing = 0
    for i in range(header_row + 1, len(df_raw)):
        first = df_raw.iloc[i, 0]
        try:
            year = int(float(first))
        except (TypeError, ValueError):
            continue
        if year < 1900 or year > 2100:
            continue
        for col_idx, month_num in col_to_month.items():
            val = df_raw.iloc[i, col_idx]
            cal_year = year + 1 if (is_hydro and month_num <= 8) else year
            try:
                month_label = f"{cal_year:04d}-{month_num:02d}"
            except Exception:                                          # noqa: BLE001
                continue
            if pd.isna(val) or str(val).strip() == "":
                n_missing += 1
                records.append({"date": month_label, "precip_mm": None,
                                "year_label": int(year)})
                continue
            try:
                fv = float(val)
            except (TypeError, ValueError):
                n_missing += 1
                records.append({"date": month_label, "precip_mm": None,
                                "year_label": int(year)})
                continue
            records.append({"date": month_label, "precip_mm": fv,
                            "year_label": int(year)})

    if not records:
        info["skip_reason"] = "no parseable year rows"
        return None, info

    df_long = pd.DataFrame.from_records(records)
    df_long["station_label"]         = station_label
    df_long["station_code"]          = meta["station_code"]
    df_long["station_name_in_sheet"] = meta["station_name_in_sheet"]
    df_long["X"] = meta["X"]; df_long["Y"] = meta["Y"]; df_long["Z"] = meta["Z"]
    df_long["source_file"]    = source_file
    df_long["source_sheet"]   = sheet_name
    df_long["year_convention"] = info["year_convention"]

    # Validation summary
    info["n_records"]      = int(len(df_long))
    info["missing_records"] = int(n_missing)
    info["n_years"]        = int(df_long["year_label"].nunique())
    valid = df_long.dropna(subset=["precip_mm"])
    if len(valid):
        info["min_date"] = str(valid["date"].min())
        info["max_date"] = str(valid["date"].max())
    info["duplicate_dates"] = int(df_long.duplicated(subset=["date"]).sum())

    df_long = df_long[["date", "precip_mm", "station_label", "station_code",
                        "station_name_in_sheet", "X", "Y", "Z",
                        "source_file", "source_sheet", "year_convention"]]
    df_long = df_long.sort_values("date").reset_index(drop=True)
    return df_long, info


# ----------------------------------------------------------------------------
# Phase 3 - Walk + standardise
# ----------------------------------------------------------------------------
def is_processable(p: Path) -> bool:
    if p.suffix.lower() not in PROCESS_EXTS:
        return False
    if p.name.startswith(SKIP_PREFIXES):
        return False
    rel = p.relative_to(ORIGINAL_DIR).as_posix()
    if any(rel.startswith(d) for d in SKIP_PROCESSED_DIRS):
        return False
    return True


def ingest() -> Tuple[List[Dict], List[Dict]]:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    log_rows: List[Dict] = []
    sheet_infos: List[Dict] = []

    for f in sorted(ORIGINAL_DIR.rglob("*")):
        if not f.is_file() or not is_processable(f):
            continue
        rel = f.relative_to(ORIGINAL_DIR).as_posix()
        log.info("[parse] %s", rel)
        try:
            xl = pd.ExcelFile(f)
        except Exception as exc:                                       # noqa: BLE001
            log.warning("  cannot open: %s", exc)
            log_rows.append({
                "original_filename": rel,
                "output_filename": "",
                "processing_timestamp": dt.datetime.utcnow().isoformat(timespec="seconds"),
                "transformation_summary": f"open_failed: {exc}",
            })
            continue

        # Default station label from filename stem
        default_label = slugify(f.stem)

        for sheet in xl.sheet_names:
            try:
                df_raw = pd.read_excel(xl, sheet_name=sheet, header=None,
                                        dtype=object)
            except Exception as exc:                                   # noqa: BLE001
                log.warning("  sheet read failed [%s]: %s", sheet, exc)
                continue
            if df_raw.empty:
                continue

            # Multi-station files use the SHEET name as station label;
            # single-station files use the FILE stem.
            station_label = (slugify(sheet) if len(xl.sheet_names) > 3
                              else default_label)

            df_long, info = parse_station_sheet(df_raw, rel, sheet, station_label)
            sheet_infos.append(info)

            if df_long is None:
                continue

            out_name = f"{station_label}__{slugify(f.stem)}__{slugify(sheet)}.csv"
            out_path = CLEAN_DIR / out_name
            df_long.to_csv(out_path, index=False)
            log_rows.append({
                "original_filename": rel,
                "output_filename":   str(out_path.relative_to(PROJECT_ROOT)),
                "processing_timestamp": dt.datetime.utcnow().isoformat(timespec="seconds"),
                "transformation_summary": (
                    f"sheet={sheet!r} convention={info['year_convention']} "
                    f"records={info['n_records']} missing={info['missing_records']} "
                    f"years={info['n_years']} range={info['min_date']}->{info['max_date']}"
                ),
            })

    return log_rows, sheet_infos


# ----------------------------------------------------------------------------
# Phase 4 - Metadata + log
# ----------------------------------------------------------------------------
def write_log(log_rows: List[Dict]) -> Path:
    p = METADATA_DIR / "ingestion_log.csv"
    fields = ["original_filename", "output_filename",
              "processing_timestamp", "transformation_summary"]
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in log_rows:
            w.writerow(r)
    log.info("[write] %s  (rows=%d)", p, len(log_rows))
    return p


def write_metadata(sheet_infos: List[Dict], log_rows: List[Dict]) -> Path:
    parsed = [i for i in sheet_infos if i.get("skip_reason") is None]
    skipped = [i for i in sheet_infos if i.get("skip_reason") is not None]

    stations = sorted({i["station_label"] for i in parsed if i["station_label"]})

    if parsed:
        all_min = min((i["min_date"] for i in parsed if i["min_date"]), default=None)
        all_max = max((i["max_date"] for i in parsed if i["max_date"]), default=None)
        total_records = sum(i["n_records"] for i in parsed)
        total_missing = sum(i["missing_records"] for i in parsed)
    else:
        all_min, all_max, total_records, total_missing = None, None, 0, 0

    meta = {
        "source": "IHFR / ONM (Algerian National Hydrological Service / Office National de la Meteorologie)",
        "ingested_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
        "n_excel_files_processed": len({i["source_file"] for i in sheet_infos}),
        "n_sheets_inspected":      len(sheet_infos),
        "n_sheets_parsed":         len(parsed),
        "n_sheets_skipped":        len(skipped),
        "n_clean_csvs_written":    len([r for r in log_rows if r["output_filename"]]),
        "stations": stations,
        "n_stations": len(stations),
        "time_coverage": {
            "min_date": all_min,
            "max_date": all_max,
        },
        "missing_value_summary": {
            "total_records": total_records,
            "missing":       total_missing,
            "missing_pct":   round(100.0 * total_missing / total_records, 3)
                              if total_records else None,
        },
        "validation": {
            "duplicates_per_sheet":
                {f"{i['source_file']}::{i['source_sheet']}": int(i["duplicate_dates"])
                 for i in parsed if i["duplicate_dates"]},
            "skipped_sheets":
                {f"{i['source_file']}::{i['source_sheet']}": i["skip_reason"]
                 for i in skipped},
        },
        "conventions_detected": sorted({i["year_convention"] for i in parsed
                                          if i.get("year_convention")}),
        "policy": {
            "interpolation": "none",
            "value_alteration": "none",
            "renaming": "none of the original files",
        },
    }
    p = METADATA_DIR / "ihfr_data_description.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log.info("[write] %s", p)
    return p


# ----------------------------------------------------------------------------
# Phase 5 - README append (idempotent)
# ----------------------------------------------------------------------------
README_SECTION_MARKER = "## Raw Data Integration — IHFR / ONM"

README_SECTION_BODY = """
This section documents the integration of the IHFR / ONM raw precipitation
archive into the structured project.  The integration is **side-by-side**:
no existing folder, file or notebook is modified.

* **Raw archive** is extracted as-is into
  `01_data/raw/ihfr/original/` (preserving the upstream sub-tree).
* **Ingestion** is performed by `E1_ingest_ihfr_onm.py`,
  which walks every `.xls` / `.xlsx` file (skipping `.dotx`, `.tmp`, and
  Excel lock files prefixed `~$`), detects the month-order convention
  (civil January-to-December or hydrological September-to-August), and
  reshapes each sheet from wide (year × month) to a tidy long format with
  columns `date`, `precip_mm`, `station_label`, `station_code`,
  `station_name_in_sheet`, `X`, `Y`, `Z`, `source_file`, `source_sheet`,
  `year_convention`.
* **Clean outputs** are one CSV per `(station, source_file, sheet)`,
  written to `01_data/external/ihfr_clean/`.
* **Provenance** is recorded in
  `01_data/raw/ihfr/metadata/ihfr_data_description.json` and
  `01_data/raw/ihfr/metadata/ingestion_log.csv`.

No interpolation, no value alteration, no renaming of the original files.
The pipeline is idempotent: re-running the script on a populated tree
produces no spurious duplicates.
"""


def append_readme_section() -> bool:
    if not README_PATH.exists():
        log.info("[readme] %s does not exist; skipping README append.", README_PATH)
        return False
    text = README_PATH.read_text(encoding="utf-8")
    if README_SECTION_MARKER in text:
        log.info("[readme] section already present; not appended.")
        return False
    if not text.endswith("\n"):
        text += "\n"
    text += "\n" + README_SECTION_MARKER + "\n" + README_SECTION_BODY
    README_PATH.write_text(text, encoding="utf-8")
    log.info("[readme] appended IHFR section to %s", README_PATH)
    return True


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main() -> int:
    if not PROJECT_ROOT.exists():
        sys.exit(f"[FAIL] project root not found: {PROJECT_ROOT}")
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    zip_path = find_zip(arg)
    log.info("data.zip          : %s", zip_path)
    log.info("project root      : %s", PROJECT_ROOT)
    log.info("ihfr original/    : %s", ORIGINAL_DIR)
    log.info("ihfr clean/       : %s", CLEAN_DIR)

    extract_zip(zip_path)

    log_rows, sheet_infos = ingest()
    write_log(log_rows)
    write_metadata(sheet_infos, log_rows)
    append_readme_section()

    n_clean = sum(1 for r in log_rows if r["output_filename"])
    log.info("=" * 64)
    log.info("DONE.  clean CSVs : %d", n_clean)
    log.info("       sheets read: %d", len(sheet_infos))
    log.info("       sheets ok  : %d",
             sum(1 for i in sheet_infos if i.get("skip_reason") is None))
    log.info("       sheets skip: %d",
             sum(1 for i in sheet_infos if i.get("skip_reason") is not None))
    log.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
