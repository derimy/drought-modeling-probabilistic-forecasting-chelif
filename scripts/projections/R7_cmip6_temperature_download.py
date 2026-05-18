"""
Author        : Imene DERRAR
Supervisor    : Abdelillah OTMANE CHERIF
Institutions  : NHSM — National Higher School of Mathematics
                IHFR — Institut Hydrométéorologique de Formation et de Recherche
Academic year : 2025–2026
Project       : Final-year engineering thesis in Applied Mathematics —
                Drought modelling and probabilistic forecasting for the Chélif basin

--------------------------------------------------------------------------------

CMIP6 monthly temperature SSP download for the scPDSI projection runs for the 3 retained
models on the temperature Taylor diagram.

Retained models (from the R8 temperature evaluation):
    ACCESS-CM2, CMCC-ESM2, GFDL-ESM4

Variables:
    tas, tasmax, tasmin   (monthly)

Experiments:
    ssp2_4_5   (2015-2040)
    ssp5_8_5   (2015-2040)

Total jobs: 3 models × 3 variables × 2 scenarios = 18.

Output layout:
    01_data/cmip6/{ssp2_4_5|ssp5_8_5}/<model>/<variable>/*.nc
"""

from __future__ import annotations

import logging
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import cdsapi
except ImportError as exc:
    raise SystemExit("pip install cdsapi") from exc


@dataclass
class Config:
    area_nwse: Tuple[float, float, float, float] = (36.5, 0.0, 34.0, 3.5)
    dataset: str = "projections-cmip6"
    temporal_resolution: str = "monthly"

    models: List[Dict[str, str]] = field(default_factory=lambda: [
        {"name": "access_cm2", "label": "ACCESS-CM2"},
        {"name": "cmcc_esm2",  "label": "CMCC-ESM2"},
        {"name": "gfdl_esm4",  "label": "GFDL-ESM4"},
    ])

    variables: Dict[str, str] = field(default_factory=lambda: {
        "tas":    "near_surface_air_temperature",
        "tasmax": "daily_maximum_near_surface_air_temperature",
        "tasmin": "daily_minimum_near_surface_air_temperature",
    })

    scenarios: List[str] = field(default_factory=lambda: ["ssp2_4_5", "ssp5_8_5"])
    year_start: int = 2015
    year_end:   int = 2040

    project_root: Path = Path(__file__).resolve().parents[3]
    out_root:     Path = field(init=False)
    zip_root:     Path = field(init=False)

    max_retries: int = 3
    retry_sleep: int = 60

    def __post_init__(self) -> None:
        self.out_root = self.project_root / "01_data" / "cmip6"
        self.zip_root = (self.project_root / "01_data" / "cmip6"
                          / "_zips_temperature_ssp")
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.zip_root.mkdir(parents=True, exist_ok=True)


CFG = Config()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("cmip6_T_ssp")


def request_one(client, model_key, model_label, scenario, var_short, var_full,
                cfg: Config) -> Path | None:
    years  = [str(y)     for y in range(cfg.year_start, cfg.year_end + 1)]
    months = [f"{m:02d}" for m in range(1, 13)]
    target_zip = cfg.zip_root / f"{model_key}__{scenario}__{var_short}.zip"

    if target_zip.exists() and target_zip.stat().st_size > 1024:
        log.info("[%s/%s/%s] already on disk, skipping",
                 model_label, scenario, var_short)
        return target_zip

    payload = {
        "temporal_resolution": cfg.temporal_resolution,
        "experiment":          scenario,
        "variable":            var_full,
        "model":               model_key,
        "year":                years,
        "month":               months,
        "area":                list(cfg.area_nwse),
    }

    for attempt in range(1, cfg.max_retries + 1):
        try:
            log.info("[%s/%s/%s] requesting (attempt %d)…",
                     model_label, scenario, var_short, attempt)
            client.retrieve(cfg.dataset, payload, str(target_zip))
            log.info("[%s/%s/%s] downloaded -> %s (%.1f MB)",
                     model_label, scenario, var_short, target_zip.name,
                     target_zip.stat().st_size / 1024 ** 2)
            return target_zip
        except Exception as exc:                                    # noqa: BLE001
            log.warning("[%s/%s/%s] attempt %d failed: %s",
                        model_label, scenario, var_short, attempt, exc)
            if attempt < cfg.max_retries:
                time.sleep(cfg.retry_sleep)
    return None


def unwrap(zip_path: Path, model_key: str, scenario: str, var_short: str,
           cfg: Config) -> int:
    target_dir = cfg.out_root / scenario / model_key / var_short
    target_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.lower().endswith(".nc"):
                dest = target_dir / Path(member).name
                with zf.open(member) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                n += 1
    log.info("[%s/%s/%s] unwrapped %d NetCDF(s) -> %s",
             model_key, scenario, var_short, n, target_dir)
    return n


def main() -> None:
    log.info("Temperature SSP downloads")
    log.info("Models:     %s",  [m["label"] for m in CFG.models])
    log.info("Variables:  %s",  list(CFG.variables))
    log.info("Scenarios:  %s",  CFG.scenarios)
    log.info("Total jobs: %d",  len(CFG.models) * len(CFG.variables) * len(CFG.scenarios))

    client = cdsapi.Client()
    summary: List[Tuple[str, str, str, str, int]] = []

    for model in CFG.models:
        for scen in CFG.scenarios:
            for var_short, var_full in CFG.variables.items():
                zp = request_one(client,
                                  model["name"], model["label"],
                                  scen, var_short, var_full, CFG)
                if zp is None:
                    summary.append((model["label"], scen, var_short, "FAILED", 0))
                    continue
                try:
                    n = unwrap(zp, model["name"], scen, var_short, CFG)
                    summary.append((model["label"], scen, var_short, "ok", n))
                except Exception as exc:                            # noqa: BLE001
                    log.error("unwrap failed: %s", exc)
                    summary.append((model["label"], scen, var_short, "UNWRAP_FAIL", 0))

    print("\n" + "=" * 80)
    print(f"{'Model':<14}{'Scenario':<12}{'Variable':<10}{'Status':<14}{'#nc':>5}")
    print("-" * 80)
    for m, s, v, st, n in summary:
        print(f"{m:<14}{s:<12}{v:<10}{st:<14}{n:>5}")
    print("=" * 80)
    n_ok = sum(1 for *_, st, _ in [(*row,) for row in summary] if st == "ok")
    print(f"\n{n_ok}/{len(summary)} jobs succeeded.")


if __name__ == "__main__":
    main()
