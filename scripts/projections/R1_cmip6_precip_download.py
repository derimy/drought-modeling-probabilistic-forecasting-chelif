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

CMIP6 monthly precipitation download for the Chélif basin.

Downloads the final ensemble members used in R2/R3: MPI-ESM1-2-LR and
HadGEM3-GC31-LL historical and SSP runs, retrieved from the CDS
`projections-cmip6` product into 01_data/cmip6/.
"""
import shutil, zipfile
from pathlib import Path
import cdsapi

ROOT  = Path(__file__).resolve().parents[3]
ZIPS  = ROOT / "01_data" / "cmip6" / "_zips"
AREA  = [36.5, 0.0, 34.0, 3.5]
MONTH = [f"{m:02d}" for m in range(1, 13)]

JOBS = [
    ("mpi_esm1_2_lr",     "historical", range(1990, 2015)),
    ("hadgem3_gc31_ll",   "historical", range(1990, 2015)),
    ("hadgem3_gc31_ll",   "ssp2_4_5",   range(2015, 2041)),
    ("hadgem3_gc31_ll",   "ssp5_8_5",   range(2015, 2041)),
]

c = cdsapi.Client()
for model, exp, years in JOBS:
    out = ZIPS / f"{model}__{exp}.zip"
    if out.exists() and out.stat().st_size > 1024:
        print(f"[{model}/{exp}] skip"); continue
    print(f"[{model}/{exp}] requesting...")
    try:
        c.retrieve("projections-cmip6", {
            "temporal_resolution": "monthly",
            "experiment":          exp,
            "variable":            "precipitation",
            "model":               model,
            "year":                [str(y) for y in years],
            "month":               MONTH,
            "area":                AREA,
        }, str(out))
    except Exception as e:
        print(f"[{model}/{exp}] FAILED: {e}"); continue
    target = ROOT / "01_data" / "cmip6" / exp / model
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out) as zf:
        for m in zf.namelist():
            if m.lower().endswith(".nc"):
                with zf.open(m) as s, open(target / Path(m).name, "wb") as d:
                    shutil.copyfileobj(s, d)
    print(f"[{model}/{exp}] OK")
print("Done.")
