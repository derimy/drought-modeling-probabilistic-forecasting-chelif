# Drought Modelling and Probabilistic Forecasting Framework — Chélif Basin

Bayesian drought modelling and probabilistic forecasting framework for the Chélif basin (Algeria), combining SPI, scPDSI, CMIP6 projections, and uncertainty-aware time-series analysis.

---

**Author**: Imene DERRAR  
**Supervisor**: Dr. Abdelillah OTMANE CHERIF  
**Institutions**: NHSM — National Higher School of Mathematics · IHFR — Institut Hydrométéorologique de Formation et de Recherche  
**Academic year**: 2025–2026  
**Project**: Final-year engineering thesis in Applied Mathematics

---

## 1. Project overview

This repository accompanies a final-year engineering thesis on drought modelling and probabilistic forecasting in the Chélif basin, a semi-arid catchment in north-west Algeria. The work builds two complementary drought indices on the same precipitation observations and propagates structural, parameter, and scenario uncertainty into probabilistic statements about future drought conditions.

Two operational deliverables are documented here:

1. A **Bayesian probabilistic forecast** of the Standardised Precipitation Index (SPI) at the six-month accumulation scale, calibrated on station data from 1990–2014 and projected forward under SSP2-4.5 and SSP5-8.5.
2. A **complementary physically-interpretable drought signal** based on the self-calibrating Palmer Drought Severity Index (scPDSI), driven by TerraClimate and ERA5-Land forcings.

## 2. Scientific motivation

The Chélif basin is one of the most water-stressed catchments in North Africa. Reservoir management, agricultural planning, and water supply all depend on credible drought information at horizons of months to decades. The thesis asks two related questions:

- Can a probabilistic monthly forecast of SPI-6 produce calibrated drought-risk statements at short horizons given the limited observational record?
- How do CMIP6 projections of precipitation and temperature, after bias correction, translate into projected drought frequency under two SSP scenarios?

The framework is deliberately conservative: it propagates parameter, model, and scenario uncertainty rather than collapsing them into a single point estimate.

## 3. Study region

Chélif basin, north-west Algeria. Nine in-situ rain gauges retained for the 1990–2015 window after a three-condition eligibility filter (completeness, gap structure, geographical representativeness). The basin centroid is at approximately 35.2 °N. Mean annual precipitation is around 350 mm with strong seasonality concentrated in the winter half-year.

## 4. SPI and scPDSI framework

**SPI (Standardised Precipitation Index).** Constructed at three accumulation scales (3, 6, 12 months) following McKee, Doesken & Kleist (1993). Per-station, per-calendar-month Gamma distributions are fitted by maximum likelihood, zero precipitation is handled with a mixed-distribution adjustment, and the cumulative probability is transformed through the inverse standard normal. SPI-6 is the operational scale used in the forecasting and projection layers.

**scPDSI (self-calibrating Palmer Drought Severity Index).** Computed via the `scPDSI` R package and the `climate_indices` Python package. Two forcing branches are produced: a primary branch with TerraClimate Penman–Monteith PET and a validation branch with ERA5-Land temperature fed through a latitude-corrected Thornthwaite formulation. The same station-derived basin-mean precipitation feeds both branches, so SPI and scPDSI are anchored on the same observed rainfall.

The two indices are treated as **separate stochastic processes**. They are not merged in the data engineering layer; any cross-index analysis appears as an explicit joint discussion in the thesis text.

## 5. Bayesian forecasting philosophy

The probabilistic forecasting layer fits a Bayesian AR(1) model to SPI-6 under a conjugate Normal–Inverse-Gamma prior. The predictive distribution is Student-t in closed form, which provides calibrated short-horizon predictive intervals without MCMC. Evaluation uses rolling-origin forecasting, the Continuous Ranked Probability Score (CRPS), interval coverage at 90 %, and the Diebold–Mariano test with a Newey–West HAC variance estimator.

The same Bayesian framework is applied to the bias-corrected CMIP6 projections to produce probabilistic SPI-6 forecasts under SSP2-4.5 and SSP5-8.5 through 2040.

## 6. CMIP6 projection workflow

Fourteen CMIP6 models are evaluated against the in-situ ONM precipitation record using a composite Taylor skill score. Ten models are retained for the temperature ensemble after variable-availability filtering. Empirical Quantile Mapping (EQM) is applied per calendar month to correct the bias between model and observation. Bias-corrected precipitation feeds the SPI projection; bias-corrected temperature feeds a Hargreaves PET projection that, together with the projected precipitation, drives the scPDSI projection under both SSP scenarios.

A frozen-calibration scPDSI variant is reported as a sensitivity check; the headline projection rests on the continuous self-calibration run.

## 7. Repository structure

```
drought-modeling-probabilistic-forecasting-chelif/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── environment.yml
├── MANIFEST.md
│
├── notebooks/
│   ├── E2_data_audit_and_structuring.ipynb
│   ├── E3_data_preprocessing_and_station_selection.ipynb
│   ├── E4_spi_construction_and_validation.ipynb
│   ├── E9_stationarity_and_nspi.ipynb
│   ├── M1_spi_bayesian_AR1.Rmd
│   ├── M2_scpdsi_arima_bootstrap.Rmd
│   └── M3_scpdsi_diagnostics.Rmd
│
├── scripts/
│   ├── scpdsi_pipeline/
│   │   ├── E1_ingest_ihfr_onm.py
│   │   ├── E5_build_basin_precip_master.py
│   │   ├── E6_download_era5land_cds.py
│   │   ├── E7_scpdsi_compute.R
│   │   └── E8_scpdsi_pipeline.py
│   └── projections/
│       ├── R1–R16  (CMIP6 download, bias correction, projection)
│       └── X1–X4   (decision summaries, geospatial export)
│
├── rendered/
│   └── HTML mirrors of the four jury-readable notebooks
│
├── environment/
│   ├── python_requirements.txt
│   └── R_sessionInfo.txt
│
└── docs/
    ├── methodology.md
    └── data_sources.md
```

The `notebooks/` folder contains the seven jury-readable computational documents. The `scripts/` folder contains the supporting data engineering and projection scripts. The `rendered/` folder gives a zero-installation way to read the four most cited notebooks. The `environment/` and root-level dependency files together describe the software stack used at submission time.

## 8. Reproducibility

Reproducibility scope: this repository ships **traceability and audit** of the computational pipeline rather than one-click re-execution. The raw climate archives that feed the pipeline (CMIP6 monthly outputs, ERA5-Land, TerraClimate, the in-situ station network) are not bundled here. See `docs/data_sources.md` for portal references and access instructions.

To reproduce the analysis on a new machine:

1. Clone the repository and create a Python environment from `requirements.txt` or `environment.yml`.
2. Set the environment variable `CHELIF_PROJECT_ROOT` to the local path that holds the raw NetCDFs, the basin shapefile, and the SPI master CSV.
3. Run the scripts in order: `scripts/scpdsi_pipeline/E1` → `E5` → `E6` → `E7` → `E8` for the data engineering layer; `notebooks/E2` → `E3` → `E4` → `E9` for SPI construction and stationarity; `notebooks/M1` → `M2` → `M3` for modelling; `scripts/projections/R1` → `R16` and `X1` → `X4` for the projection pipeline.

The R Markdown files (M1, M2, M3) are knitted with `rmarkdown::render()`. The R session record in `environment/R_sessionInfo.txt` captures the version of every R package used.

## 9. Main outputs and figures

The four Jupyter notebooks in `notebooks/` ship with their executed outputs and embedded figures. The HTML mirrors in `rendered/` reproduce the same content for browsers without a Jupyter installation. Key results cited in the thesis:

- 1990–2015 station selection: nine stations retained after the C1/C2/C3 eligibility filter; 1.08 % aggregate interpolation fraction (E.3).
- SPI construction: per-station Gamma fits with KS-test acceptance and drought-class frequencies consistent with the McKee threshold table (E.4).
- Stationarity audit: five of seven aggregate tests fail to reject stationarity; the residual signal localises to the 2006–2011 episode at SPI-6 and SPI-12 (E.9).
- Bayesian AR(1) forecasting: closed-form Student-t predictive intervals with horizon-wise CRPS reported against a SARIMA baseline (M1).
- CMIP6 projections: 14-model precipitation ensemble, 10-model temperature ensemble, EQM-corrected fields, SPI-6 and scPDSI projections through 2040 under SSP2-4.5 and SSP5-8.5 (R2 / R3 / R4 / R6 / R12).

## 10. Technologies

**Python.** NumPy, pandas, SciPy, statsmodels, scikit-learn, matplotlib, xarray, netCDF4, rasterio, geopandas, `climate-indices`, `cdsapi`, Jupyter.

**R.** `tidyverse`, `forecast`, `tseries`, `urca`, `scoringRules`, `scPDSI`, `MCMCpack`, `reticulate`, `rmarkdown`, `knitr`.

Full version pins are recorded in `environment/python_requirements.txt` and `environment/R_sessionInfo.txt`.

## 11. Academic context

This work was carried out as a final-year engineering thesis in Applied Mathematics at the National Higher School of Mathematics (NHSM), in collaboration with the Institut Hydrométéorologique de Formation et de Recherche (IHFR), under the supervision of Dr. Abdelillah OTMANE CHERIF. The thesis is defended in the 2025–2026 academic year.

## 12. Citation and contact

If you find this work useful, please cite the thesis. A formal citation block will be added once the thesis manuscript is archived.

For questions about the framework, methodological choices, or the data engineering layer, open an issue on this repository.

---

## Licence

This repository is released under the Creative Commons Attribution 4.0 International licence (CC-BY-4.0). See `LICENSE` for the full text.
