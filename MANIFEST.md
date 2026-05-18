# MANIFEST — Provenance Index

Every file in this package is referenced from the thesis appendix by the identifier in the **Index** column. Filenames are stable: the names in this table are the names the thesis cites.

## Chapter 3 — SPI construction (Jupyter notebooks, outputs embedded)

| Index | File | Role |
|:-----:|:-----|:-----|
| E.2 | `chapter3_spi/E2_data_audit_and_structuring.ipynb` | Audit and structuring of the raw IHFR / ONM station network. |
| E.3 | `chapter3_spi/E3_data_preprocessing_and_station_selection.ipynb` | Quality control, gap-filling, station selection (1990–2015). |
| E.4 | `chapter3_spi/E4_spi_construction_and_validation.ipynb` | SPI-3 / SPI-6 / SPI-12 construction and validation. |
| E.9 | `chapter3_spi/E9_stationarity_and_nspi.ipynb` | Stationarity diagnostics and non-stationary SPI sensitivity. |

## Chapter 3 — scPDSI input pipeline (Python + R)

| Index | File | Role |
|:-----:|:-----|:-----|
| E.1 | `chapter3_scpdsi/E1_ingest_ihfr_onm.py` | Ingest raw IHFR and ONM monthly station files. |
| E.5 | `chapter3_scpdsi/E5_build_basin_precip_master.py` | Build the basin-aggregated precipitation master series. |
| E.6 | `chapter3_scpdsi/E6_download_era5land_cds.py` | Download ERA5-Land monthly forcings (Copernicus CDS). |
| E.7 | `chapter3_scpdsi/E7_scpdsi_compute.R` | Compute scPDSI with frozen calibration via `climate_indices` (R↔Python). |
| E.8 | `chapter3_scpdsi/E8_scpdsi_pipeline.py` | End-to-end Python wrapper orchestrating the scPDSI pipeline. |

## Chapter 4 — Modelling (R Markdown)

| Index | File | Role |
|:-----:|:-----|:-----|
| M.1 | `chapter4_modeling/M1_spi_bayesian_AR1.Rmd` | Bayesian AR(1) under Normal-Inverse-Gamma prior, applied to SPI. |
| M.2 | `chapter4_modeling/M2_scpdsi_arima_bootstrap.Rmd` | Residual-bootstrap ARIMA(1,0,0) for scPDSI. |
| M.3 | `chapter4_modeling/M3_scpdsi_diagnostics.Rmd` | Stationarity, residual, and structural diagnostics for the scPDSI model. |

## Chapter 5 — Projections (Python + R, in pipeline order)

| Index | File | Role |
|:-----:|:-----|:-----|
| R.1  | `chapter5_projections/R1_cmip6_precip_download.py` | CMIP6 precipitation download (EC-Earth3-Veg-LR, EC-Earth3-CC, CESM2). |
| R.2  | `chapter5_projections/R2_cmip6_taylor_evaluation.py` | Taylor-diagram evaluation of CMIP6 precipitation models. |
| R.3  | `chapter5_projections/R3_bias_correction.py` | Empirical Quantile Mapping for precipitation. |
| R.4  | `chapter5_projections/R4_spi_projection.py` | Project SPI under SSP2-4.5 and SSP5-8.5. |
| R.5  | `chapter5_projections/R5_spi_drought_analysis.py` | Drought frequency, duration, and severity statistics from projected SPI. |
| R.6  | `chapter5_projections/R6_bayesian_projection.py` | Bayesian predictive intervals applied to projected SPI. |
| R.7  | `chapter5_projections/R7_cmip6_temperature_download.py` | CMIP6 temperature download (ACCESS-CM2, CMCC-ESM2, GFDL-ESM4). |
| R.8  | `chapter5_projections/R8_temperature_taylor_evaluation.py` | Taylor-diagram evaluation of CMIP6 temperature models. |
| R.9  | `chapter5_projections/R9_temperature_bias_correction.py` | EQM bias correction for temperature. |
| R.10 | `chapter5_projections/R10_pet_hargreaves_projection.py` | Project PET via Hargreaves from corrected temperature. |
| R.11 | `chapter5_projections/R11_scpdsi_build_inputs.py` | Assemble projected P / PET inputs for the scPDSI projection. |
| R.12 | `chapter5_projections/R12_scpdsi_projection_compute.R` | Compute projected scPDSI series via `climate_indices`. |
| R.13 | `chapter5_projections/R13_scpdsi_projection_analysis.py` | Analysis of projected scPDSI series. |
| R.14 | `chapter5_projections/R14_scpdsi_gap_closure.py` | Closure of frozen-vs-continuous calibration gap. |
| R.15 | `chapter5_projections/R15_scpdsi_frozen_calibration.py` | Frozen-calibration scPDSI variant. |
| R.16 | `chapter5_projections/R16_chapter5_figures.py` | Final figures for Chapter 5. |

## Chapter 6 — Export and decision summaries

| Index | File | Role |
|:-----:|:-----|:-----|
| X.1 | `chapter5_projections/X1_decision_summary.py` | Decision summary across scenarios and models. |
| X.2 | `chapter5_projections/X2_arcgis_export.py` | Export geo-referenced rasters and shapefiles for ArcGIS. |
| X.3 | `chapter5_projections/X3_drought_frequency_per_station.py` | Per-station drought frequency tables. |
| X.4 | `chapter5_projections/X4_per_model_drought_frequency.py` | Per-CMIP6-model drought frequency tables. |

## Static rendered exhibits

| File | Source notebook |
|:-----|:----------------|
| `rendered_appendix/E2_data_audit_and_structuring.html`            | E.2 |
| `rendered_appendix/E3_data_preprocessing_and_station_selection.html` | E.3 |
| `rendered_appendix/E4_spi_construction_and_validation.html`       | E.4 |
| `rendered_appendix/E9_stationarity_and_nspi.html`                 | E.9 |
