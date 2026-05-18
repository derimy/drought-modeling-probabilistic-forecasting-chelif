# Methodology — short reference

This file is a short, plain-English overview of the methodological choices used in the thesis. It is meant for someone landing on the repository for the first time. The thesis manuscript provides the full mathematical detail.

## Data engineering

Nine in-situ rain gauges from the IHFR / ONM network are retained for the 1990–2015 study window after a three-condition filter: at least 90 % completeness, no contiguous gap longer than three months, and geographical coverage of the basin. Linear interpolation fills the remaining short gaps; the aggregate interpolation fraction is 1.08 % of station-months. Provenance flags track which values are observed and which are interpolated.

The same nine-station record produces both indices used downstream. The basin-mean monthly precipitation series (`basin_precip_master_1990_2015.csv`) is the canonical precipitation input for SPI and for scPDSI.

## SPI

The Standardised Precipitation Index is computed at scales 3, 6 and 12 months. Each calendar month is fitted with a two-parameter Gamma distribution by maximum likelihood, on the strictly positive accumulations. A zero adjustment mixes a point mass at zero with the Gamma cumulative. The probit transform of the cumulative gives the SPI. KS goodness-of-fit and drought-class frequency tables validate the construction.

SPI-6 is the operational scale used for forecasting and projection. It captures the seasonal-to-medium-term water deficit relevant to reservoir inflow and summer water supply.

## scPDSI

The self-calibrating Palmer Drought Severity Index is computed via the `scPDSI` R package and the `climate_indices` Python package. Two forcing branches are produced:

- **TerraClimate (primary)**: native modified Penman–Monteith PET, ~4 km resolution, climate-informed downscaling.
- **ERA5-Land (validation)**: PET re-derived from monthly 2-m temperature via latitude-corrected Thornthwaite (1948), because the monthly ERA5-Land product does not expose Tmax and Tmin.

The same station-derived precipitation feeds both branches; only PET and AWC are taken from the gridded products. SPI and scPDSI are treated as separate stochastic processes at the data engineering layer.

## Stationarity audit

Mann–Kendall with Hamed–Rao autocorrelation correction and Benjamini–Hochberg FDR control, Pettitt change-point test, Levene/Brown–Forsythe variance test, and a non-stationary Gamma model with time-varying parameters fitted by L-BFGS-B. The seven-evidence aggregator reports a `sensitive` robustness label for the Chélif basin: the basin is statistically near-stationary at aggregate scale, with a localised residual signal at the 2006–2011 episode. A non-stationary SPI (NSPI) is constructed as a sensitivity branch only; the canonical stationary SPI remains the operational index used in the modelling and projection chapters.

## Probabilistic forecasting

A Bayesian AR(1) model is fitted to SPI-6 under a conjugate Normal–Inverse-Gamma prior. The predictive distribution is Student-t in closed form. Rolling-origin evaluation against a SARIMA baseline reports horizon-wise CRPS, 90 % interval coverage, and the Diebold–Mariano test with a Newey–West HAC variance estimator. The classical SARIMA grid is also reported for transparency, but Bayesian AR(1) is the operational probabilistic framework.

scPDSI is modelled by ARIMA with residual-bootstrap predictive simulation. Short-horizon forecasts are reported as the most reliable; uncertainty grows substantially beyond 3–6 months.

## CMIP6 projections

Fourteen CMIP6 models are evaluated against the in-situ precipitation record using a composite Taylor skill score, with a 70 % cycle-correlation guard and a 20 % bias guard. Ten temperature models pass the variable-availability filter. Empirical Quantile Mapping (EQM) is applied per calendar month to remove the bias between model and observation. Bias-corrected precipitation feeds the SPI projection; bias-corrected temperature feeds a Hargreaves PET projection that, with the projected precipitation, drives the scPDSI projection under SSP2-4.5 and SSP5-8.5 through 2040.

A frozen-historical-calibration scPDSI variant is reported as a sensitivity branch. The headline projection rests on the continuous self-calibration run.

## Honest scope

The framework provides probabilistic drought forecasts under stated assumptions. It does not claim to resolve trends with timescales longer than ~13 years (the 26-year window cannot). It does not claim that the CMIP6 ensemble fully samples climate uncertainty; the EQM bias correction is a first-order adjustment, not a truth generator. Short-horizon forecasts are calibrated; long-horizon scPDSI projections carry visible inter-model spread. These limitations are stated in the thesis text and in the conclusion of each modelling notebook.
