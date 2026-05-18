# Data sources

The raw climate archives that feed the pipeline are **not bundled** with this repository. They exceed the storage envelope of a public Git repo and are subject to portal-specific access terms. Each source is documented below with the portal, the variables retrieved, and the time window used.

## 1. In-situ precipitation — IHFR / ONM network

- **Source**: Institut Hydrométéorologique de Formation et de Recherche (IHFR), Algiers, and Office National de la Météorologie (ONM), Algeria.
- **Variable**: monthly precipitation (mm/month).
- **Period covered**: roughly 1950s through 2019, with archive-dependent gaps.
- **Window used in this work**: 1990-01 to 2015-12, monthly resolution.
- **Stations retained**: 9 after the C1/C2/C3 eligibility filter (see notebook E.3).
- **Access**: data are obtained through institutional channels with IHFR and ONM. They are not publicly redistributable.
- **Format in the pipeline**: long-format CSV with columns `master_station_id`, `date`, `precip_mm`, `precip_filled`, `was_interpolated`, `value_source`, `source_dataset`.

## 2. ERA5-Land monthly reanalysis

- **Source**: Copernicus Climate Data Store (CDS), product `reanalysis-era5-land-monthly-means`.
- **Portal**: https://cds.climate.copernicus.eu/
- **Variables retrieved**: `total_precipitation`, `2m_temperature`. (Tmax and Tmin are not exposed by the monthly product; PET is derived locally via Thornthwaite. See `R6_download_era5land_cds.py`.)
- **Period**: 1989–2019.
- **Bounding box**: [36.5 °N, 0.0 °E, 34.0 °N, 3.5 °E].
- **Access**: free CDS account, accept ERA5-Land terms once on the dataset page.

## 3. TerraClimate

- **Source**: Climatology Lab, University of California Merced.
- **Portal**: https://www.climatologylab.org/terraclimate.html
- **Variables retrieved**: `ppt`, `pet`, `tmean`, `tmax`, `tmin`, `aet`, `def`.
- **Period**: 1990–2015 for the scPDSI historical run.
- **Resolution**: ~4 km, climate-informed downscaling of CRU TS and CRUNCEP. Native PET via modified Penman–Monteith.

## 4. CMIP6 monthly outputs

- **Source**: Copernicus Climate Data Store (CDS), product `projections-cmip6`.
- **Portal**: https://cds.climate.copernicus.eu/
- **Variables retrieved**: precipitation (`pr`) for the 14-model precipitation ensemble; 2-m temperature variables (`tas`, `tasmax`, `tasmin`) for the 10-model temperature ensemble.
- **Experiments**: `historical` (1990–2014), `ssp2_4_5` and `ssp5_8_5` (2015–2040).
- **Models — precipitation ensemble (14)**: ACCESS-CM2, CESM2, CMCC-ESM2, CNRM-CM6-1, EC-Earth3-CC, EC-Earth3-Veg-LR, GFDL-ESM4, HadGEM3-GC31-LL, INM-CM5-0, IPSL-CM6A-LR, MIROC6, MPI-ESM1-2-LR, MRI-ESM2-0, NorESM2-MM.
- **Models — temperature ensemble (10)**: ACCESS-CM2, CMCC-ESM2, CNRM-CM6-1, GFDL-ESM4, HadGEM3-GC31-LL, INM-CM5-0, IPSL-CM6A-LR, MIROC6, MPI-ESM1-2-LR, MRI-ESM2-0.
- **Access**: free CDS account, accept the relevant CMIP6 dataset terms.

## 5. Basin geometry

- **Source**: shapefile of the Chélif basin polygon used for spatial clipping.
- **CRS**: EPSG:4326.
- **Use**: cosine-latitude-weighted basin-mean of gridded variables in the scPDSI pipeline.

## 6. Local environment variable

The Python and R scripts honour `CHELIF_PROJECT_ROOT` as the data root. Set this environment variable to the directory that contains `01_data/`, `02_processed/`, and the basin shapefile before running any data-loading script.

```bash
# Bash / Linux / macOS
export CHELIF_PROJECT_ROOT=/path/to/chelif/data

# PowerShell / Windows
$env:CHELIF_PROJECT_ROOT = "C:\path\to\chelif\data"
```

If the variable is not set, scripts fall back to the current working directory.

## 7. Citation chain for the data

When citing the analysis, please also cite the upstream data providers:
- ERA5-Land: Muñoz Sabater (2019), C3S.
- TerraClimate: Abatzoglou et al. (2018), Scientific Data.
- CMIP6: Eyring et al. (2016), GMD.
- IHFR / ONM: institutional release (not publicly cited).
