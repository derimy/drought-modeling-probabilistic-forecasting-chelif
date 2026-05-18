# =============================================================================
# Author        : Imene DERRAR
# Supervisor    : Abdelillah OTMANE CHERIF
# Institutions  : NHSM — National Higher School of Mathematics
#                 IHFR — Institut Hydrométéorologique de Formation et de Recherche
# Academic year : 2025–2026
# Project       : Final-year engineering thesis in Applied Mathematics —
#                 Drought modelling and probabilistic forecasting for the Chélif basin
# =============================================================================

# =============================================================================
# E7_scpdsi_compute.R
# -----------------------------------------------------------------------------
# Self-calibrating Palmer Drought Severity Index (scPDSI) — Chélif basin.
#
# Thesis  : "Estimation and Prediction of Drought Indices using Statistical and
#           Bayesian Time-Series Models."
#
# Design rules (post-review):
#   - SPI and scPDSI are TREATED AS SEPARATE STOCHASTIC PROCESSES.
#   - This script does NOT merge them.  No drought_indices_master.csv.
#   - Any join between SPI and scPDSI must happen explicitly inside a
#     downstream analysis notebook with a documented hypothesis.
#
# Alignment guarantee:
#   The SPI master CSV (spi_regional_series.csv) defines the canonical
#   monthly DatetimeIndex.  This script reads it at startup, derives every
#   subsequent date filter from it, and asserts identity before writing
#   scpdsi_master.csv. The date range is inherited from the SPI reference index.
#
# Inputs  : 02_processed/scpdsi_inputs/terraclimate_basin_monthly.csv
#           02_processed/scpdsi_inputs/era5land_basin_monthly.csv
#           03_outputs/02_spi_construction_and_validation/spi_regional_series.csv
#
# Outputs : 02_processed/drought_indices/scpdsi_master.csv
#           02_processed/drought_indices/scpdsi_awc_sensitivity.csv
#           02_processed/drought_indices/scpdsi_meta.json   <-- provenance
#           03_outputs/scpdsi/figures/*.pdf
#           03_outputs/scpdsi/tables/validation_metrics.csv
# =============================================================================

# ---------------------------- 0. Packages ------------------------------------
required <- c("scPDSI", "tidyverse", "lubridate", "zoo", "patchwork", "jsonlite")
to_install <- setdiff(required, rownames(installed.packages()))
if (length(to_install) > 0) {
  install.packages(to_install, repos = "https://cloud.r-project.org")
}
suppressPackageStartupMessages({
  library(scPDSI)
  library(tidyverse)
  library(lubridate)
  library(zoo)
  library(patchwork)
  library(jsonlite)
})

# ---------------------------- 1. Configuration -------------------------------
# All scPDSI inputs/outputs live under _structured_project/. Raw NetCDFs and the
# basin shapefile remain in LEGACY_ROOT/01_data_raw/ (they are large external
# data, not project-generated artifacts). Edit either constant if the project
# is moved.
# Resolved from the CHELIF_PROJECT_ROOT environment variable when set,
# otherwise from the current working directory.
LEGACY_ROOT     <- Sys.getenv("CHELIF_PROJECT_ROOT", unset = getwd())
STRUCTURED_ROOT <- file.path(LEGACY_ROOT, "_structured_project")

CFG <- list(
  # Canonical reference index — the SPI master defines the time axis.
  spi_master_csv = file.path(
    STRUCTURED_ROOT, "04_outputs", "02_spi_construction_and_validation",
    "spi_regional_series.csv"
  ),
  spi_date_col   = "date",       # change to whatever your SPI file uses

  # Physics
  awc_central_mm = 120,                          # SoilGrids basin mean
  awc_grid_mm    = c(90, 110, 120, 130, 150),    # +/- 25% sensitivity sweep
  lat_basin_deg  = 35.2,
  pet_method_era5  = paste0(
    "Thornthwaite (1948), latitude-corrected. ",
    "Used because the CDS reanalysis-era5-land-monthly-means product does not ",
    "expose monthly Tmax/Tmin (Hargreaves-Samani therefore not feasible from ",
    "this product without re-downloading hourly data)."
  ),
  pet_method_terra = "TerraClimate native modified Penman-Monteith",

  # Paths — all under STRUCTURED_ROOT
  in_dir         = file.path(STRUCTURED_ROOT, "02_processed", "scpdsi_inputs"),
  out_idx_dir    = file.path(STRUCTURED_ROOT, "02_processed", "drought_indices"),
  out_fig_dir    = file.path(STRUCTURED_ROOT, "04_outputs", "scpdsi", "figures"),
  out_tab_dir    = file.path(STRUCTURED_ROOT, "04_outputs", "scpdsi", "tables")
)
dir.create(CFG$out_idx_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(CFG$out_fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(CFG$out_tab_dir, recursive = TRUE, showWarnings = FALSE)

log_msg <- function(...) cat(format(Sys.time(), "%H:%M:%S"), " | ",
                             sprintf(...), "\n", sep = "")

# ---------------------------- 2. Load canonical SPI index --------------------
load_spi_index <- function(path, date_col) {
  if (!file.exists(path)) {
    stop(sprintf("SPI master CSV not found: %s\n", path),
         "This file is the canonical reference index for scPDSI alignment.")
  }
  df <- readr::read_csv(path, show_col_types = FALSE)
  if (date_col %in% names(df)) {
    idx <- as.Date(df[[date_col]])
  } else if (all(c("year", "month") %in% names(df))) {
    idx <- as.Date(sprintf("%04d-%02d-01", df$year, df$month))
  } else {
    stop(sprintf(
      "SPI master %s has neither '%s' nor (year, month). Columns: %s",
      path, date_col, paste(names(df), collapse = ", ")
    ))
  }
  idx <- as.Date(format(idx, "%Y-%m-01"))         # force month-start
  if (any(duplicated(idx))) stop("SPI master index has duplicate months")
  if (is.unsorted(idx))     stop("SPI master index is not sorted")
  expected <- seq(min(idx), max(idx), by = "month")
  if (!identical(idx, expected)) {
    miss <- setdiff(as.character(expected), as.character(idx))
    stop(sprintf("SPI master has gaps. Missing %d months, e.g. %s",
                 length(miss), paste(head(miss, 5), collapse = ", ")))
  }
  idx
}

SPI_INDEX <- load_spi_index(CFG$spi_master_csv, CFG$spi_date_col)
SPI_START <- min(SPI_INDEX)
SPI_END   <- max(SPI_INDEX)
log_msg("[align] SPI master defines canonical index: %s -> %s (%d months)",
        SPI_START, SPI_END, length(SPI_INDEX))

# ---------------------------- 3. Loaders -------------------------------------
load_forcing <- function(path, label) {
  if (!file.exists(path)) stop(sprintf("Forcing file missing: %s", path))
  df <- readr::read_csv(path, show_col_types = FALSE) %>%
    mutate(date = as.Date(date)) %>%
    filter(date >= SPI_START, date <= SPI_END) %>%
    arrange(date)

  # Alignment check against the SPI canonical index.
  if (!identical(df$date, SPI_INDEX)) {
    stop(sprintf(
      "[%s] monthly index does NOT match the SPI master.\n  SPI: %s -> %s (%d)\n  Got: %s -> %s (%d)",
      label, SPI_START, SPI_END, length(SPI_INDEX),
      min(df$date), max(df$date), nrow(df)
    ))
  }
  if (anyNA(df$ppt_mm) || anyNA(df$pet_mm)) {
    stop(sprintf("[%s] NA in ppt/pet - check Python QC step", label))
  }
  log_msg("[%s] forcings loaded and aligned (%d months)", label, nrow(df))
  df
}

# ---------------------------- 4. Core scPDSI step ----------------------------
compute_scpdsi <- function(df, awc_mm, label) {
  fit <- scPDSI::pdsi(
    P     = df$ppt_mm,
    PE    = df$pet_mm,
    AWC   = awc_mm,
    start = year(min(df$date)),
    end   = year(max(df$date)),
    sc    = TRUE
  )
  # scPDSI 0.1.3 may or may not expose PDSI (classical) as a full vector.
  # Take it only if length matches, else fall back to X.
  pdsi_classic <- tryCatch(as.numeric(fit$PDSI), error = function(e) NULL)
  if (is.null(pdsi_classic) || length(pdsi_classic) != length(df$date)) {
    pdsi_classic <- NA_real_
  }
  z_idx <- tryCatch(as.numeric(fit$Z), error = function(e) NA_real_)
  if (length(z_idx) != length(df$date)) z_idx <- NA_real_

  tibble(
    date         = df$date,
    scPDSI       = as.numeric(fit$X),
    PDSI_classic = pdsi_classic,
    Z_index      = z_idx,
    source       = label,
    AWC_mm       = awc_mm
  )
}

# ---------------------------- 5. Drought classification ----------------------
palmer_class <- function(x) {
  cut(x,
      breaks = c(-Inf, -4, -3, -2, -1, 1, 2, 3, 4, Inf),
      labels = c("Extreme drought", "Severe drought", "Moderate drought",
                 "Mild drought",  "Near normal",
                 "Mild wet", "Moderate wet", "Severe wet", "Extreme wet"),
      right  = FALSE)
}

# ---------------------------- 6. Run primary computations --------------------
log_msg("Loading forcings ...")
tc <- load_forcing(file.path(CFG$in_dir, "terraclimate_basin_monthly.csv"), "TerraClimate")
e5 <- load_forcing(file.path(CFG$in_dir, "era5land_basin_monthly.csv"),     "ERA5-Land")

log_msg("Computing scPDSI (TerraClimate + ERA5-Land, AWC = %d mm)", CFG$awc_central_mm)
scpdsi_tc <- compute_scpdsi(tc, CFG$awc_central_mm, "TerraClimate")
scpdsi_e5 <- compute_scpdsi(e5, CFG$awc_central_mm, "ERA5-Land")

scpdsi_master <- scpdsi_tc %>%
  select(date, scPDSI_TerraClimate = scPDSI) %>%
  inner_join(scpdsi_e5 %>% select(date, scPDSI_ERA5 = scPDSI), by = "date")

# Final identity check before writing.
stopifnot(identical(scpdsi_master$date, SPI_INDEX))
readr::write_csv(scpdsi_master, file.path(CFG$out_idx_dir, "scpdsi_master.csv"))
log_msg("Wrote scpdsi_master.csv (%d rows)", nrow(scpdsi_master))

# ---------------------------- 7. AWC sensitivity loop ------------------------
# Sweep AWC over BOTH PET methods so the variance-decomposition step
# (analysis notebook §8) can attribute total scPDSI variance to AWC vs PET
# vs residual. Output schema unchanged; PET method is encoded in `source`.
log_msg("Running AWC sensitivity sweep (TerraClimate + ERA5-Land)")
sens_tc <- map_dfr(CFG$awc_grid_mm, function(a) {
  compute_scpdsi(tc, awc_mm = a, label = paste0("TC_AWC_",   a)) %>%
    dplyr::mutate(pet_method = "TerraClimate")
})
sens_e5 <- map_dfr(CFG$awc_grid_mm, function(a) {
  compute_scpdsi(e5, awc_mm = a, label = paste0("ERA5_AWC_", a)) %>%
    dplyr::mutate(pet_method = "ERA5-Land")
})
sens <- dplyr::bind_rows(sens_tc, sens_e5)
readr::write_csv(sens, file.path(CFG$out_idx_dir, "scpdsi_awc_sensitivity.csv"))
log_msg("Wrote scpdsi_awc_sensitivity.csv (%d rows, %d AWC values x 2 PET methods)",
        nrow(sens), length(CFG$awc_grid_mm))

# ---------------------------- 8. Metadata sidecar ----------------------------
meta <- list(
  index = list(
    canonical_source = CFG$spi_master_csv,
    start            = format(SPI_START, "%Y-%m-%d"),
    end              = format(SPI_END,   "%Y-%m-%d"),
    n_months         = length(SPI_INDEX),
    frequency        = "monthly (month-start)"
  ),
  datasets = list(
    primary = list(
      name = "TerraClimate",
      version = "v4 (Climatology Lab)",
      resolution = "1/24 deg (~4 km)",
      pet_method = CFG$pet_method_terra
    ),
    validation = list(
      name = "ERA5-Land",
      version = "Copernicus C3S monthly means",
      resolution = "0.1 deg (~9 km)",
      pet_method = CFG$pet_method_era5
    )
  ),
  awc = list(
    central_mm        = CFG$awc_central_mm,
    sensitivity_grid  = CFG$awc_grid_mm,
    rationale         = "+/- 25% around SoilGrids basin-mean estimate"
  ),
  basin = list(
    name = "Chelif",
    centroid_lat_deg = CFG$lat_basin_deg,
    polygon_source   = "HydroBASINS, MAIN_BAS = 1110030450"
  ),
  assumptions = list(
    "Spatial aggregation: forcings are area-weighted-averaged BEFORE scPDSI.",
    "Self-calibrating PDSI (sc = TRUE) re-derives K and duration factors.",
    "AWC is treated as time-invariant; uncertainty bounded by the sweep above.",
    "Stationarity is assumed within the calibration window."
  ),
  alignment = list(
    rule        = "scPDSI inherits the SPI master DatetimeIndex; the date range is inherited from the SPI reference index.",
    enforcement = "stopifnot(identical(scpdsi$date, SPI_INDEX)) before write."
  ),
  pipeline = list(
    spi_treated_as = "separate stochastic process",
    merge_policy   = "no merge with SPI at the data-engineering layer",
    join_policy    = "any joint analysis must be performed in a downstream notebook with an explicit hypothesis"
  ),
  produced_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
)
write_json(meta, file.path(CFG$out_idx_dir, "scpdsi_meta.json"),
           pretty = TRUE, auto_unbox = TRUE)
log_msg("Wrote scpdsi_meta.json")

# ---------------------------- 9. Validation metrics --------------------------
log_msg("Computing TerraClimate-vs-ERA5 validation metrics")
v <- scpdsi_master %>%
  mutate(diff = scPDSI_TerraClimate - scPDSI_ERA5,
         class_tc = palmer_class(scPDSI_TerraClimate),
         class_e5 = palmer_class(scPDSI_ERA5))

pearson_r        <- cor(v$scPDSI_TerraClimate, v$scPDSI_ERA5, use = "complete.obs")
rmse_idx         <- sqrt(mean(v$diff^2, na.rm = TRUE))
class_agreement  <- mean(v$class_tc == v$class_e5, na.rm = TRUE)

severe_tc <- v$scPDSI_TerraClimate <= -2
severe_e5 <- v$scPDSI_ERA5         <= -2
hit_rate_d2 <- mean(severe_tc == severe_e5, na.rm = TRUE)

metrics <- tibble(
  metric = c("Pearson r", "RMSE (idx units)",
             "Drought-class agreement (D0-D4)",
             "Hit rate on D2+ events"),
  value  = c(pearson_r, rmse_idx, class_agreement, hit_rate_d2),
  threshold = c(">= 0.85", "<= 0.6", ">= 0.75", ">= 0.80"),
  passes = c(pearson_r        >= 0.85,
             rmse_idx         <= 0.60,
             class_agreement  >= 0.75,
             hit_rate_d2      >= 0.80)
)
readr::write_csv(metrics, file.path(CFG$out_tab_dir, "validation_metrics.csv"))
log_msg("Validation: r=%.3f  RMSE=%.3f  class-agreement=%.2f  hit-rate(D2+)=%.2f",
        pearson_r, rmse_idx, class_agreement, hit_rate_d2)

# ---------------------------- 10. Diagnostic figures -------------------------
theme_set(theme_minimal(base_size = 11) +
            theme(panel.grid.minor = element_blank(),
                  legend.position  = "bottom"))

p1 <- scpdsi_master %>%
  pivot_longer(starts_with("scPDSI_"), names_to = "source", values_to = "value") %>%
  mutate(source = recode(source,
                         scPDSI_TerraClimate = "TerraClimate",
                         scPDSI_ERA5         = "ERA5-Land")) %>%
  ggplot(aes(date, value, colour = source)) +
  annotate("rect", xmin = SPI_START, xmax = SPI_END,
           ymin = -2, ymax = -1, alpha = 0.10, fill = "#E08E79") +
  annotate("rect", xmin = SPI_START, xmax = SPI_END,
           ymin = -3, ymax = -2, alpha = 0.15, fill = "#C0504D") +
  annotate("rect", xmin = SPI_START, xmax = SPI_END,
           ymin = -Inf, ymax = -3, alpha = 0.18, fill = "#7E0000") +
  geom_hline(yintercept = 0, linewidth = 0.3, linetype = 2, colour = "grey40") +
  geom_line(linewidth = 0.5) +
  scale_colour_manual(values = c("TerraClimate" = "#1F3864", "ERA5-Land" = "#C00000")) +
  labs(title = "scPDSI - Chelif basin",
       subtitle = "Two independent gridded products, common monthly axis (inherited from SPI)",
       x = NULL, y = "scPDSI", colour = NULL)

ggsave(file.path(CFG$out_fig_dir, "fig_scpdsi_timeseries.pdf"),
       p1, width = 10, height = 4.5)

p2 <- sens %>%
  ggplot(aes(date, scPDSI, group = AWC_mm,
             colour = factor(AWC_mm))) +
  geom_line(linewidth = 0.4, alpha = 0.85) +
  scale_colour_brewer(palette = "RdYlBu", direction = -1) +
  labs(title = "TerraClimate scPDSI - AWC sensitivity",
       subtitle = "AWC sweep +/- 25% around the SoilGrids central estimate",
       x = NULL, y = "scPDSI",
       colour = "AWC (mm)")

ggsave(file.path(CFG$out_fig_dir, "fig_scpdsi_awc_sensitivity.pdf"),
       p2, width = 10, height = 4)

p3a <- ggplot(v, aes(scPDSI_TerraClimate, scPDSI_ERA5)) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, colour = "grey50") +
  geom_point(alpha = 0.5, size = 0.9, colour = "#1F3864") +
  labs(title = "TerraClimate vs ERA5-Land",
       subtitle = sprintf("Pearson r = %.3f", pearson_r),
       x = "scPDSI (TerraClimate)", y = "scPDSI (ERA5-Land)")

p3b <- ggplot(v, aes(date, diff)) +
  geom_hline(yintercept = 0, colour = "grey50") +
  geom_line(linewidth = 0.4, colour = "#C00000") +
  labs(title = "Residuals (TerraClimate - ERA5-Land)",
       subtitle = sprintf("RMSE = %.3f", rmse_idx),
       x = NULL, y = "Delta scPDSI")

ggsave(file.path(CFG$out_fig_dir, "fig_validation_tc_vs_era5.pdf"),
       p3a + p3b + plot_layout(ncol = 2), width = 11, height = 4)

episodes <- function(s, dates, threshold = -1) {
  rle_obj <- rle(s <= threshold)
  ends   <- cumsum(rle_obj$lengths)
  starts <- ends - rle_obj$lengths + 1
  tibble(
    start  = dates[starts][rle_obj$values],
    end    = dates[ends][rle_obj$values],
    length = rle_obj$lengths[rle_obj$values]
  )
}

ep_tc <- episodes(scpdsi_master$scPDSI_TerraClimate, scpdsi_master$date) %>%
  mutate(source = "TerraClimate")
ep_e5 <- episodes(scpdsi_master$scPDSI_ERA5,         scpdsi_master$date) %>%
  mutate(source = "ERA5-Land")
ep <- bind_rows(ep_tc, ep_e5) %>% filter(length >= 3)

p4 <- ggplot(ep, aes(xmin = start, xmax = end, ymin = 0, ymax = length,
                     fill = source)) +
  geom_rect(alpha = 0.85) +
  scale_fill_manual(values = c("TerraClimate" = "#1F3864", "ERA5-Land" = "#C00000")) +
  facet_wrap(~ source, ncol = 1) +
  labs(title = "Drought episodes (scPDSI <= -1, length >= 3 months)",
       x = NULL, y = "Episode length (months)") +
  theme(legend.position = "none")

ggsave(file.path(CFG$out_fig_dir, "fig_scpdsi_drought_episodes.pdf"),
       p4, width = 10, height = 5)

log_msg("All figures written to %s", CFG$out_fig_dir)

# ---------------------------- 11. Console summary ----------------------------
cat("\n=================== scPDSI computation summary ===================\n")
cat(sprintf("Window         : %s -> %s   (%d months, inherited from SPI master)\n",
            SPI_START, SPI_END, length(SPI_INDEX)))
cat(sprintf("AWC (central)  : %d mm    AWC sweep: %s\n",
            CFG$awc_central_mm, paste(CFG$awc_grid_mm, collapse = ", ")))
cat(sprintf("Pearson r      : %.3f   (target >= 0.85)\n", pearson_r))
cat(sprintf("RMSE           : %.3f   (target <= 0.60)\n", rmse_idx))
cat(sprintf("Class agreement: %.2f   (target >= 0.75)\n", class_agreement))
cat(sprintf("Hit rate (D2+) : %.2f   (target >= 0.80)\n", hit_rate_d2))
cat("Pipeline policy: SPI and scPDSI are SEPARATE processes.  No merge file.\n")
cat("===================================================================\n")
