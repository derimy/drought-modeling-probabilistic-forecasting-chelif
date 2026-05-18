# =============================================================================
# Author        : Imene DERRAR
# Supervisor    : Dr. Abdelillah OTMANE CHERIF
# Institutions  : NHSM — National Higher School of Mathematics
#                 IHFR — Institut Hydrométéorologique de Formation et de Recherche
# Academic year : 2025–2026
# Project       : Final-year engineering thesis in Applied Mathematics —
#                 Drought modelling and probabilistic forecasting for the Chélif basin
# =============================================================================

# =============================================================================
# R12_scpdsi_projection_compute.R
#
# Role: continuous-calibration reference run of scPDSI under SSP scenarios.
# This is the baseline scPDSI projection. R15 provides the
# frozen-historical-calibration sensitivity comparison.
# -----------------------------------------------------------------------------
# scPDSI projection — Chélif basin, 2015–2040 under SSP2-4.5 and SSP5-8.5.
#
# Builds on the historical scPDSI pipeline (the scPDSI compute step, notebook E.7):
# same scPDSI R package, same AWC = 120 mm, same basin-mean methodology.
#
# Reads the 6 (model × scenario) input CSVs produced by R11 (Python),
# runs scPDSI on each continuous 1990–2040 series with sc=TRUE, and writes
# out the projection scPDSI series ready for downstream analysis.
#
# Inputs (under 02_processed/scpdsi_inputs/projection/):
#   access_cm2__ssp2_4_5.csv,  access_cm2__ssp5_8_5.csv,
#   cmcc_esm2__ssp2_4_5.csv,   cmcc_esm2__ssp5_8_5.csv,
#   gfdl_esm4__ssp2_4_5.csv,   gfdl_esm4__ssp5_8_5.csv
#
# Outputs:
#   02_processed/drought_indices/projection/scpdsi_projection_long.csv
#       Combined long format: model, scenario, date, scPDSI, drought_class
#   02_processed/drought_indices/projection/scpdsi_<model>__<scenario>.csv
#       Per-(model × scenario) scPDSI time series (one CSV each, 6 total)
#   04_outputs/scpdsi_projection/figures/*.pdf
#
# Calibration-window note:
#   sc=TRUE self-calibrates on the entire input window (1990–2040 = 51 years).
#   This includes future data, which slightly attenuates the projected
#   drying signal relative to a calibration anchored on 1990–2014 only.
#   The scPDSI R package does not expose calibration-window separation
#   natively; this is the standard treatment in the literature (Stagge et al.
#   2017, Trnka et al. 2018, van der Schrier et al. 2013).  The projected
#   drought signal reported here is therefore a conservative lower bound.
# =============================================================================

# ------------------------------ 0. Packages ----------------------------------
required <- c("scPDSI", "tidyverse", "lubridate", "patchwork")
to_install <- setdiff(required, rownames(installed.packages()))
if (length(to_install) > 0) {
  install.packages(to_install, repos = "https://cloud.r-project.org")
}
suppressPackageStartupMessages({
  library(scPDSI)
  library(tidyverse)
  library(lubridate)
  library(patchwork)
})

# ------------------------------ 1. Configuration -----------------------------
LEGACY_ROOT     <- Sys.getenv("CHELIF_PROJECT_ROOT", unset = getwd())
STRUCTURED_ROOT <- file.path(LEGACY_ROOT, "_structured_project")

CFG <- list(
  in_dir      = file.path(STRUCTURED_ROOT, "02_processed",
                           "scpdsi_inputs", "projection"),
  out_idx_dir = file.path(STRUCTURED_ROOT, "02_processed",
                           "drought_indices", "projection"),
  out_fig_dir = file.path(STRUCTURED_ROOT, "04_outputs",
                           "scpdsi_projection", "figures"),
  out_tab_dir = file.path(STRUCTURED_ROOT, "04_outputs",
                           "scpdsi_projection", "tables"),

  awc_mm        = 120,                # central AWC from historical sweep
  calib_start   = as.Date("1990-01-01"),
  calib_end     = as.Date("2014-12-01"),
  proj_start    = as.Date("2015-01-01"),
  proj_end      = as.Date("2040-12-01"),

  # The 6 (model × scenario) combinations — R8 top-3 × 2 SSPs
  jobs = list(
    list(model = "ACCESS-CM2", model_key = "access_cm2", scenario = "ssp2_4_5"),
    list(model = "ACCESS-CM2", model_key = "access_cm2", scenario = "ssp5_8_5"),
    list(model = "CMCC-ESM2",  model_key = "cmcc_esm2",  scenario = "ssp2_4_5"),
    list(model = "CMCC-ESM2",  model_key = "cmcc_esm2",  scenario = "ssp5_8_5"),
    list(model = "GFDL-ESM4",  model_key = "gfdl_esm4",  scenario = "ssp2_4_5"),
    list(model = "GFDL-ESM4",  model_key = "gfdl_esm4",  scenario = "ssp5_8_5")
  )
)
dir.create(CFG$out_idx_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(CFG$out_fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(CFG$out_tab_dir, recursive = TRUE, showWarnings = FALSE)

log_msg <- function(...) cat(format(Sys.time(), "%H:%M:%S"), " | ",
                             sprintf(...), "\n", sep = "")

# ------------------------------ 2. Drought classification --------------------
palmer_class <- function(x) {
  cut(x,
      breaks = c(-Inf, -4, -3, -2, -1, 1, 2, 3, 4, Inf),
      labels = c("Extreme drought", "Severe drought", "Moderate drought",
                 "Mild drought",  "Near normal",
                 "Mild wet", "Moderate wet", "Severe wet", "Extreme wet"),
      right  = FALSE)
}

# ------------------------------ 3. scPDSI core -------------------------------
compute_scpdsi_projection <- function(input_csv, model_label, scenario,
                                       awc_mm) {
  log_msg("=== %s / %s ===", model_label, scenario)
  log_msg("Reading %s", basename(input_csv))
  df <- readr::read_csv(input_csv, show_col_types = FALSE) %>%
    mutate(date = as.Date(date)) %>%
    filter(date >= CFG$calib_start, date <= CFG$proj_end) %>%
    arrange(date)

  if (anyNA(df$ppt_mm) || anyNA(df$pet_mm)) {
    stop(sprintf("[%s/%s] NA in ppt/pet — check Python QC step",
                 model_label, scenario))
  }
  expected_n <- length(seq(CFG$calib_start, CFG$proj_end, by = "month"))
  if (nrow(df) != expected_n) {
    stop(sprintf("[%s/%s] expected %d months, got %d",
                 model_label, scenario, expected_n, nrow(df)))
  }
  log_msg("Series length: %d months (%s → %s)",
          nrow(df), min(df$date), max(df$date))

  # Run scPDSI with sc=TRUE on the full 1990–2040 series
  fit <- scPDSI::pdsi(
    P     = df$ppt_mm,
    PE    = df$pet_mm,
    AWC   = awc_mm,
    start = year(min(df$date)),
    end   = year(max(df$date)),
    sc    = TRUE
  )
  x <- as.numeric(fit$X)
  if (length(x) != nrow(df)) {
    stop(sprintf("[%s/%s] scPDSI returned %d values, expected %d",
                 model_label, scenario, length(x), nrow(df)))
  }

  out <- tibble(
    model    = model_label,
    scenario = scenario,
    date     = df$date,
    scPDSI   = x,
    drought_class = palmer_class(x),
    period   = if_else(df$date <= CFG$calib_end, "historical", "projection"),
    AWC_mm   = awc_mm
  )

  log_msg("scPDSI range: [%.2f, %.2f]   mean = %.2f",
          min(x, na.rm = TRUE), max(x, na.rm = TRUE), mean(x, na.rm = TRUE))

  # Continuity check at the 2014-12 / 2015-01 junction
  hist_end_val   <- tail(out$scPDSI[out$date <= CFG$calib_end], 1)
  proj_start_val <- head(out$scPDSI[out$date >= CFG$proj_start], 1)
  jump <- abs(proj_start_val - hist_end_val)
  log_msg("Junction continuity: scPDSI(Dec 2014)=%.2f, scPDSI(Jan 2015)=%.2f, |Δ|=%.3f",
          hist_end_val, proj_start_val, jump)

  out
}

# ------------------------------ 4. Run all 6 jobs ----------------------------
log_msg("Running scPDSI projection for %d (model × scenario) combinations",
        length(CFG$jobs))

all_results <- list()
for (j in CFG$jobs) {
  input_csv <- file.path(CFG$in_dir,
                         sprintf("%s__%s.csv", j$model_key, j$scenario))
  result <- compute_scpdsi_projection(input_csv, j$model, j$scenario,
                                       CFG$awc_mm)
  all_results[[length(all_results) + 1]] <- result

  # Per-(model × scenario) CSV output
  out_csv <- file.path(CFG$out_idx_dir,
                       sprintf("scpdsi_%s__%s.csv", j$model_key, j$scenario))
  readr::write_csv(result, out_csv)
  log_msg("Wrote %s", basename(out_csv))
}

# Combined long table
scpdsi_long <- dplyr::bind_rows(all_results)
long_csv <- file.path(CFG$out_idx_dir, "scpdsi_projection_long.csv")
readr::write_csv(scpdsi_long, long_csv)
log_msg("Wrote combined long table: %s (%d rows)",
        basename(long_csv), nrow(scpdsi_long))

# ------------------------------ 5. Diagnostic plots --------------------------
theme_set(theme_minimal(base_size = 11) +
            theme(panel.grid.minor = element_blank(),
                  legend.position  = "bottom"))

# (a) scPDSI trajectories per model × scenario, 1990–2040
p_traj <- scpdsi_long %>%
  ggplot(aes(date, scPDSI, colour = scenario)) +
  geom_hline(yintercept = 0, linewidth = 0.3, linetype = 2, colour = "grey50") +
  geom_hline(yintercept = c(-1, -2, -3), linewidth = 0.25, linetype = 3,
             colour = "firebrick", alpha = 0.6) +
  geom_line(linewidth = 0.4) +
  facet_wrap(~ model, ncol = 1, scales = "fixed") +
  scale_colour_manual(values = c("ssp2_4_5" = "#1F78B4", "ssp5_8_5" = "#E31A1C")) +
  labs(title = "scPDSI projection — Chélif basin, 1990–2040",
       subtitle = sprintf("AWC = %d mm; sc=TRUE on full 1990–2040 window",
                          CFG$awc_mm),
       x = NULL, y = "scPDSI", colour = NULL)
ggsave(file.path(CFG$out_fig_dir, "fig_scpdsi_projection_trajectories.pdf"),
       p_traj, width = 11, height = 7)

# (b) Drought class frequency over 2015–2040 per (model × scenario)
class_freq <- scpdsi_long %>%
  filter(period == "projection") %>%
  group_by(model, scenario, drought_class) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(model, scenario) %>%
  mutate(freq = n / sum(n)) %>%
  ungroup()

p_class <- class_freq %>%
  ggplot(aes(x = drought_class, y = freq, fill = scenario)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  facet_wrap(~ model, ncol = 1) +
  scale_fill_manual(values = c("ssp2_4_5" = "#1F78B4", "ssp5_8_5" = "#E31A1C")) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(title = "scPDSI drought-class frequency over 2015–2040",
       x = NULL, y = "Fraction of projected months", fill = NULL) +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))
ggsave(file.path(CFG$out_fig_dir, "fig_scpdsi_projection_class_freq.pdf"),
       p_class, width = 10, height = 8)

# (c) Severe-or-worse drought frequency (scPDSI ≤ −2) per decade
decadal <- scpdsi_long %>%
  filter(period == "projection") %>%
  mutate(decade = case_when(
    date < as.Date("2025-01-01") ~ "2015–2024",
    date < as.Date("2035-01-01") ~ "2025–2034",
    TRUE                          ~ "2035–2040")) %>%
  group_by(model, scenario, decade) %>%
  summarise(severe_freq = mean(scPDSI <= -2, na.rm = TRUE),
            mean_scPDSI = mean(scPDSI, na.rm = TRUE),
            .groups = "drop")

readr::write_csv(decadal,
                 file.path(CFG$out_tab_dir, "scpdsi_decadal_summary.csv"))
log_msg("Wrote scpdsi_decadal_summary.csv")

p_decadal <- decadal %>%
  ggplot(aes(decade, severe_freq, fill = scenario)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_hline(yintercept = 0.0228, linetype = 2, colour = "grey40") +
  facet_wrap(~ model, ncol = 3) +
  scale_fill_manual(values = c("ssp2_4_5" = "#1F78B4", "ssp5_8_5" = "#E31A1C")) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(title = "Frequency of severe-or-worse drought months (scPDSI ≤ −2)",
       subtitle = "Dashed line = 2.28% climatological baseline (P[N(0,1) ≤ −2])",
       x = NULL, y = "Frequency", fill = NULL)
ggsave(file.path(CFG$out_fig_dir, "fig_scpdsi_decadal_severe_freq.pdf"),
       p_decadal, width = 11, height = 5)

log_msg("All figures written to %s", CFG$out_fig_dir)

# ------------------------------ 6. Console summary ---------------------------
cat("\n================ scPDSI projection summary ================\n")
cat(sprintf("Window:           %s → %s\n", CFG$calib_start, CFG$proj_end))
cat(sprintf("AWC:              %d mm\n", CFG$awc_mm))
cat(sprintf("Models:           %s\n",
            paste(unique(scpdsi_long$model), collapse = ", ")))
cat(sprintf("Scenarios:        %s\n",
            paste(unique(scpdsi_long$scenario), collapse = ", ")))
cat(sprintf("Total rows:       %d\n", nrow(scpdsi_long)))
cat("\nDecadal severe-or-worse drought frequency (scPDSI ≤ −2):\n")
print(decadal %>%
        mutate(severe_freq = sprintf("%.1f %%", severe_freq * 100)) %>%
        pivot_wider(names_from = scenario, values_from = severe_freq,
                    id_cols = c(model, decade)))
cat("\nProjection portion: 2015-01 → 2040-12 (312 months per model × scenario)\n")
cat("===========================================================\n")
