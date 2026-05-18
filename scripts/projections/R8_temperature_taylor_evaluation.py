"""
Author        : Imene DERRAR
Supervisor    : Abdelillah OTMANE CHERIF
Institutions  : NHSM — National Higher School of Mathematics
                IHFR — Institut Hydrométéorologique de Formation et de Recherche
Academic year : 2025–2026
Project       : Final-year engineering thesis in Applied Mathematics —
                Drought modelling and probabilistic forecasting for the Chélif basin

--------------------------------------------------------------------------------

R8 — CMIP6 temperature evaluation against basin-mean references
==========================================================================

Three Taylor diagrams (one per temperature variable) and a composite skill
score that ranks the 12 candidate models for the scPDSI projection.

Pipeline
--------
1.  Load the two basin-mean reference series produced by the historical
    scPDSI pipeline:
        - ERA5-Land     (tmean_C)            → reference for `tas`
        - TerraClimate  (tmax_C, tmin_C)     → references for `tasmax`, `tasmin`
2.  For each of the 10 CMIP6 temperature-ensemble models × 3 variables, load the
    monthly historical NetCDF, aggregate to basin-mean with cosine-latitude
    weighting (the same method as the scPDSI pipeline (notebook E.8)),
    convert Kelvin → Celsius.
3.  Per (model × variable): compute Pearson ρ, σ-ratio, centred RMSE,
    mean bias, annual-cycle correlation.  Aggregate into the Taylor
    skill score S²_Taylor.
4.  Combine into one composite score S = mean of variable-specific
    S²_Taylor × bias / cycle weights, with the same recipe used in the
    SPI Taylor evaluation (R2).
5.  Plot three Taylor panels (tas, tasmax, tasmin) and an overall ranking
    bar chart.  Write the skill table to CSV.

The top three models on the composite score are the candidates for the
scPDSI SSP download and bias correction stage.

Inputs
------
    02_processed/scpdsi_inputs/era5land_basin_monthly.csv
    02_processed/scpdsi_inputs/terraclimate_basin_monthly.csv
    01_data/cmip6/historical/<model>/<var>/<tas|tasmax|tasmin>_*.nc

Outputs (under 04_outputs/R8_temperature_taylor/)
    tables/skill_metrics_temperature.csv     long format, 1 row/(model × var)
    tables/skill_metrics_composite.csv       1 row/model, composite + ranking
    tables/best_three_temperature.csv        top-3 models retained
    figures/taylor_temperature.png           3 Taylor panels
    figures/composite_ranking.png            bar chart of composite S by model
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist import floating_axes, grid_finder

# --------------------------------------------------------------------------- #
# 0. CONFIGURATION
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    bbox_nwse: Tuple[float, float, float, float] = (36.5, 0.0, 34.0, 3.5)

    eval_start: str = "1990-01"
    eval_end:   str = "2014-12"

    # Variable -> (CMIP6 short name, reference dataset, reference column)
    variables: Dict[str, Tuple[str, str]] = field(default_factory=lambda: {
        "tas":    ("ERA5-Land",    "tmean_C"),
        "tasmax": ("TerraClimate", "tmax_C"),
        "tasmin": ("TerraClimate", "tmin_C"),
    })

    # Composite skill weights (apply within each variable, then average across vars)
    w_taylor: float = 0.60
    w_bias:   float = 0.20
    w_cycle:  float = 0.20

    # Per-(model × variable) folder mapping.  TEN models retained for the
    # temperature Taylor evaluation — all with NATIVE successful downloads
    # of `tas`, `tasmax`, `tasmin` from the CDS CMIP6 catalogue.
    #
    # A few candidate models had to be excluded because the SSP runs were
    # not available at the required monthly resolution for our subsetting
    # box across all three temperature variables. The temperature ensemble
    # for the scPDSI projection is therefore based on 10 structurally
    # diverse models with no missing variable.
    models: Dict[str, Dict[str, str | None]] = field(default_factory=lambda: {
        "ACCESS-CM2":       {"tas": "access_cm2",       "tasmax": "access_cm2",       "tasmin": "access_cm2"},
        "CMCC-ESM2":        {"tas": "cmcc_esm2",        "tasmax": "cmcc_esm2",        "tasmin": "cmcc_esm2"},
        "CNRM-CM6-1":       {"tas": "cnrm_cm6_1",       "tasmax": "cnrm_cm6_1",       "tasmin": "cnrm_cm6_1"},
        "GFDL-ESM4":        {"tas": "gfdl_esm4",        "tasmax": "gfdl_esm4",        "tasmin": "gfdl_esm4"},
        "HadGEM3-GC31-LL":  {"tas": "hadgem3_gc31_ll",  "tasmax": "hadgem3_gc31_ll",  "tasmin": "hadgem3_gc31_ll"},
        "INM-CM5-0":        {"tas": "inm_cm5_0",        "tasmax": "inm_cm5_0",        "tasmin": "inm_cm5_0"},
        "IPSL-CM6A-LR":     {"tas": "ipsl_cm6a_lr",     "tasmax": "ipsl_cm6a_lr",     "tasmin": "ipsl_cm6a_lr"},
        "MIROC6":           {"tas": "miroc6",           "tasmax": "miroc6",           "tasmin": "miroc6"},
        "MPI-ESM1-2-LR":    {"tas": "mpi_esm1_2_lr",    "tasmax": "mpi_esm1_2_lr",    "tasmin": "mpi_esm1_2_lr"},
        "MRI-ESM2-0":       {"tas": "mri_esm2_0",       "tasmax": "mri_esm2_0",       "tasmin": "mri_esm2_0"},
    })

    project_root: Path = Path(__file__).resolve().parents[3]
    era5_csv:     Path = field(init=False)
    terra_csv:    Path = field(init=False)
    cmip6_dir:    Path = field(init=False)
    out_dir:      Path = field(init=False)

    def __post_init__(self) -> None:
        self.era5_csv  = (self.project_root / "02_processed" / "scpdsi_inputs"
                          / "era5land_basin_monthly.csv")
        self.terra_csv = (self.project_root / "02_processed" / "scpdsi_inputs"
                          / "terraclimate_basin_monthly.csv")
        self.cmip6_dir = self.project_root / "01_data" / "cmip6" / "historical"
        self.out_dir   = self.project_root / "04_outputs" / "R8_temperature_taylor"
        (self.out_dir / "tables").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "figures").mkdir(parents=True, exist_ok=True)


CFG = Config()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("r8_temp_taylor")


# --------------------------------------------------------------------------- #
# 1. UTILITIES (basin-mean aggregation, cftime handling)
# --------------------------------------------------------------------------- #

def _normalize_dim_names(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    if "latitude"   in ds.coords: rename["latitude"]  = "lat"
    if "longitude"  in ds.coords: rename["longitude"] = "lon"
    if "valid_time" in ds.coords and "time" not in ds.coords:
        rename["valid_time"] = "time"
    return ds.rename(rename) if rename else ds


def _wrap_lon_0_360_to_180(ds: xr.Dataset) -> xr.Dataset:
    if "lon" not in ds.coords:
        return ds
    if float(ds["lon"].max()) > 180.0:
        new_lon = (((ds["lon"] + 180.0) % 360.0) - 180.0)
        ds = ds.assign_coords(lon=new_lon).sortby("lon")
    return ds


def _clip_box(ds: xr.Dataset, nwse) -> xr.Dataset:
    n, w, s, e = nwse
    lat = ds["lat"]
    lat_slice = slice(n, s) if float(lat[0]) > float(lat[-1]) else slice(s, n)
    return ds.sel(lat=lat_slice, lon=slice(w, e))


def _basin_mean(da: xr.DataArray) -> pd.Series:
    """Cosine-latitude weighted basin mean over (lat, lon), returns a
    monthly pandas Series indexed at month-start."""
    weights = np.cos(np.deg2rad(da["lat"]))
    weights = weights.where(weights > 0, 0)
    s = da.weighted(weights).mean(dim=("lat", "lon")).to_pandas()
    s = _force_dt_index(s)
    return s


def _force_dt_index(s: pd.Series) -> pd.Series:
    """Force a month-start pandas DatetimeIndex, robust to cftime objects."""
    if not isinstance(s.index, pd.DatetimeIndex):
        yms = []
        for t in s.index:
            try:
                yms.append((int(t.year), int(t.month)))
            except AttributeError:
                ts = pd.Timestamp(t)
                yms.append((ts.year, ts.month))
        s.index = pd.to_datetime([f"{y:04d}-{m:02d}-01" for y, m in yms])
    s.index = s.index.to_period("M").to_timestamp()
    return s


# --------------------------------------------------------------------------- #
# 2. LOAD REFERENCES
# --------------------------------------------------------------------------- #

def load_references(cfg: Config) -> Dict[str, pd.Series]:
    """Return a dict mapping variable name -> reference pandas Series."""
    era5 = pd.read_csv(cfg.era5_csv,  parse_dates=["date"]).set_index("date")
    terra = pd.read_csv(cfg.terra_csv, parse_dates=["date"]).set_index("date")
    era5.index  = pd.to_datetime(era5.index).to_period("M").to_timestamp()
    terra.index = pd.to_datetime(terra.index).to_period("M").to_timestamp()
    out: Dict[str, pd.Series] = {}
    for var, (src, col) in cfg.variables.items():
        df = era5 if src == "ERA5-Land" else terra
        s = df[col].astype(float)
        s = s.loc[cfg.eval_start: cfg.eval_end].dropna()
        out[var] = s
        log.info("[ref] %-7s -> %-13s.%-8s  n=%d  mean=%.2f °C",
                 var, src, col, len(s), s.mean())
    return out


# --------------------------------------------------------------------------- #
# 3. LOAD MODEL FIELD AND COMPUTE BASIN-MEAN
# --------------------------------------------------------------------------- #

# CMIP6 publishes these variables with these short-names in the NetCDF
# `data_vars`.  The download script saves them under sub-folders named by
# the variable short name (tas / tasmax / tasmin).
_CMIP6_NETCDF_VAR = {
    "tas":    "tas",
    "tasmax": "tasmax",
    "tasmin": "tasmin",
}

def load_model_basin_mean(cfg: Config, model_folder: str, var: str) -> pd.Series | None:
    folder = cfg.cmip6_dir / model_folder / var
    files = sorted(folder.glob(f"{_CMIP6_NETCDF_VAR[var]}_*.nc"))
    if not files:
        log.warning("[%s/%s] no NetCDF under %s", model_folder, var, folder)
        return None
    ds = xr.open_mfdataset(files, combine="by_coords",
                           use_cftime=True, chunks={"time": 60})
    ds = _normalize_dim_names(ds)
    ds = _wrap_lon_0_360_to_180(ds)
    if _CMIP6_NETCDF_VAR[var] not in ds.data_vars:
        log.warning("[%s/%s] variable %s missing from NetCDF (data_vars=%s)",
                    model_folder, var, _CMIP6_NETCDF_VAR[var], list(ds.data_vars))
        return None
    da = ds[_CMIP6_NETCDF_VAR[var]]
    da = _clip_box(da, cfg.bbox_nwse)

    if hasattr(da["time"].values[0], "calendar") or da["time"].dtype == object:
        da = da.convert_calendar("standard", align_on="date")

    s = _basin_mean(da)
    s = s.loc[cfg.eval_start: cfg.eval_end]

    # K -> °C
    if s.mean() > 100:
        s = s - 273.15
    return s.astype(float)


# --------------------------------------------------------------------------- #
# 4. METRICS
# --------------------------------------------------------------------------- #

def taylor_skill(rho: float, sigma_ratio: float) -> float:
    if not np.isfinite(rho) or not np.isfinite(sigma_ratio) or sigma_ratio <= 0:
        return np.nan
    return 4 * (1 + rho)**4 / ((sigma_ratio + 1/sigma_ratio)**2 * 16)


def annual_cycle_corr(a: pd.Series, b: pd.Series) -> float:
    ca = a.groupby(a.index.month).mean()
    cb = b.groupby(b.index.month).mean()
    return float(np.corrcoef(ca.values, cb.values)[0, 1])


def metric_row(model: str, var: str, ref: pd.Series, mod: pd.Series,
               cfg: Config) -> Dict[str, float]:
    df = pd.concat([ref.rename("ref"), mod.rename("mod")], axis=1).dropna()
    if len(df) < 24:
        log.warning("[%s / %s] only %d aligned months — skipping", model, var, len(df))
        return {"model": model, "variable": var, "n": len(df)}
    r, m = df["ref"].values, df["mod"].values
    sigma_r = float(r.std(ddof=1))
    sigma_m = float(m.std(ddof=1))
    rho     = float(np.corrcoef(r, m)[0, 1])
    crmse   = float(np.sqrt(sigma_m**2 + sigma_r**2 - 2*sigma_m*sigma_r*rho))
    bias    = float(m.mean() - r.mean())                                          # °C, not relative
    bias_norm = abs(bias) / (sigma_r if sigma_r > 0 else 1.0)
    cyc     = annual_cycle_corr(df["mod"], df["ref"])
    s2      = taylor_skill(rho, sigma_m / sigma_r)
    composite = (cfg.w_taylor * (s2 if np.isfinite(s2) else 0.0)
                 + cfg.w_bias  * (1.0 - min(bias_norm, 1.0))
                 + cfg.w_cycle * max(cyc, 0.0))
    return {
        "model":        model,
        "variable":     var,
        "n":            len(df),
        "sigma_ref":    sigma_r,
        "sigma_model":  sigma_m,
        "sigma_ratio":  sigma_m / sigma_r if sigma_r > 0 else np.nan,
        "rho":          rho,
        "crmse":        crmse,
        "bias_C":       bias,
        "bias_norm":    bias_norm,
        "cycle_corr":   cyc,
        "taylor_s2":    s2,
        "composite_S":  composite,
    }


# --------------------------------------------------------------------------- #
# 5. TAYLOR DIAGRAM (sub-panel, re-usable)
# --------------------------------------------------------------------------- #

class TaylorPanel:
    def __init__(self, fig: plt.Figure, rect: int, sigma_max: float = 1.6,
                 title: str | None = None) -> None:
        tr = PolarAxes.PolarTransform()
        rho_ticks   = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
        theta_ticks = np.arccos(rho_ticks)
        gl1 = grid_finder.FixedLocator(theta_ticks)
        tf1 = grid_finder.DictFormatter(dict(zip(theta_ticks, [str(r) for r in rho_ticks])))
        ghelper = floating_axes.GridHelperCurveLinear(
            tr, extremes=(0, np.pi / 2, 0, sigma_max),
            grid_locator1=gl1, tick_formatter1=tf1)
        ax = floating_axes.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)
        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text(r"$\rho$")
        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text(r"$\sigma_{\text{mod}}/\sigma_{\text{ref}}$")
        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")
        ax.axis["bottom"].set_visible(False)
        ax.grid(True)
        self._ax = ax
        self.ax  = ax.get_aux_axes(tr)
        # ref point + unit-std arc
        self.ax.plot(0, 1.0, "k*", ms=11)
        t = np.linspace(0, np.pi / 2, 100)
        self.ax.plot(t, np.ones_like(t), "k--", lw=1, alpha=0.5)
        # iso-RMSE contours
        rs, ts = np.meshgrid(np.linspace(0, sigma_max, 100),
                             np.linspace(0, np.pi / 2, 100))
        rmse = np.sqrt(1 + rs**2 - 2 * rs * np.cos(ts))
        cs = self.ax.contour(ts, rs, rmse, levels=[0.5, 1.0, 1.5],
                             colors="0.7", linewidths=0.6)
        self.ax.clabel(cs, inline=True, fontsize=6, fmt="%.1f")
        if title:
            ax.set_title(title, fontsize=11, pad=18)

    def add(self, sigma_ratio: float, rho: float, label: str,
             color: str | None = None) -> None:
        if not (np.isfinite(sigma_ratio) and np.isfinite(rho)):
            return
        theta = np.arccos(np.clip(rho, -1.0, 1.0))
        self.ax.plot(theta, sigma_ratio, marker="o", ms=7,
                     color=color, label=label)


def plot_taylor(metrics: pd.DataFrame, out_path: Path) -> None:
    variables = list(CFG.variables.keys())
    fig = plt.figure(figsize=(5.0 * len(variables), 5.5))
    cmap = plt.get_cmap("tab20")
    model_order = list(CFG.models.keys())
    for i, var in enumerate(variables):
        sigma_max = max(1.6, 1.05 * metrics[metrics["variable"] == var]
                                          ["sigma_ratio"].max(skipna=True))
        panel = TaylorPanel(fig, int(f"1{len(variables)}{i + 1}"),
                            sigma_max=sigma_max, title=var)
        for j, m in enumerate(model_order):
            row = metrics[(metrics["model"] == m) & (metrics["variable"] == var)]
            if row.empty: continue
            panel.add(row["sigma_ratio"].iloc[0], row["rho"].iloc[0],
                      label=m, color=cmap(j % 20))
    handles, labels = panel.ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Taylor diagrams — CMIP6 monthly temperature vs basin-mean references, "
                 "1990–2014", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_composite_bar(composite: pd.DataFrame, out_path: Path) -> None:
    ord_ = composite.sort_values("composite_S_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(ord_["model"], ord_["composite_S_mean"], color="steelblue", alpha=0.85)
    ax.invert_yaxis()
    ax.set_xlabel("Composite skill score S (mean over tas / tasmax / tasmin)")
    ax.set_title("Composite temperature skill — 10 CMIP6 models, top three retained "
                 "for scPDSI projection", fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    ax.axvline(ord_["composite_S_mean"].iloc[2], color="firebrick",
               linestyle="--", lw=0.8)
    ax.text(ord_["composite_S_mean"].iloc[2], len(ord_) - 1,
            "  top-3 cutoff", va="center", color="firebrick", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 6. MAIN
# --------------------------------------------------------------------------- #

def main() -> None:
    log.info("Project root:        %s", CFG.project_root)
    log.info("Bounding box NWSE:   %s", CFG.bbox_nwse)
    log.info("Evaluation window:   %s .. %s", CFG.eval_start, CFG.eval_end)

    refs = load_references(CFG)

    rows: List[Dict] = []
    for label, var_folder_map in CFG.models.items():
        for var in CFG.variables:
            folder = var_folder_map.get(var)
            if folder is None:
                log.info("[%-15s %-7s] skipped — no folder mapped (CDS catalogue gap)",
                         label, var)
                continue
            try:
                s_mod = load_model_basin_mean(CFG, folder, var)
            except Exception as exc:
                log.error("[%s/%s] load failed: %s", label, var, exc); continue
            if s_mod is None:
                continue
            row = metric_row(label, var, refs[var], s_mod, CFG)
            row["source_folder"] = folder
            rows.append(row)
            log.info("[%-15s %-7s] (src=%-18s) rho=%.2f σ̂=%.2f bias=%+.2f °C S=%.3f",
                     label, var, folder,
                     row.get("rho", np.nan), row.get("sigma_ratio", np.nan),
                     row.get("bias_C", np.nan), row.get("composite_S", np.nan))

    if not rows:
        log.error("No metrics produced — check CMIP6 paths."); return

    metrics = pd.DataFrame(rows)
    metrics_path = CFG.out_dir / "tables" / "skill_metrics_temperature.csv"
    metrics.to_csv(metrics_path, index=False, float_format="%.4f")
    log.info("Per-variable metrics -> %s", metrics_path)

    composite = (metrics.groupby("model")["composite_S"]
                  .agg(["mean", "min", "max", "count"])
                  .rename(columns={"mean": "composite_S_mean",
                                    "min":  "composite_S_min",
                                    "max":  "composite_S_max",
                                    "count": "n_variables"})
                  .reset_index()
                  .sort_values("composite_S_mean", ascending=False)
                  .reset_index(drop=True))
    composite.insert(0, "rank", composite.index + 1)
    composite_path = CFG.out_dir / "tables" / "skill_metrics_composite.csv"
    composite.to_csv(composite_path, index=False, float_format="%.4f")
    log.info("Composite ranking    -> %s", composite_path)

    best3 = composite.head(3).copy()
    best3.to_csv(CFG.out_dir / "tables" / "best_three_temperature.csv",
                  index=False, float_format="%.4f")
    log.info("Top 3 retained models    : %s", list(best3["model"]))

    plot_taylor(metrics,
                 CFG.out_dir / "figures" / "taylor_temperature.png")
    plot_composite_bar(composite,
                        CFG.out_dir / "figures" / "composite_ranking.png")

    print("\n=== Composite ranking (mean of S across tas / tasmax / tasmin) ===")
    print(composite.to_string(index=False))
    print("\n=== Top 3 retained for scPDSI projection ===")
    print(best3.to_string(index=False))


if __name__ == "__main__":
    main()
