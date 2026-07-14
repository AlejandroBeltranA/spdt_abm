#!/usr/bin/env python3
"""
Multi-year SERL calibration → ABM parameters.

Default scope: pool 2020–2022 SERL respondents; hold out 2023 for the in-sample
DESNZ cross-check. The 2020 year is COVID-anomalous; drop with --years 2021 2022
if residuals look bad after the first run.

Outputs (under research/applied/results/calibration/<label>/):
  params.yaml        — anchors (gas/elec), HDD slopes (gas/elec),
                       building-type multipliers
  diagnostics.json   — fit quality metrics per fuel
  hdd_regression.png — slope visualisation (core vs shoulder months)

Usage:
  python -m research.applied.scripts.calibrate_serl \
      --years 2020 2021 2022 \
      --label 2020_2022 \
      --min-n 80

The `calibrate()` function is the bootstrap entry point — N=200 resamples of
SERL respondents will reuse it (task #4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import sys
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))     # research/applied/scripts/
sys.path.insert(0, str(_THIS.parents[3])) # repo root

from utils import repo_root, save_params, cache_dir  # noqa: E402
from progress import ProgressTracker  # noqa: E402
from household_energy.serl_calibration_v2 import load_serl_targets  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

SUMMER_MONTHS = [6, 7, 8]
CORE_HEATING_MONTHS = [10, 11, 12, 1, 2, 3]
# `heating_months` is the set of months when the model engages heating logic.
# Wider than CORE_HEATING_MONTHS (which is just the OLS regression window) —
# matches notebooks/serl_calibration_clean.ipynb cell 14 to keep behaviour
# parity with the existing validation framework.
DEFAULT_HEATING_MONTHS = [10, 11, 12, 1, 2, 3, 4]
DEFAULT_MIN_N = 80
# Outdoor temperature below which the model engages heating. Hard-coded at
# 13.0 °C in serl_calibration_clean.ipynb cell 14 — kept here for drop-in
# parity. CLI flag `--heating-trigger-c` overrides if needed.
DEFAULT_HEATING_TRIGGER_C = 13.0
# Uncaps the heating slope from the package default of 0.10 (which would
# silently cap calibrated slopes around 0.20+). Matches the notebook output.
DEFAULT_HEAT_SLOPE_MAX = 5.0
BTYPE_TO_CONFIG = {
    "Detached":                          "detached house",
    "Semi-detached":                     "semi-detached house",
    "Terraced":                          "mid-terraced house",
    "Purpose-built flat":                "block of flats",
    "Converted flat or shared house":    "small block of flats/dwelling converted in to flats",
    "Commercial building or no answer":  "detached house",
}


# ─────────────────────────────────────────────────────────────────────────────
# Core calibration (callable, used by bootstrap)
# ─────────────────────────────────────────────────────────────────────────────

def _serl_monthly_pooled(
    daily_targets: pd.DataFrame,
    *,
    years: list[int],
    heating_fuel: str,
) -> pd.DataFrame:
    """n-weighted monthly kWh/home/day pooled across `years` for one heating fuel."""
    sub = daily_targets[
        (daily_targets["period_type"].astype(str) == "monthly")
        & (daily_targets["year"].astype(int).isin([int(y) for y in years]))
        & (daily_targets["weekday_weekend"].astype(str) == "both")
        & (daily_targets["heating_fuel"].astype(str) == heating_fuel)
        & (daily_targets["has_pv"].astype(str).isin(["No", "All"]))
        & (daily_targets["seg3_var"].astype(str) == "none")
        & (daily_targets["quantity"].astype(str).isin(["Electricity imports", "Gas"]))
    ].copy()
    sub["n_rounded"] = pd.to_numeric(sub["n_rounded"], errors="coerce").fillna(0.0)
    sub["mean"]      = pd.to_numeric(sub["mean"], errors="coerce")
    sub["wx"]        = sub["mean"] * sub["n_rounded"]
    out = sub.groupby(["month", "quantity"], as_index=False).agg(
        wx=("wx", "sum"), n=("n_rounded", "sum"),
    )
    out["serl_kwh_per_home_day"] = out["wx"] / out["n"].replace({0: np.nan})
    out["fuel"] = out["quantity"].map({"Electricity imports": "electric", "Gas": "gas"})
    return out[["month", "fuel", "serl_kwh_per_home_day", "n"]]


def _summer_baseline_pooled(
    daily_targets: pd.DataFrame,
    *,
    years: list[int],
    summer_months: list[int],
    min_n: int,
) -> pd.DataFrame:
    """n-weighted summer kWh/home/day by building_type × fuel, pooled across years."""
    FUEL_MAP = {"Electricity imports": "electric", "Gas": "gas"}
    d = daily_targets[
        (daily_targets["year"].astype(int).isin([int(y) for y in years]))
        & (daily_targets["weekday_weekend"].astype(str) == "both")
        & (daily_targets["has_pv"].astype(str).isin(["No", "All"]))
        & (daily_targets["period_type"].astype(str) == "monthly")
        & (daily_targets["heating_fuel"].astype(str) == "All")
        & (daily_targets["seg3_var"].astype(str) == "building_type")
        & (daily_targets["quantity"].astype(str).isin(FUEL_MAP))
    ].copy()
    d["month"] = pd.to_numeric(d["month"], errors="coerce").astype("Int64")
    d = d[d["month"].isin([int(m) for m in summer_months])].copy()
    d["n_rounded"] = pd.to_numeric(d["n_rounded"], errors="coerce").fillna(0.0)
    d = d[d["n_rounded"] >= float(min_n)].copy()
    d["fuel"] = d["quantity"].astype(str).map(FUEL_MAP)
    d["seg3_value"] = d["seg3_value"].astype(str)
    d["mean"] = pd.to_numeric(d["mean"], errors="coerce")
    d["wx"] = d["mean"] * d["n_rounded"]

    baseline = d.groupby(["fuel", "seg3_value"], as_index=False).agg(
        wx=("wx", "sum"), n_total=("n_rounded", "sum"),
    )
    baseline["summer_kwh_per_home_day"] = baseline["wx"] / baseline["n_total"].replace({0: np.nan})
    return baseline[["fuel", "seg3_value", "summer_kwh_per_home_day", "n_total"]]


def _hdd_pooled(daily_targets: pd.DataFrame, *, years: list[int]) -> pd.DataFrame:
    """Monthly HDD (n-weighted) pooled across `years`."""
    raw = daily_targets[
        (daily_targets["period_type"].astype(str) == "monthly")
        & (daily_targets["year"].astype(int).isin([int(y) for y in years]))
        & (daily_targets["weekday_weekend"].astype(str) == "both")
        & (daily_targets["heating_fuel"].astype(str) == "Gas")
        & (daily_targets["quantity"].astype(str) == "Gas")
        & (daily_targets["has_pv"].astype(str).isin(["No", "All"]))
        & (daily_targets["seg3_var"].astype(str) == "none")
    ].copy()
    raw["month"]     = pd.to_numeric(raw["month"], errors="coerce").astype("Int64")
    raw["n_rounded"] = pd.to_numeric(raw["n_rounded"], errors="coerce").fillna(0.0)
    raw["wh"]        = pd.to_numeric(raw["mean_hdd"], errors="coerce") * raw["n_rounded"]
    out = raw[raw["n_rounded"] > 0].groupby("month", as_index=False).agg(
        n=("n_rounded", "sum"), wh=("wh", "sum"),
    )
    out["hdd"] = out["wh"] / out["n"].replace({0: np.nan})
    return out[["month", "hdd", "n"]]


def calibrate(
    daily_targets: pd.DataFrame,
    *,
    years: list[int],
    summer_months: list[int] = None,
    core_heating_months: list[int] = None,
    min_n: int = DEFAULT_MIN_N,
) -> dict:
    """Pooled multi-year calibration → parameter dict.

    Returns:
      {
        "years":              [int, ...],
        "anchors": {"gas": float, "electric": float},        # kWh/h
        "hdd_slopes": {"gas": float, "electric": float},     # kWh/h/°C
        "building_type_multipliers": {"gas": {...}, "electric": {...}},
        "fit_diagnostics": {
          "gas":      {"n_points": int, "r_squared": float, "monthly": [...]},
          "electric": {...},
        },
      }
    """
    summer_months = summer_months or SUMMER_MONTHS
    core_heating_months = core_heating_months or CORE_HEATING_MONTHS

    # 1) pooled monthly series per heating-fuel cohort
    serl_m_gas  = _serl_monthly_pooled(daily_targets, years=years, heating_fuel="Gas")
    serl_m_elec = _serl_monthly_pooled(daily_targets, years=years, heating_fuel="Electric")
    summer_bl   = _summer_baseline_pooled(daily_targets, years=years,
                                           summer_months=summer_months, min_n=min_n)
    hdd_m       = _hdd_pooled(daily_targets, years=years)

    # 2) baseline anchors
    _gas_summer = serl_m_gas.loc[
        (serl_m_gas["fuel"] == "gas") & serl_m_gas["month"].isin(summer_months),
        "serl_kwh_per_home_day",
    ].mean()
    gas_anchor = float(_gas_summer / 24.0)

    _e = summer_bl[summer_bl["fuel"] == "electric"]
    _ok = np.isfinite(_e["summer_kwh_per_home_day"].values) & (_e["n_total"].values > 0)
    elec_anchor = float(
        np.average(
            _e["summer_kwh_per_home_day"].values[_ok],
            weights=_e["n_total"].values[_ok],
        ) / 24.0
    )

    anchors = {"gas": gas_anchor, "electric": elec_anchor}

    # 3) building-type multipliers
    bt_mults: dict[str, dict[str, float]] = {"gas": {}, "electric": {}}
    for fuel, anchor in anchors.items():
        rows = summer_bl[summer_bl["fuel"] == fuel]
        for _, row in rows.iterrows():
            btype    = str(row["seg3_value"])
            cfg_name = BTYPE_TO_CONFIG.get(btype)
            k = row["summer_kwh_per_home_day"]
            if cfg_name and anchor > 0 and np.isfinite(k):
                m = round(float(k) / 24.0 / anchor, 6)
                bt_mults[fuel][cfg_name] = m
                if cfg_name == "mid-terraced house":
                    bt_mults[fuel]["end-terraced house"] = m
                if cfg_name == "block of flats":
                    bt_mults[fuel]["large block of flats"] = m

    # 4) HDD slopes (OLS through origin on core heating months)
    slopes: dict[str, float] = {}
    diagnostics: dict[str, dict] = {}
    sources = {
        "gas":      serl_m_gas [serl_m_gas ["fuel"] == "gas"],
        "electric": serl_m_elec[serl_m_elec["fuel"] == "electric"],
    }
    for fuel, src in sources.items():
        M_floor = anchors[fuel] * 24.0
        df = src.merge(hdd_m[["month", "hdd"]], on="month", how="inner")
        df_core = df[df["month"].isin(core_heating_months)].copy()
        df_core["excess"] = (df_core["serl_kwh_per_home_day"] - M_floor).clip(lower=0.0)
        df_core = df_core.dropna(subset=["hdd", "excess"])
        h = df_core["hdd"].to_numpy(float)
        e = df_core["excess"].to_numpy(float)
        ok = np.isfinite(h) & np.isfinite(e) & (h > 0)
        if ok.sum() >= 3:
            b_day = float(np.dot(h[ok], e[ok]) / np.dot(h[ok], h[ok]))
            pred = b_day * h[ok]
            ss_res = float(np.sum((e[ok] - pred) ** 2))
            ss_tot = float(np.sum((e[ok] - e[ok].mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        else:
            b_day, r2 = 0.2 * 24, float("nan")
        slopes[fuel] = b_day / 24.0
        diagnostics[fuel] = {
            "n_points": int(ok.sum()),
            "r_squared": r2,
            "monthly": df.to_dict(orient="records"),
            "floor_kwh_day": M_floor,
            "slope_kwh_day_per_degC": b_day,
        }

    return {
        "years":                     [int(y) for y in years],
        "anchors":                   anchors,
        "hdd_slopes":                slopes,
        "building_type_multipliers": bt_mults,
        "fit_diagnostics":           diagnostics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_hdd_regression(
    cal: dict,
    *,
    core_heating_months: list[int],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, fuel in zip(axes, ["gas", "electric"]):
        diag = cal["fit_diagnostics"][fuel]
        df_all  = pd.DataFrame(diag["monthly"])
        df_core = df_all[df_all["month"].isin(core_heating_months)]
        df_shld = df_all[~df_all["month"].isin(core_heating_months)]
        M_floor = diag["floor_kwh_day"]
        beta    = diag["slope_kwh_day_per_degC"]

        ax.scatter(df_core["hdd"], df_core["serl_kwh_per_home_day"],
                   s=60, zorder=3, label="Core months (fit)")
        ax.scatter(df_shld["hdd"], df_shld["serl_kwh_per_home_day"],
                   s=60, zorder=3, marker="x", color="gray", label="Shoulder (excluded)")
        if len(df_all):
            grid = np.linspace(0, float(df_all["hdd"].max()) * 1.05, 80)
            ax.plot(grid, M_floor + beta * grid, "r-", lw=2,
                    label=f"β={beta:.2f} kWh/day/HDD  (floor={M_floor:.1f}, R²={diag['r_squared']:.2f})")
            ax.axhline(M_floor, color="gray", lw=1, ls="--")
        ax.set_xlabel("Monthly mean HDD")
        ax.set_ylabel("kWh/home/day")
        ax.set_title(f"{fuel.capitalize()} — pooled {min(cal['years'])}–{max(cal['years'])}")
        ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=int, nargs="+", default=[2020, 2021, 2022],
                   help="SERL years to pool for calibration (default: 2020 2021 2022)")
    p.add_argument("--label", type=str, default=None,
                   help="Output subdir label (default: '<min>_<max>' from --years)")
    p.add_argument("--min-n", type=int, default=DEFAULT_MIN_N,
                   help="Minimum SERL sample size per segment to trust (default: 80)")
    p.add_argument("--heating-trigger-c", type=float, default=None,
                   help="Outdoor temp °C below which heating engages. Default: "
                        "None → fit from SERL temperature-band response via "
                        "fit_heating_setpoint.py (year=--setpoint-fit-year). "
                        "Pass a value to override the SERL fit.")
    p.add_argument("--setpoint-fit-year", type=int, default=2023,
                   help="SERL year to use for the heating_setpoint fit "
                        "(default 2023 — Alex calibrates to 2023 only)")
    p.add_argument("--setpoint-fit-script", type=Path, default=None,
                   help="Override path to fit_heating_setpoint.py (default: "
                        "alongside this script)")
    p.add_argument("--skip-area-scaling-fit", action="store_true",
                   help="Skip fit_area_scaling.py and fall back to the "
                        "power-law area scaling (heat_slope_area_exp) only.")
    p.add_argument("--area-scaling-fit-year", type=int, default=2023,
                   help="SERL year for the area-scaling fit (default 2023)")
    p.add_argument("--area-scaling-fit-script", type=Path, default=None,
                   help="Override path to fit_area_scaling.py")
    p.add_argument("--skip-sap-band-fit", action="store_true",
                   help="Skip fit_sap_band_mult.py; emits an empty SAP-band "
                        "lookup (no SAP composition lift).")
    p.add_argument("--skip-age-fit", action="store_true",
                   help="Skip fit_age_mult.py; emits an empty age lookup.")
    p.add_argument("--composition-fit-year", type=int, default=2023,
                   help="SERL year for the SAP and age fits (default 2023)")
    p.add_argument("--skip-presence-spikes-fit", action="store_true",
                   help="Skip fit_presence_spikes.py; falls back to the prior "
                        "hand-tuned per-person + spike defaults.")
    p.add_argument("--presence-spikes-fit-year", type=int, default=2023,
                   help="SERL year for the presence-spikes fit (default 2023)")
    p.add_argument("--skip-elec-baseline-area-fit", action="store_true",
                   help="Skip fit_elec_baseline_area.py; falls back to the "
                        "unsourced power-law area scaling in agent.py.")
    p.add_argument("--skip-diurnal-profile-fit", action="store_true",
                   help="Skip fit_diurnal_profile.py; baseline stays flat "
                        "across the day (no diurnal reshaping).")
    p.add_argument("--heat-slope-max", type=float, default=DEFAULT_HEAT_SLOPE_MAX,
                   help="Cap on the effective heating slope; default 5.0 uncaps "
                        "the package default of 0.10 so calibrated 0.20+ slopes aren't silently capped")
    p.add_argument("--heating-months", type=int, nargs="+", default=DEFAULT_HEATING_MONTHS,
                   help="Months (1-12) in which the model engages heating "
                        f"(default: {DEFAULT_HEATING_MONTHS})")
    p.add_argument("--output-root", type=Path, default=None,
                   help="Override results location (default: research/applied/results/calibration/)")
    p.add_argument("--bootstrap", type=int, default=0,
                   help="If >0, parametric-bootstrap N draws of every SERL fit "
                        "(in-process anchors/slope + each subprocess fit) from "
                        "SERL published SE, writing per-parameter se/CI bands. "
                        "Feeds the sensitivity-analysis parameter table.")
    p.add_argument("--bootstrap-seed", type=int, default=0)
    args = p.parse_args()

    boot_args = (
        ["--bootstrap", str(args.bootstrap), "--bootstrap-seed", str(args.bootstrap_seed)]
        if args.bootstrap > 0 else []
    )

    years = sorted(set(args.years))
    # Match the existing convention used by serl_calibration_clean.ipynb:
    #   results/calibration_<STAMP>/calibrated_config.yaml
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    label = args.label or stamp
    if args.output_root:
        out_dir = args.output_root / label
    else:
        out_dir = repo_root() / "results" / f"calibration_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = ProgressTracker(out_dir, job_name=f"calibrate_serl_{label}")
    tracker.start(f"years={years} min_n={args.min_n}")
    tracker.milestone(f"output_dir={out_dir}")

    with tracker.section("load SERL targets"):
        daily_targets, _hourly_targets = load_serl_targets(repo_root())
    tracker.milestone("SERL loaded", n_rows=len(daily_targets))

    with tracker.section("calibrate (pool anchors + slopes + multipliers)"):
        cal = calibrate(daily_targets, years=years, min_n=args.min_n)
    tracker.milestone(
        "calibration fit",
        gas_anchor=f"{cal['anchors']['gas']:.4f}",
        elec_anchor=f"{cal['anchors']['electric']:.4f}",
        gas_slope=f"{cal['hdd_slopes']['gas']:.4f}",
        elec_slope=f"{cal['hdd_slopes']['electric']:.4f}",
        gas_r2=f"{cal['fit_diagnostics']['gas']['r_squared']:.3f}",
        elec_r2=f"{cal['fit_diagnostics']['electric']['r_squared']:.3f}",
    )

    # ── Fit heating setpoint from SERL temperature-band response ──
    #
    # Run fit_heating_setpoint.py as a subprocess (decision #3 in the
    # 2026-06-08 handoff: subprocess, not import — each fit script logs
    # its own diagnostics and is reproducible standalone). The script
    # writes a YAML; we read back the fitted value.
    if args.heating_trigger_c is None:
        import subprocess
        fit_script = args.setpoint_fit_script or (_THIS.parent / "fit_heating_setpoint.py")
        setpoint_yaml = out_dir / "heating_setpoint_fit.yaml"
        with tracker.section("fit heating_setpoint_C from SERL"):
            cmd = [
                sys.executable, str(fit_script),
                "--year", str(args.setpoint_fit_year),
                "--input", str(repo_root() / "data" / "serl_8963_targets" / "daily_targets.csv"),
                "--output", str(setpoint_yaml),
                "--quiet",
            ]
            cmd += boot_args
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"fit_heating_setpoint.py failed (rc={proc.returncode}):\n"
                    f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                )
            with open(setpoint_yaml) as f:
                setpoint_fit = yaml.safe_load(f)
        heating_trigger_c = float(setpoint_fit["heating_setpoint_C"])
        tracker.milestone(
            "heating_setpoint fit",
            heating_setpoint_C=f"{heating_trigger_c:.3f}",
            slope=f"{setpoint_fit['fit']['slope_kwh_per_day_per_deg']:.3f}",
            baseline=f"{setpoint_fit['fit']['baseline_kwh_per_day']:.2f}",
        )
    else:
        heating_trigger_c = float(args.heating_trigger_c)
        tracker.milestone("heating_setpoint override", heating_setpoint_C=f"{heating_trigger_c:.3f}")

    # ── Fit per-floor-area slope multipliers from SERL ──
    #
    # Same subprocess pattern as the setpoint fit. Produces a lookup
    # {band: multiplier} normalised to the 51-100 m² reference band,
    # which agent.py applies in place of the unsourced power-law
    # exponent. Diagnostic-only: also reports the implied exponent for
    # readers who want a single number (~0.81 on SERL 2023).
    heat_slope_area_bands: dict | None = None
    if not args.skip_area_scaling_fit:
        import subprocess
        area_fit_script = args.area_scaling_fit_script or (_THIS.parent / "fit_area_scaling.py")
        area_yaml = out_dir / "area_scaling_fit.yaml"
        with tracker.section("fit heat_slope_area_bands from SERL"):
            cmd = [
                sys.executable, str(area_fit_script),
                "--year", str(args.area_scaling_fit_year),
                "--input", str(repo_root() / "data" / "serl_8963_targets" / "daily_targets.csv"),
                "--output", str(area_yaml),
                "--quiet",
            ]
            cmd += boot_args
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"fit_area_scaling.py failed (rc={proc.returncode}):\n"
                    f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                )
            with open(area_yaml) as f:
                area_fit = yaml.safe_load(f)
        heat_slope_area_bands = area_fit["heat_slope_area_bands"]
        tracker.milestone(
            "area_scaling fit",
            implied_exp=f"{area_fit['implied_exponent']:.3f}",
            ref_band=area_fit["reference_band"],
            **{k.replace(" ", "_"): f"{v:.3f}" for k, v in heat_slope_area_bands.items()},
        )
    else:
        tracker.milestone("area_scaling fit skipped — using power-law fallback")

    # ── Fit SAP-band heating multipliers from SERL ──
    sap_band_mult: dict | None = None
    if not args.skip_sap_band_fit:
        import subprocess
        sap_script = _THIS.parent / "fit_sap_band_mult.py"
        sap_yaml = out_dir / "sap_band_mult_fit.yaml"
        with tracker.section("fit sap_band_mult_heating_gas from SERL"):
            cmd = [sys.executable, str(sap_script),
                   "--year", str(args.composition_fit_year),
                   "--input", str(repo_root() / "data" / "serl_8963_targets" / "daily_targets.csv"),
                   "--output", str(sap_yaml), "--quiet"]
            cmd += boot_args
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"fit_sap_band_mult.py failed (rc={proc.returncode}):\n"
                    f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                )
            with open(sap_yaml) as f:
                sap_fit = yaml.safe_load(f)
        sap_band_mult = sap_fit["sap_band_mult_heating_gas"]
        tracker.milestone(
            "sap_band_mult fit",
            ref_band=sap_fit["reference_band"],
            **{k.replace(" ", "_"): f"{v:.3f}" for k, v in sap_band_mult.items()},
        )
    else:
        tracker.milestone("sap_band_mult fit skipped")

    # ── Fit building-age heating multipliers from SERL ──
    age_mult: dict | None = None
    if not args.skip_age_fit:
        import subprocess
        age_script = _THIS.parent / "fit_age_mult.py"
        age_yaml = out_dir / "age_mult_fit.yaml"
        with tracker.section("fit building_age_mult_heating_gas from SERL"):
            cmd = [sys.executable, str(age_script),
                   "--year", str(args.composition_fit_year),
                   "--input", str(repo_root() / "data" / "serl_8963_targets" / "daily_targets.csv"),
                   "--output", str(age_yaml), "--quiet"]
            cmd += boot_args
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"fit_age_mult.py failed (rc={proc.returncode}):\n"
                    f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                )
            with open(age_yaml) as f:
                age_fit = yaml.safe_load(f)
        age_mult = age_fit["building_age_mult_heating_gas"]
        tracker.milestone(
            "building_age_mult fit",
            ref_band=age_fit["reference_band"],
            **{k.replace(" ", "_").replace("-", ""): f"{v:.3f}" for k, v in age_mult.items()},
        )
    else:
        tracker.milestone("building_age_mult fit skipped")

    # ── Fit electricity baseline per-floor-area lookup from SERL ──
    # (Phase 5: replaces the unsourced power-law `baseline_area_exp=0.20`
    # clipped to [0.85, 1.25] in agent.py. SERL exposes per-band
    # Electricity × floor_area_m2 monthly with R²~0.95.)
    baseline_elec_area_bands: dict | None = None
    if not args.skip_elec_baseline_area_fit:
        import subprocess
        eba_script = _THIS.parent / "fit_elec_baseline_area.py"
        eba_yaml = out_dir / "elec_baseline_area_fit.yaml"
        with tracker.section("fit baseline_elec_area_bands from SERL"):
            cmd = [sys.executable, str(eba_script),
                   "--year", str(args.composition_fit_year),
                   "--input", str(repo_root() / "data" / "serl_8963_targets" / "daily_targets.csv"),
                   "--output", str(eba_yaml), "--quiet"]
            cmd += boot_args
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"fit_elec_baseline_area.py failed (rc={proc.returncode}):\n"
                    f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                )
            with open(eba_yaml) as f:
                eba_fit = yaml.safe_load(f)
        baseline_elec_area_bands = eba_fit["baseline_elec_area_bands"]
        tracker.milestone(
            "elec_baseline_area fit",
            implied_exp=f"{eba_fit['implied_exponent']:.3f}",
            ref_band=eba_fit["reference_band"],
            **{k.replace(" ", "_"): f"{v:.3f}" for k, v in baseline_elec_area_bands.items()},
        )
    else:
        tracker.milestone("elec_baseline_area fit skipped — using power-law fallback")

    # ── Fit per-person presence-spike electricity params from SERL ──
    presence_spikes: dict | None = None
    if not args.skip_presence_spikes_fit:
        import subprocess
        ps_script = _THIS.parent / "fit_presence_spikes.py"
        ps_yaml = out_dir / "presence_spikes_fit.yaml"
        with tracker.section("fit presence-spike per-person loads from SERL"):
            cmd = [sys.executable, str(ps_script),
                   "--year", str(args.presence_spikes_fit_year),
                   "--input", str(repo_root() / "data" / "serl_8963_targets" / "diurnal_targets_hourly_mean.csv"),
                   "--output", str(ps_yaml), "--quiet"]
            cmd += boot_args
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"fit_presence_spikes.py failed (rc={proc.returncode}):\n"
                    f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                )
            with open(ps_yaml) as f:
                ps_fit = yaml.safe_load(f)
        presence_spikes = {
            "energy_per_person_home":  ps_fit["energy_per_person_home"],
            "energy_per_person_away":  ps_fit["energy_per_person_away"],
            "awake_home_spike_mult":   ps_fit["awake_home_spike_mult"],
            "sleep_home_spike_mult":   ps_fit["sleep_home_spike_mult"],
            # Carried for the elec_anchor recentring step below — NOT
            # emitted into the model config (these are calibration inputs,
            # not model params).
            "naive_per_person_home_awake": ps_fit.get("naive_per_person_home_awake"),
            "panel_mean_occupants":        ps_fit.get("panel_mean_occupants"),
        }
        tracker.milestone(
            "presence_spikes fit",
            energy_per_person_home=f"{presence_spikes['energy_per_person_home']:.4f}",
            sleep_mult=f"{presence_spikes['sleep_home_spike_mult']:.3f}",
        )
    else:
        tracker.milestone("presence_spikes fit skipped")

    # ── Fit baseline electricity diurnal profile from SERL ──
    #
    # presence_spikes (above) collapses the per-hour signal into awake/sleep
    # scalars, and the baseline anchor is applied flat every hour. That leaves
    # the model's electricity a "box" — overnight too high, evening peak too
    # shallow — even though the daily mean (and thus the annual total) is right.
    # This fit emits a mean-1.0 24h profile that model._reset_base_loads
    # multiplies onto the baseline, restoring the SERL diurnal shape without
    # moving the daily mean. Mean-preserving → annual totals and every other
    # fitted parameter are untouched.
    base_profile_24h_electric: list | None = None
    if not args.skip_diurnal_profile_fit:
        import subprocess
        dp_script = _THIS.parent / "fit_diurnal_profile.py"
        dp_yaml = out_dir / "diurnal_profile_fit.yaml"
        with tracker.section("fit baseline electricity diurnal profile from SERL"):
            cmd = [sys.executable, str(dp_script),
                   "--year", str(args.presence_spikes_fit_year),
                   "--input", str(repo_root() / "data" / "serl_8963_targets" / "diurnal_targets_hourly_mean.csv"),
                   "--output", str(dp_yaml), "--quiet"]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"fit_diurnal_profile.py failed (rc={proc.returncode}):\n"
                    f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                )
            with open(dp_yaml) as f:
                dp_fit = yaml.safe_load(f)
        base_profile_24h_electric = dp_fit["base_profile_24h_electric"]
        tracker.milestone(
            "diurnal_profile fit",
            swing=f"{dp_fit['diagnostics']['electric']['swing_max_over_min']:.2f}x",
            peak_h=dp_fit["diagnostics"]["electric"]["peak_hour"],
        )
    else:
        tracker.milestone("diurnal_profile fit skipped")

    # ── Recentre electricity anchor (Phase 5b) ──
    #
    # The SERL per-person fit is partialled (marginal per-added-person
    # behavioural slope ≈ 0.031 kWh/h), but baseline_anchor_elec_kwh_per_hour
    # was implicitly co-calibrated with the naive per-person slope (≈ 0.098).
    # When we emit the partialled value without recentring, the panel-mean
    # household integrates to ~10–25% below the SERL panel mean, since
    # average-occupancy per-person load that was previously inside the anchor
    # is no longer being put back in.
    #
    # Algebra: each household integrates
    #     E_i = anchor + β · n_i · ω · 8760
    # where ω is the empirically-observed "at-home and awake" fraction across
    # the year (~0.59 from the E01008344 P4b–P5 probe diff). To preserve the
    # panel-mean level under β: naive → partialled, recentre by
    #     ΔB = (β_naive − β_partialled) × n̄_panel × ω
    # which lands ~0.09 kWh/h on 2023 SERL.
    #
    # Notes:
    #   - n̄_panel is the SERL panel mean (~2.28 in 2023), NOT a target-city
    #     synthetic-population mean. Using the city mean would smuggle a
    #     level correction into the transfer step.
    #   - ω = 0.59 is calibrated against the agent-model gating
    #     (add_person_load runs only when at_home & awake). Analytical
    #     estimate 16h_awake/24 × ~75%_at_home ≈ 0.50 sits near this.
    #   - Applies to electricity only; the spike loads route mostly to
    #     electricity via gas_spike_share. Gas anchor untouched.
    AT_HOME_AWAKE_OMEGA = 0.59  # empirical, from P4b–P5 LSOA probe diff
    elec_anchor_calibrated = cal["anchors"]["electric"]
    elec_anchor_recentred = elec_anchor_calibrated
    if presence_spikes is not None:
        n_panel = float(presence_spikes.get("panel_mean_occupants", float("nan")))
        beta_naive = float(presence_spikes.get("naive_per_person_home_awake", float("nan")))
        beta_partialled = float(presence_spikes["energy_per_person_home"])
        if all(np.isfinite([n_panel, beta_naive, beta_partialled])):
            delta_B = (beta_naive - beta_partialled) * n_panel * AT_HOME_AWAKE_OMEGA
            elec_anchor_recentred = elec_anchor_calibrated + delta_B
            tracker.milestone(
                "elec_anchor recentred",
                base=f"{elec_anchor_calibrated:.4f}",
                delta=f"{delta_B:+.4f}",
                recentred=f"{elec_anchor_recentred:.4f}",
                n_panel=f"{n_panel:.3f}",
                beta_naive=f"{beta_naive:.4f}",
                beta_partialled=f"{beta_partialled:.4f}",
                omega=f"{AT_HOME_AWAKE_OMEGA:.2f}",
            )

    # ── Persist parameters in a model.config-compatible shape ──
    #
    # Schema matches what the EnergyModel actually reads:
    #   model.py:114        → heating_trigger_temp_C
    #   model.py:131        → heating_slope_kWh_per_deg (single slope; the
    #                          per-fuel split is handled by separate fuel anchors,
    #                          not by separate slopes — only one slope is used)
    #   model.py:136        → heat_slope_max (defaults to 0.10, would silently
    #                          cap calibrated 0.20+; uncap to 5.0)
    #   model.py:137-138    → heating_months (set of months when heating engages)
    #   model.py:215-221    → property_type_mult_base_{gas,electric} (NOT
    #                          building_type_multipliers_*)
    #
    # Backwards-compat: also emits heating_setpoint_C (old key) so configs
    # work against older model.py versions without the rename shim.
    params_for_model = {
        "model": {
            "heating_trigger_temp_C":             heating_trigger_c,
            "heating_setpoint_C":                 heating_trigger_c,  # legacy alias
            "heating_slope_kWh_per_deg":          cal["hdd_slopes"]["gas"],
            "heat_slope_max":                     args.heat_slope_max,
            "heating_months":                     list(args.heating_months),
            "baseline_anchor_gas_kwh_per_hour":   cal["anchors"]["gas"],
            "baseline_anchor_elec_kwh_per_hour":  elec_anchor_recentred,
            "use_separate_fuel_baseline_anchors": True,
            "property_type_mult_base_gas":        cal["building_type_multipliers"]["gas"],
            "property_type_mult_base_electric":   cal["building_type_multipliers"]["electric"],
            # SERL-fitted per-floor-area slope multipliers (Phase 2). When
            # present, agent.py uses this lookup in place of the power-law
            # `heat_slope_area_exp` fallback.
            **({"heat_slope_area_bands": heat_slope_area_bands}
               if heat_slope_area_bands is not None else {}),
            # SERL-fitted SAP and building-age slope multipliers
            # (Phase 3). agent.py:830 reads these lookups and multiplies
            # them as `_heating_sensitivity_mult`. SAP and age are
            # correlated in the SERL panel; the marginal multiplication
            # absorbs some shared signal twice — validated empirically
            # via probe + three-city transfer.
            **({"sap_band_mult_heating_gas": sap_band_mult}
               if sap_band_mult is not None else {}),
            **({"building_age_mult_heating_gas": age_mult}
               if age_mult is not None else {}),
            # Disable the unsourced continuous SAP slope interpolation
            # (agent.py:583-588 defaults: lo=0.70, hi=1.30 over SAP 40-90).
            # The SERL-fitted band lookup above is now the single SAP
            # source. The spike-kind interpolation (line 612) is left
            # alone — it's used by the occupancy-spike path, not heating.
            # All four SAP-scaling sides neutralised (Phase 5b). The slope
            # side is superseded by the SERL-fitted sap_band_mult lookup.
            # The spike side (used by occupancy spike loads, agent.py:611)
            # is also neutralised — there's no SERL signal that identifies
            # SAP-conditional behavioural spike load.
            "sap_scaling": {
                "slope_mult_lo": 1.0, "slope_mult_hi": 1.0,
                "spike_mult_lo": 1.0, "spike_mult_hi": 1.0,
                "cap_mult_lo":   1.0, "cap_mult_hi":   1.0,
            },
            # Wealth-bucket multiplier neutralised (Phase 5b). No SERL
            # signal identifies wealth-conditional load directly; IMD
            # would be the natural proxy but it isn't fit. Keeping the
            # channel unsourced introduces uncalibrated noise.
            "wealth_mult_map": {
                "very_low": 1.0, "low": 1.0, "mid": 1.0,
                "high": 1.0, "very_high": 1.0,
            },
            # SERL-fitted per-person presence-spike electricity params
            # (fit_presence_spikes.py). awake_home_spike_mult=1.0 is the
            # reference; sleep_home_spike_mult is sleep/awake slope ratio.
            # energy_per_person_away stays at the literature default
            # (SERL aggregate diurnal can't identify it).
            **(
                {k: v for k, v in presence_spikes.items()
                 if k not in {"naive_per_person_home_awake", "panel_mean_occupants"}}
                if presence_spikes is not None else {}
            ),
            # SERL-fitted electricity baseline per-floor-area lookup
            # (Phase 5, fit_elec_baseline_area.py). When present, agent.py
            # uses this band lookup in _baseline_area_multiplier in place
            # of the unsourced power-law fallback (`baseline_area_exp`
            # clipped to [0.85, 1.25]).
            **({"baseline_elec_area_bands": baseline_elec_area_bands}
               if baseline_elec_area_bands is not None else {}),
            # SERL-fitted baseline electricity diurnal profile
            # (fit_diurnal_profile.py). Mean-1.0 24h shape applied to the
            # per-hour baseline in model._reset_base_loads; reshapes the day
            # without moving the daily mean (annual totals unchanged).
            **({"base_profile_24h_electric": base_profile_24h_electric}
               if base_profile_24h_electric is not None else {}),
        },
        "meta": {
            "calibration_years":   cal["years"],
            "min_n":               args.min_n,
            "summer_months":       SUMMER_MONTHS,
            "core_heating_months": CORE_HEATING_MONTHS,
            # Diagnostic-only: per-fuel slopes (electric slope is informative
            # but the model uses the single `heating_slope_kWh_per_deg` value
            # above; fuel allocation happens via baseline anchors + heating
            # bucket classification in agent.py).
            "hdd_slope_gas":       cal["hdd_slopes"]["gas"],
            "hdd_slope_electric":  cal["hdd_slopes"]["electric"],
        },
    }
    save_params(params_for_model, out_dir / "calibrated_config.yaml")

    # ── Parametric bootstrap of the in-process fit (gas slope + anchors) ──
    #
    # The subprocess fit scripts each emit their own bootstrap block; the
    # anchors and the headline gas HDD slope are computed in-process by
    # calibrate(), so bootstrap them here with the same SERL-published-SE
    # perturbation. Re-runs calibrate() on N draws of daily_targets['mean'].
    # The elec anchor band is the *calibrated* (pre-recentre) anchor; the
    # +ΔB recentre is a fixed offset applied on top, so its SE carries through.
    calib_bootstrap = None
    if args.bootstrap > 0:
        from bootstrap_bands import bootstrap_leaves, derive_se, merge_point_and_bands

        def _cal_leaves(d):
            c = calibrate(d, years=years, min_n=args.min_n)
            return {
                "heating_slope_kWh_per_deg":         c["hdd_slopes"]["gas"],
                "baseline_anchor_gas_kwh_per_hour":  c["anchors"]["gas"],
                "baseline_anchor_elec_kwh_per_hour": c["anchors"]["electric"],
            }

        with tracker.section("bootstrap in-process anchors + gas slope"):
            bands = bootstrap_leaves(
                daily_targets, _cal_leaves, value_col="mean",
                se=derive_se(daily_targets, value_col="mean"),
                n_boot=args.bootstrap, seed=args.bootstrap_seed,
            )
        calib_bootstrap = merge_point_and_bands(
            {
                "heating_slope_kWh_per_deg":         cal["hdd_slopes"]["gas"],
                "baseline_anchor_gas_kwh_per_hour":  cal["anchors"]["gas"],
                "baseline_anchor_elec_kwh_per_hour": cal["anchors"]["electric"],
            },
            bands,
        )
        tracker.milestone(
            "in-process bootstrap",
            slope_se=f"{calib_bootstrap['heating_slope_kWh_per_deg']['se']:.5f}",
            gas_anchor_se=f"{calib_bootstrap['baseline_anchor_gas_kwh_per_hour']['se']:.5f}",
            elec_anchor_se=f"{calib_bootstrap['baseline_anchor_elec_kwh_per_hour']['se']:.5f}",
        )

    # Persist diagnostics
    diag = {
        "anchors":    cal["anchors"],
        "hdd_slopes": cal["hdd_slopes"],
        "fit": {
            fuel: {k: v for k, v in d.items() if k != "monthly"}
            for fuel, d in cal["fit_diagnostics"].items()
        },
        "years":      cal["years"],
        **({"bootstrap": calib_bootstrap} if calib_bootstrap is not None else {}),
    }
    with open(out_dir / "diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2, default=float)

    # Plot
    with tracker.section("plot HDD regression"):
        plot_hdd_regression(cal, core_heating_months=CORE_HEATING_MONTHS,
                            out_path=out_dir / "hdd_regression.png")

    tracker.finish(f"wrote calibrated_config.yaml + diagnostics.json + hdd_regression.png to {out_dir}")


if __name__ == "__main__":
    main()
