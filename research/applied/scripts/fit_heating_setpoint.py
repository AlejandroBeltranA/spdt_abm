#!/usr/bin/env python3
"""
Fit the heating-engagement temperature (``heating_setpoint_C`` / outdoor
trigger) from SERL daily-targets temperature-band response.

Background
----------
The handoff names this as the first observation-based parameter to land in
the v5 cleanup. SERL publishes daily gas use stratified by outdoor
temperature band; the relationship is piecewise-linear with a sharp kink
where heating stops engaging. Below the kink, daily gas rises roughly
linearly in (setpoint − T_outdoor); above it, gas settles to a baseline
floor (DHW + cooking + standing losses). We fit the kink and read off the
outdoor temperature at which it occurs — that is the value
``EnergyModel.heating_trigger_temp_C`` should take.

Model
-----
    daily_kwh(T) = baseline + slope · max(0, setpoint − T)

with three free parameters (baseline, slope, setpoint). Fit by weighted
nonlinear least squares, weights = sqrt(n_rounded) per band (proportional
to SERL precision). Default scope: 2023, gas-heated households,
``has_pv=='All'``, ``period_type=='annual'``.

Output
------
``results/calibration_serl_fits/heating_setpoint.yaml`` containing the
fitted setpoint plus diagnostic block (per-band fitted vs observed, RMSE,
slope, baseline, n by band).

CLI
---
    python research/applied/scripts/fit_heating_setpoint.py \
        --year 2023 \
        --input  data/serl_8963_targets/daily_targets.csv \
        --output results/calibration_serl_fits/heating_setpoint.yaml

Designed to be callable as a subprocess from ``calibrate_serl.py`` (decision
#3 in the 2026-06-08 handoff: subprocess, not import — each fit script logs
its diagnostics independently and is reproducible standalone).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares


# Midpoint of each SERL temperature band, in °C. The lowest and highest
# bands ('-5_to_0', '20_to_25') are treated as their nominal midpoints;
# the fit is dominated by the four middle bands (0-20°C) which bracket
# the kink, so end-band exact treatment doesn't move the answer.
BAND_MIDPOINTS = {
    "-5_to_0": -2.5,
    "0_to_5":   2.5,
    "5_to_10":  7.5,
    "10_to_15": 12.5,
    "15_to_20": 17.5,
    "20_to_25": 22.5,
}


def load_temperature_band_data(
    input_path: Path,
    *,
    year: int,
    quantity: str = "Gas",
    heating_fuel: str = "Gas",
    has_pv: str = "All",
) -> pd.DataFrame:
    """Load and filter the SERL temperature-band response for a single year/fuel."""
    df = pd.read_csv(input_path)
    sub = df[
        (df["quantity"] == quantity)
        & (df["year"] == year)
        & (df["heating_fuel"] == heating_fuel)
        & (df["has_pv"] == has_pv)
        & (df["seg3_var"] == "temperature_band")
        & (df["period_type"] == "annual")
    ].copy()
    if sub.empty:
        raise RuntimeError(
            f"No SERL temperature_band rows found for "
            f"quantity={quantity!r} year={year} heating_fuel={heating_fuel!r} "
            f"has_pv={has_pv!r} in {input_path}"
        )
    sub["temp_C"] = sub["seg3_value"].map(BAND_MIDPOINTS)
    missing = sub[sub["temp_C"].isna()]
    if not missing.empty:
        raise RuntimeError(
            f"Unrecognised temperature_band values: {missing['seg3_value'].unique().tolist()}. "
            f"Extend BAND_MIDPOINTS in {__file__}."
        )
    sub = sub.sort_values("temp_C").reset_index(drop=True)
    return sub[["seg3_value", "temp_C", "mean", "n_rounded", "sd", "se"]]


def fit_hinge(temps: np.ndarray, kwh: np.ndarray, weights: np.ndarray) -> dict:
    """Weighted nonlinear LS fit of ``kwh = baseline + slope·max(0, setpoint − T)``.

    Returns a dict with fitted params and per-point fitted values + residuals.
    """
    def residuals(p):
        baseline, slope, setpoint = p
        pred = baseline + slope * np.maximum(0.0, setpoint - temps)
        return (pred - kwh) * weights

    # Initial guess: baseline = warm-band floor; slope = -drop/rise across
    # the kink-bracketing pair; setpoint = 15°C (literature ballpark).
    baseline_init = float(kwh[temps > 18.0].mean()) if (temps > 18.0).any() else float(kwh.min())
    cold_band = kwh[temps < 5.0]
    slope_init = float((cold_band.mean() - baseline_init) / (15.0 - temps[temps < 5.0].mean())) if cold_band.size else 5.0
    p0 = np.array([baseline_init, max(0.1, slope_init), 15.0])

    res = least_squares(
        residuals, p0,
        bounds=([0.0, 0.0, 5.0], [50.0, 50.0, 25.0]),
        method="trf",
    )
    baseline, slope, setpoint = res.x.tolist()
    pred = baseline + slope * np.maximum(0.0, setpoint - temps)
    rmse = float(np.sqrt(np.mean(((pred - kwh) * weights) ** 2) / np.mean(weights ** 2)))
    return {
        "baseline_kwh_per_day": float(baseline),
        "slope_kwh_per_day_per_deg": float(slope),
        "heating_setpoint_C": float(setpoint),
        "rmse_weighted_kwh_per_day": rmse,
        "fitted_kwh": pred.tolist(),
        "converged": bool(res.success),
        "cost": float(res.cost),
    }


def _leaf_values(data: pd.DataFrame) -> dict:
    """Config-bound leaf (``heating_setpoint_C``) plus the hinge slope/baseline
    from a temperature-band table. Reused by the point fit and the bootstrap.
    """
    temps = data["temp_C"].to_numpy(dtype=float)
    kwh = data["mean"].to_numpy(dtype=float)
    weights = np.sqrt(data["n_rounded"].to_numpy(dtype=float))
    fr = fit_hinge(temps, kwh, weights)
    return {
        "heating_setpoint_C":         fr["heating_setpoint_C"],
        "slope_kwh_per_day_per_deg":  fr["slope_kwh_per_day_per_deg"],
        "baseline_kwh_per_day":       fr["baseline_kwh_per_day"],
    }


def fit(
    input_path: Path,
    *,
    year: int = 2023,
    heating_fuel: str = "Gas",
) -> dict:
    """End-to-end: load → fit → return a structured result dict."""
    data = load_temperature_band_data(input_path, year=year, heating_fuel=heating_fuel)
    temps = data["temp_C"].to_numpy(dtype=float)
    kwh = data["mean"].to_numpy(dtype=float)
    weights = np.sqrt(data["n_rounded"].to_numpy(dtype=float))

    fit_result = fit_hinge(temps, kwh, weights)

    bands = []
    for row, fitted in zip(data.itertuples(index=False), fit_result["fitted_kwh"]):
        bands.append({
            "band": row.seg3_value,
            "temp_C_midpoint": float(row.temp_C),
            "observed_kwh_per_day": float(row.mean),
            "fitted_kwh_per_day": float(fitted),
            "n_rounded": int(row.n_rounded),
        })

    return {
        "heating_setpoint_C": fit_result["heating_setpoint_C"],
        "fit": {
            "baseline_kwh_per_day": fit_result["baseline_kwh_per_day"],
            "slope_kwh_per_day_per_deg": fit_result["slope_kwh_per_day_per_deg"],
            "rmse_weighted_kwh_per_day": fit_result["rmse_weighted_kwh_per_day"],
            "converged": fit_result["converged"],
        },
        "diagnostics": {
            "year": int(year),
            "heating_fuel": heating_fuel,
            "source": str(input_path),
            "bands": bands,
        },
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path,
                   default=Path("data/serl_8963_targets/daily_targets.csv"),
                   help="Path to SERL daily_targets.csv.")
    p.add_argument("--year", type=int, default=2023,
                   help="SERL year to fit. Default 2023 (Alex calibrates to 2023 only).")
    p.add_argument("--heating-fuel", type=str, default="Gas",
                   help="SERL heating_fuel filter. Default 'Gas'.")
    p.add_argument("--output", type=Path,
                   default=Path("results/calibration_serl_fits/heating_setpoint.yaml"),
                   help="YAML output path.")
    p.add_argument("--bootstrap", type=int, default=0,
                   help="If >0, parametric-bootstrap N draws of the hinge fit "
                        "from SERL published SE and write a 'bootstrap' block "
                        "(se/CI on setpoint, slope, baseline) into the YAML.")
    p.add_argument("--bootstrap-seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-band diagnostic print.")
    args = p.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: SERL input not found: {args.input}", file=sys.stderr)
        return 2

    result = fit(args.input, year=args.year, heating_fuel=args.heating_fuel)

    if args.bootstrap > 0:
        from bootstrap_bands import bootstrap_leaves, derive_se, merge_point_and_bands
        data = load_temperature_band_data(args.input, year=args.year, heating_fuel=args.heating_fuel)
        bands = bootstrap_leaves(
            data, _leaf_values, value_col="mean",
            se=derive_se(data, value_col="mean"),
            n_boot=args.bootstrap, seed=args.bootstrap_seed,
        )
        point = {
            "heating_setpoint_C":        result["heating_setpoint_C"],
            "slope_kwh_per_day_per_deg": result["fit"]["slope_kwh_per_day_per_deg"],
            "baseline_kwh_per_day":      result["fit"]["baseline_kwh_per_day"],
        }
        result["bootstrap"] = merge_point_and_bands(point, bands)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        yaml.safe_dump(result, f, sort_keys=False)

    if not args.quiet:
        print(f"Fitted heating_setpoint_C = {result['heating_setpoint_C']:.3f} °C")
        print(f"  baseline = {result['fit']['baseline_kwh_per_day']:.2f} kWh/day")
        print(f"  slope    = {result['fit']['slope_kwh_per_day_per_deg']:.3f} kWh/day/°C")
        print(f"  RMSE     = {result['fit']['rmse_weighted_kwh_per_day']:.2f} kWh/day (weighted)")
        print(f"  converged: {result['fit']['converged']}")
        print()
        print(f"{'band':>10} {'T_mid':>7} {'observed':>10} {'fitted':>10} {'n':>8}")
        for b in result["diagnostics"]["bands"]:
            print(f"{b['band']:>10} {b['temp_C_midpoint']:>7.1f} "
                  f"{b['observed_kwh_per_day']:>10.2f} {b['fitted_kwh_per_day']:>10.2f} "
                  f"{b['n_rounded']:>8d}")
        print()
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
