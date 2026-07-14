#!/usr/bin/env python3
"""
Fit per-floor-area-band heating slope multipliers from SERL daily targets.

Background
----------
Phase 1 of the v5 cleanup landed linear heating with the SERL-OLS gas slope
(0.2097 kWh/h/°C) and a SERL-fitted setpoint (16.47°C). It also removed the
v2 stock-composition multipliers (`sap_band_mult_heating_gas`,
`building_age_mult_heating_gas`). The single-LSOA probe on Newcastle median
E01008344 came in at 9,564 kWh/yr per gas-dwelling — vs the target
SERL × Newcastle-composition (1.190) ≈ 12,600. The model is now 10–15% below
SERL itself for Newcastle and Waltham Forest, because nothing lifts dwellings
above the SERL panel mean by their characteristics.

The single composition lever left in ``agent.py`` is the floor-area scaling
in ``_compute_heat_slope`` (line ~475), currently a power law with
``heat_slope_area_exp = 0.6``. That exponent is unsourced. SERL provides
``seg3_var=='floor_area_m2'`` rows giving daily gas use by area band — by
fitting daily_kwh ~ HDD per band we get the slope each band actually has in
SERL, and the per-band ratio to the reference band is the multiplier the
simulator should use.

Method
------
- Filter to ``quantity='Gas'``, ``heating_fuel='Gas'``, ``has_pv='All'``,
  ``period_type='monthly'``, ``seg3_var='floor_area_m2'``, year 2023.
- For each of the 5 bands (50 or less / 51-100 / 101-150 / 151-200 / >200),
  fit ``daily_kwh = baseline + slope · HDD`` via weighted least squares,
  weights = sqrt(n_rounded). The intercept absorbs band-specific baseload
  (DHW, cooking, standing losses) and is reported as a diagnostic.
- Normalise slopes against the reference band 51-100 (largest panel weight).
  The result is a lookup ``{band: multiplier}`` that ``agent.py`` applies
  when computing per-dwelling heating slope.
- Also report the implied power-law exponent (OLS on log(slope) vs
  log(area_midpoint)) — this is the value ``heat_slope_area_exp`` would
  collapse to if forced into a single number. Diagnostic only; the lookup
  is the authoritative output.

Output
------
``results/calibration_serl_fits/area_scaling.yaml`` containing the lookup
plus diagnostics (per-band slope, baseline, n, implied exponent, fit
quality).

CLI
---
    python research/applied/scripts/fit_area_scaling.py \
        --year 2023 \
        --input  data/serl_8963_targets/daily_targets.csv \
        --output results/calibration_serl_fits/area_scaling.yaml

Designed to be called as a subprocess from ``calibrate_serl.py`` (the
established v5 pattern; each fit script logs its own diagnostics).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


# SERL floor-area bands. Midpoints used to fit the implied power-law
# exponent only — the lookup itself doesn't depend on these. End bands
# get a representative interior value (not the open edge).
BAND_MIDPOINTS_M2 = {
    "50 or less":  40.0,
    "51 to 100":   75.0,
    "101 to 150": 125.0,
    "151 to 200": 175.0,
    "Over 200":   240.0,
}
BAND_ORDER = ["50 or less", "51 to 100", "101 to 150", "151 to 200", "Over 200"]
REFERENCE_BAND = "51 to 100"  # largest SERL panel weight


def load_area_band_data(
    input_path: Path,
    *,
    year: int,
    quantity: str = "Gas",
    heating_fuel: str = "Gas",
    has_pv: str = "All",
) -> pd.DataFrame:
    """Load monthly SERL daily-mean rows segmented by floor-area band."""
    df = pd.read_csv(input_path)
    sub = df[
        (df["quantity"] == quantity)
        & (df["year"] == year)
        & (df["heating_fuel"] == heating_fuel)
        & (df["has_pv"] == has_pv)
        & (df["seg3_var"] == "floor_area_m2")
        & (df["period_type"] == "monthly")
    ].copy()
    if sub.empty:
        raise RuntimeError(
            f"No SERL floor_area_m2 monthly rows for "
            f"quantity={quantity!r} year={year} heating_fuel={heating_fuel!r} "
            f"has_pv={has_pv!r} in {input_path}"
        )
    missing_bands = set(BAND_ORDER) - set(sub["seg3_value"].unique())
    if missing_bands:
        raise RuntimeError(f"Missing area bands in SERL data: {missing_bands}")
    return sub[["seg3_value", "month", "mean", "mean_hdd", "n_rounded", "sd", "se"]]


def fit_band_slope(rows: pd.DataFrame) -> dict:
    """Weighted OLS of ``daily_kwh = baseline + slope · HDD`` for one band."""
    hdd = rows["mean_hdd"].to_numpy(dtype=float)
    kwh = rows["mean"].to_numpy(dtype=float)
    w = np.sqrt(rows["n_rounded"].to_numpy(dtype=float))

    # Weighted least squares via the normal equations on (1, HDD) columns.
    X = np.column_stack([np.ones_like(hdd), hdd])
    W = np.diag(w)
    # Solve (X' W² X) β = X' W² y
    A = X.T @ (W @ W) @ X
    b = X.T @ (W @ W) @ kwh
    beta = np.linalg.solve(A, b)
    baseline, slope = float(beta[0]), float(beta[1])

    pred = X @ beta
    resid = kwh - pred
    # Weighted R²
    ss_res = float(np.sum((resid * w) ** 2))
    ybar = float(np.average(kwh, weights=w ** 2))
    ss_tot = float(np.sum(((kwh - ybar) * w) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "baseline_kwh_per_day":     baseline,
        "slope_kwh_per_day_per_HDD": slope,
        "r2_weighted":              r2,
        "n_obs":                    int(len(rows)),
        "mean_n_rounded":           float(rows["n_rounded"].mean()),
    }


def fit_implied_exponent(slopes: dict[str, float]) -> dict:
    """OLS regress log(slope) ~ log(area_midpoint). Diagnostic only."""
    bands = [b for b in BAND_ORDER if slopes[b] > 0]
    x = np.log(np.array([BAND_MIDPOINTS_M2[b] for b in bands]))
    y = np.log(np.array([slopes[b] for b in bands]))
    # OLS slope = exponent; intercept = log(slope at 1 m²)
    n = len(x)
    xm, ym = x.mean(), y.mean()
    cov = ((x - xm) * (y - ym)).sum()
    var = ((x - xm) ** 2).sum()
    alpha = float(cov / var) if var > 0 else float("nan")
    log_c = float(ym - alpha * xm)
    pred = log_c + alpha * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - ym) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"implied_exponent": alpha, "log_c": log_c, "r2": r2}


def _leaf_values(data: pd.DataFrame) -> dict:
    """Config-bound leaves (the per-band slope multipliers) from a band table.

    Reused by both the point fit and the parametric bootstrap, so a perturbed
    ``mean`` column flows through exactly the same normalisation the model sees.
    """
    per_band = {b: fit_band_slope(data[data["seg3_value"] == b]) for b in BAND_ORDER}
    ref_slope = per_band[REFERENCE_BAND]["slope_kwh_per_day_per_HDD"]
    if ref_slope <= 0:
        raise RuntimeError(
            f"Reference band {REFERENCE_BAND!r} fitted non-positive slope "
            f"({ref_slope}); cannot normalise."
        )
    return {b: per_band[b]["slope_kwh_per_day_per_HDD"] / ref_slope for b in BAND_ORDER}


def fit(
    input_path: Path,
    *,
    year: int = 2023,
    heating_fuel: str = "Gas",
) -> dict:
    """End-to-end fit, returns a structured result dict."""
    data = load_area_band_data(input_path, year=year, heating_fuel=heating_fuel)

    per_band = {}
    for band in BAND_ORDER:
        per_band[band] = fit_band_slope(data[data["seg3_value"] == band])

    ref_slope = per_band[REFERENCE_BAND]["slope_kwh_per_day_per_HDD"]
    if ref_slope <= 0:
        raise RuntimeError(
            f"Reference band {REFERENCE_BAND!r} fitted non-positive slope "
            f"({ref_slope}); cannot normalise."
        )

    lookup = {
        band: per_band[band]["slope_kwh_per_day_per_HDD"] / ref_slope
        for band in BAND_ORDER
    }
    implied = fit_implied_exponent(
        {b: per_band[b]["slope_kwh_per_day_per_HDD"] for b in BAND_ORDER}
    )

    return {
        "heat_slope_area_bands": lookup,
        "reference_band":         REFERENCE_BAND,
        "implied_exponent":       implied["implied_exponent"],
        "diagnostics": {
            "year":             int(year),
            "heating_fuel":     heating_fuel,
            "source":           str(input_path),
            "per_band":         per_band,
            "implied_exponent_fit": implied,
        },
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path,
                   default=Path("data/serl_8963_targets/daily_targets.csv"))
    p.add_argument("--year", type=int, default=2023)
    p.add_argument("--heating-fuel", type=str, default="Gas")
    p.add_argument("--output", type=Path,
                   default=Path("results/calibration_serl_fits/area_scaling.yaml"))
    p.add_argument("--bootstrap", type=int, default=0,
                   help="If >0, parametric-bootstrap N draws of the per-band "
                        "multipliers from SERL published SE and write a "
                        "'bootstrap' block (se/CI per band) into the YAML.")
    p.add_argument("--bootstrap-seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: SERL input not found: {args.input}", file=sys.stderr)
        return 2

    result = fit(args.input, year=args.year, heating_fuel=args.heating_fuel)

    if args.bootstrap > 0:
        from bootstrap_bands import bootstrap_leaves, derive_se, merge_point_and_bands
        data = load_area_band_data(args.input, year=args.year, heating_fuel=args.heating_fuel)
        bands = bootstrap_leaves(
            data, _leaf_values, value_col="mean",
            se=derive_se(data, value_col="mean"),
            n_boot=args.bootstrap, seed=args.bootstrap_seed,
        )
        result["bootstrap"] = merge_point_and_bands(result["heat_slope_area_bands"], bands)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        yaml.safe_dump(result, f, sort_keys=False)

    if not args.quiet:
        print(f"Reference band: {result['reference_band']}")
        print(f"Implied power-law exponent (diagnostic): "
              f"{result['implied_exponent']:.3f} "
              f"(R²={result['diagnostics']['implied_exponent_fit']['r2']:.3f})")
        print()
        print(f"{'band':>12} {'slope':>9} {'baseline':>9} {'multiplier':>11} {'r²':>7} {'n_panel':>9}")
        for band in BAND_ORDER:
            d = result["diagnostics"]["per_band"][band]
            mult = result["heat_slope_area_bands"][band]
            print(f"{band:>12} {d['slope_kwh_per_day_per_HDD']:>9.3f} "
                  f"{d['baseline_kwh_per_day']:>9.2f} {mult:>11.3f} "
                  f"{d['r2_weighted']:>7.3f} {d['mean_n_rounded']:>9.0f}")
        print()
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
