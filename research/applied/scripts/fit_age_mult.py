#!/usr/bin/env python3
"""
Fit per-building-age-band heating slope multipliers from SERL daily targets.

Background
----------
Older buildings have higher heat-loss rates and therefore steeper gas-use
response to HDD. SERL exposes this directly via
``seg3_var='building_age'``, with 7 era bands plus a No-data bucket.

This script mirrors ``fit_area_scaling.py`` and ``fit_sap_band_mult.py``:
weighted-OLS per band of ``daily_kwh = baseline + slope · HDD``, with
multipliers normalised to the reference band ``1950 - 1975`` (modal in
the SERL panel at ~1,927/month).

The output populates ``building_age_mult_heating_gas`` — currently empty
in v5 phase-2 configs because Phase 1 cleared the v2 hand-tuned lookup
and Phase 2 only fitted floor area. This script (with ``fit_sap_band_mult``)
closes the remaining composition lift.

Caveat: SAP and age are correlated in the SERL panel (older buildings
skew worse SAP). Applying both this lookup and the SAP one marginally
will absorb some shared signal twice. Pass --skip flags in
``calibrate_serl.py`` if attribution gets confusing; otherwise validate
by probe + three-city transfer and adjust if Newcastle High-tier
overshoots.

Method
------
- Filter to ``quantity='Gas'``, ``heating_fuel='Gas'``, ``has_pv='All'``,
  ``period_type='monthly'``, ``seg3_var='building_age'``, year 2023.
- Bands: ``Before 1900``, ``1900 - 1929``, ``1930 - 1949``,
  ``1950 - 1975`` (reference), ``1976 - 1990``, ``1991 - 2002``,
  ``2003 onwards``. ``No data`` gets multiplier 1.0 — it's a missing-data
  bucket, not an era.
- Per-band weighted OLS, weights = sqrt(n_rounded).

Output
------
``results/calibration_serl_fits/age_mult.yaml`` containing the lookup
and per-band diagnostics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


BAND_ORDER = [
    "Before 1900",
    "1900 - 1929",
    "1930 - 1949",
    "1950 - 1975",
    "1976 - 1990",
    "1991 - 2002",
    "2003 onwards",
]
REFERENCE_BAND = "1950 - 1975"
NO_DATA_BAND = "No data"  # set to 1.0 in output; not era-meaningful


def load_age_band_data(
    input_path: Path,
    *,
    year: int,
    quantity: str = "Gas",
    heating_fuel: str = "Gas",
    has_pv: str = "All",
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    sub = df[
        (df["quantity"] == quantity)
        & (df["year"] == year)
        & (df["heating_fuel"] == heating_fuel)
        & (df["has_pv"] == has_pv)
        & (df["seg3_var"] == "building_age")
        & (df["period_type"] == "monthly")
    ].copy()
    if sub.empty:
        raise RuntimeError(
            f"No SERL building_age monthly rows for year={year} "
            f"heating_fuel={heating_fuel!r} in {input_path}"
        )
    missing = set(BAND_ORDER) - set(sub["seg3_value"].unique())
    if missing:
        raise RuntimeError(f"Missing age bands in SERL data: {missing}")
    return sub[["seg3_value", "month", "mean", "mean_hdd", "n_rounded", "sd", "se"]]


def fit_band_slope(rows: pd.DataFrame) -> dict:
    """Weighted OLS of ``daily_kwh = baseline + slope · HDD`` for one band."""
    hdd = rows["mean_hdd"].to_numpy(dtype=float)
    kwh = rows["mean"].to_numpy(dtype=float)
    w = np.sqrt(rows["n_rounded"].to_numpy(dtype=float))
    X = np.column_stack([np.ones_like(hdd), hdd])
    W2 = np.diag(w ** 2)
    beta = np.linalg.solve(X.T @ W2 @ X, X.T @ W2 @ kwh)
    baseline, slope = float(beta[0]), float(beta[1])
    pred = X @ beta
    resid = kwh - pred
    ss_res = float(np.sum((resid * w) ** 2))
    ybar = float(np.average(kwh, weights=w ** 2))
    ss_tot = float(np.sum(((kwh - ybar) * w) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "baseline_kwh_per_day":      baseline,
        "slope_kwh_per_day_per_HDD": slope,
        "r2_weighted":               r2,
        "n_obs":                     int(len(rows)),
        "mean_n_rounded":            float(rows["n_rounded"].mean()),
    }


def _leaf_values(data: pd.DataFrame) -> dict:
    """Config-bound leaves (per-age-band slope multipliers) from a band table.

    Reused by the point fit and the parametric bootstrap. The ``No data``
    bucket is fixed at 1.0 by construction, so it carries a zero band.
    """
    per_band = {b: fit_band_slope(data[data["seg3_value"] == b]) for b in BAND_ORDER}
    ref_slope = per_band[REFERENCE_BAND]["slope_kwh_per_day_per_HDD"]
    if ref_slope <= 0:
        raise RuntimeError(
            f"Reference band {REFERENCE_BAND!r} fitted non-positive slope "
            f"({ref_slope}); cannot normalise."
        )
    lookup = {b: per_band[b]["slope_kwh_per_day_per_HDD"] / ref_slope for b in BAND_ORDER}
    lookup[NO_DATA_BAND] = 1.0
    return lookup


def fit(
    input_path: Path,
    *,
    year: int = 2023,
    heating_fuel: str = "Gas",
) -> dict:
    data = load_age_band_data(input_path, year=year, heating_fuel=heating_fuel)
    per_band = {b: fit_band_slope(data[data["seg3_value"] == b]) for b in BAND_ORDER}

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
    lookup[NO_DATA_BAND] = 1.0  # missing-data bucket: neutral

    return {
        "building_age_mult_heating_gas": lookup,
        "reference_band":                REFERENCE_BAND,
        "diagnostics": {
            "year":         int(year),
            "heating_fuel": heating_fuel,
            "source":       str(input_path),
            "per_band":     per_band,
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
                   default=Path("results/calibration_serl_fits/age_mult.yaml"))
    p.add_argument("--bootstrap", type=int, default=0,
                   help="If >0, parametric-bootstrap N draws of the per-band "
                        "multipliers from SERL published SE and write a "
                        "'bootstrap' block into the YAML.")
    p.add_argument("--bootstrap-seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: SERL input not found: {args.input}", file=sys.stderr)
        return 2

    result = fit(args.input, year=args.year, heating_fuel=args.heating_fuel)

    if args.bootstrap > 0:
        from bootstrap_bands import bootstrap_leaves, derive_se, merge_point_and_bands
        data = load_age_band_data(args.input, year=args.year, heating_fuel=args.heating_fuel)
        bands = bootstrap_leaves(
            data, _leaf_values, value_col="mean",
            se=derive_se(data, value_col="mean"),
            n_boot=args.bootstrap, seed=args.bootstrap_seed,
        )
        result["bootstrap"] = merge_point_and_bands(result["building_age_mult_heating_gas"], bands)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        yaml.safe_dump(result, f, sort_keys=False)

    if not args.quiet:
        print(f"Reference band: {result['reference_band']}")
        print()
        print(f"{'band':>14} {'slope':>8} {'baseline':>9} {'multiplier':>11} {'r²':>7} {'n_panel':>9}")
        for band in BAND_ORDER:
            d = result["diagnostics"]["per_band"][band]
            mult = result["building_age_mult_heating_gas"][band]
            print(f"{band:>14} {d['slope_kwh_per_day_per_HDD']:>8.3f} "
                  f"{d['baseline_kwh_per_day']:>9.2f} {mult:>11.3f} "
                  f"{d['r2_weighted']:>7.3f} {d['mean_n_rounded']:>9.0f}")
        print(f"{'No data':>14} {'—':>8} {'—':>9} {1.0:>11.3f} {'—':>7} {'—':>9}  (neutral; missing-data bucket)")
        print()
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
