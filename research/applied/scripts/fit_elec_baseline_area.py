#!/usr/bin/env python3
"""
Fit per-floor-area electricity baseline multipliers from SERL monthly data.

Background
----------
``agent.py:_baseline_area_multiplier`` (line ~397) scales the per-dwelling
baseline anchor by ``(area / 70)^0.20`` clipped to [0.85, 1.25]. Both the
exponent and the clip are unsourced and strangle the SERL signal — SERL
``Electricity imports × floor_area_m2`` monthly rows show baseline
electricity actually spans 0.75× (50 m²) to 2.4× (>200 m²) of the 51-100 m²
reference band, far wider than the clipped power-law allowed.

This is the electricity-side analogue of ``fit_area_scaling.py`` (which
handled gas heating slope). The signal is dominated by area-correlated
standby load (fridges, freezers, modems, hot-water tanks, security
systems) and is behaviour-invariant: per-m² slope at 03:00 ≈ per-m²
slope at 15:00 (ratio 0.99 on SERL 2023 diurnal data), which is exactly
what one expects from always-on standby that scales with home size
rather than with people moving around.

Together with the partialled per-person fit in ``fit_presence_spikes.py``
(see that module's docstring on the cross-segmentation unconfounding),
this lookup gives the model a properly-decomposed electric baseline:

    baseline_per_dwelling = anchor × area_multiplier(dwelling_area)
    + Σ_residents add_person_load(state)

where ``area_multiplier`` captures area-correlated standby and the
per-person term captures behavioural load only.

Method
------
- Filter to ``quantity='Electricity imports'``, ``has_pv='All'``,
  ``period_type='monthly'``, ``seg3_var='floor_area_m2'``,
  ``heating_fuel='All'``, year 2023.
- Bands: ``50 or less``, ``51 to 100`` (reference), ``101 to 150``,
  ``151 to 200``, ``Over 200``.
- Per band: take the weighted mean daily kWh across the 12 monthly rows
  (weights = sqrt(n_rounded)). Convert to mean kWh/h.
- Normalise against the reference-band mean. Output dict matches the
  shape of ``heat_slope_area_bands`` so agent.py can reuse the existing
  band lookup helper.

Output
------
``results/calibration_serl_fits/elec_baseline_area.yaml`` containing
the lookup, implied power-law exponent (diagnostic), and per-band
diagnostics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


BAND_ORDER = ["50 or less", "51 to 100", "101 to 150", "151 to 200", "Over 200"]
REFERENCE_BAND = "51 to 100"
BAND_MIDPOINTS_M2 = {
    "50 or less":  40.0,
    "51 to 100":   75.0,
    "101 to 150": 125.0,
    "151 to 200": 175.0,
    "Over 200":   240.0,
}


def load_area_band_data(
    input_path: Path,
    *,
    year: int,
    has_pv: str = "All",
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    sub = df[
        (df["quantity"] == "Electricity imports")
        & (df["year"] == year)
        & (df["has_pv"] == has_pv)
        & (df["seg3_var"] == "floor_area_m2")
        & (df["period_type"] == "monthly")
        & (df["heating_fuel"] == "All")
    ].copy()
    if sub.empty:
        raise RuntimeError(
            f"No SERL Electricity-imports × floor_area_m2 monthly rows "
            f"for year={year} in {input_path}"
        )
    missing = set(BAND_ORDER) - set(sub["seg3_value"].unique())
    if missing:
        raise RuntimeError(f"Missing area bands in SERL data: {missing}")
    return sub[["seg3_value", "month", "mean", "n_rounded", "sd", "se"]]


def weighted_mean_kwh_per_band(rows: pd.DataFrame) -> dict:
    w = np.sqrt(rows["n_rounded"].to_numpy(dtype=float))
    y = rows["mean"].to_numpy(dtype=float)
    mean = float(np.average(y, weights=w ** 2))
    return {
        "mean_kwh_per_day":  mean,
        "mean_kwh_per_hour": mean / 24.0,
        "n_obs":             int(len(rows)),
        "mean_n_rounded":    float(rows["n_rounded"].mean()),
    }


def fit_implied_exponent(per_band: dict) -> dict:
    """OLS regress log(kwh_per_hour) ~ log(area_midpoint). Diagnostic only."""
    bands = [b for b in BAND_ORDER if per_band[b]["mean_kwh_per_hour"] > 0]
    x = np.log(np.array([BAND_MIDPOINTS_M2[b] for b in bands]))
    y = np.log(np.array([per_band[b]["mean_kwh_per_hour"] for b in bands]))
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
    """Config-bound leaves (per-area-band baseline elec multipliers).

    Reused by the point fit and the parametric bootstrap.
    """
    per_band = {b: weighted_mean_kwh_per_band(data[data["seg3_value"] == b]) for b in BAND_ORDER}
    ref = per_band[REFERENCE_BAND]["mean_kwh_per_hour"]
    if ref <= 0:
        raise RuntimeError(f"Reference band {REFERENCE_BAND!r} fitted non-positive mean ({ref}).")
    return {b: per_band[b]["mean_kwh_per_hour"] / ref for b in BAND_ORDER}


def fit(input_path: Path, *, year: int = 2023) -> dict:
    data = load_area_band_data(input_path, year=year)
    per_band = {b: weighted_mean_kwh_per_band(data[data["seg3_value"] == b]) for b in BAND_ORDER}

    ref = per_band[REFERENCE_BAND]["mean_kwh_per_hour"]
    if ref <= 0:
        raise RuntimeError(f"Reference band {REFERENCE_BAND!r} fitted non-positive mean ({ref}).")

    lookup = {b: per_band[b]["mean_kwh_per_hour"] / ref for b in BAND_ORDER}
    implied = fit_implied_exponent(per_band)

    return {
        "baseline_elec_area_bands": lookup,
        "reference_band":           REFERENCE_BAND,
        "implied_exponent":         implied["implied_exponent"],
        "diagnostics": {
            "year":              int(year),
            "source":            str(input_path),
            "per_band":          per_band,
            "implied_exponent_fit": implied,
        },
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path,
                   default=Path("data/serl_8963_targets/daily_targets.csv"))
    p.add_argument("--year", type=int, default=2023)
    p.add_argument("--output", type=Path,
                   default=Path("results/calibration_serl_fits/elec_baseline_area.yaml"))
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

    result = fit(args.input, year=args.year)

    if args.bootstrap > 0:
        from bootstrap_bands import bootstrap_leaves, derive_se, merge_point_and_bands
        data = load_area_band_data(args.input, year=args.year)
        bands = bootstrap_leaves(
            data, _leaf_values, value_col="mean",
            se=derive_se(data, value_col="mean"),
            n_boot=args.bootstrap, seed=args.bootstrap_seed,
        )
        result["bootstrap"] = merge_point_and_bands(result["baseline_elec_area_bands"], bands)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        yaml.safe_dump(result, f, sort_keys=False)

    if not args.quiet:
        print(f"Reference band: {result['reference_band']}")
        print(f"Implied power-law exponent (diagnostic): "
              f"{result['implied_exponent']:.3f} "
              f"(R²={result['diagnostics']['implied_exponent_fit']['r2']:.3f})")
        print()
        print(f"{'band':>12} {'kWh/day':>9} {'kWh/h':>8} {'multiplier':>11} {'n_panel':>9}")
        for band in BAND_ORDER:
            d = result["diagnostics"]["per_band"][band]
            mult = result["baseline_elec_area_bands"][band]
            print(f"{band:>12} {d['mean_kwh_per_day']:>9.2f} {d['mean_kwh_per_hour']:>8.4f} "
                  f"{mult:>11.3f} {d['mean_n_rounded']:>9.0f}")
        print()
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
