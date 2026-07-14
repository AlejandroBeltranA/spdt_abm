#!/usr/bin/env python3
"""
Fit the baseline electricity diurnal profile from SERL hourly data.

Background — the shape gap v5 left open
---------------------------------------
v5 calibrates the electricity *level* well: ``fit_presence_spikes.py`` and
``fit_elec_baseline_area.py`` pin the per-person load and the baseline anchor
to SERL. But both reduce the 24-hour signal to scalars:

  - ``fit_presence_spikes`` collapses the per-hour per-person slopes into two
    windows (awake 07-22, sleep 23-06) → two numbers.
  - the per-m² standby goes into a single flat ``baseline_anchor`` applied
    every hour by ``model._reset_base_loads``.

So the model's electricity is a *box*: a flat baseline plus a two-state
presence step. Against the SERL diurnal curve (a ~2.5x swing from a 05:00
trough to an 18:00 peak) that reads as "night too high, evening peak too
shallow" — the two errors are fitted to the daily mean, so they cancel in the
annual total but the shape is wrong.

What this script fits
---------------------
A single mean-1.0, 24-element profile, ``base_profile_24h_electric``, that
``model._reset_base_loads`` multiplies onto the per-hour baseline. Because the
profile is normalised to mean 1.0, it reshapes the baseline across the day
**without changing its daily mean** — so the annual total, the baseline anchor,
and every other v5-calibrated parameter are untouched. It only adds the time
dimension v5 collapsed.

Target slice
------------
SERL ``Electricity imports``, ``seg3_var='none'`` (the ungrouped population
aggregate — the full panel, ~655k home-days), ``has_pv='All'``,
``heating_fuel='All'``, ``weekday_weekend='both'``, for the calibration year.
The shape is dominated by appliance/plug load; electric-heated homes (~8% of
GB) contribute a little, but in the model their heating keeps its own
``heating_profile_24h`` — this profile only ever touches the baseline channel,
so there is no double-counting.

Gas
---
The total-gas diurnal shape is dominated by *space heating*, which the model
already shapes via ``heating_profile_24h`` and the climate signal. Applying the
total-gas shape to the (cooking) gas baseline would double-count heating, so
``base_profile_24h_gas`` is left flat here. The gas diurnal belongs in a
separate refit of ``heating_profile_24h`` as a residual over the climate
signal — see RUNBOOK; not done in this script.

Output
------
``results/calibration_serl_fits/diurnal_profile.yaml`` with the 24-element
profile and a per-hour diagnostic table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


def fit_profile(
    input_path: Path,
    *,
    quantity: str = "Electricity imports",
    year: int = 2023,
) -> dict:
    """Return the mean-1.0 24h baseline diurnal profile for one fuel quantity."""
    df = pd.read_csv(input_path)
    sub = df[
        (df["quantity"] == quantity)
        & (df["year"] == year)
        & (df["seg3_var"] == "none")
        & (df["has_pv"] == "All")
        & (df["heating_fuel"] == "All")
        & (df["weekday_weekend"] == "both")
    ].copy()
    if sub.empty:
        raise RuntimeError(
            f"No SERL '{quantity}' x none rows for year={year} in {input_path}"
        )
    sub = sub.sort_values("hour")
    hours = sub["hour"].to_numpy(dtype=int)
    if not np.array_equal(hours, np.arange(24)):
        raise RuntimeError(
            f"Expected hours 0..23 for '{quantity}' year={year}, got {hours.tolist()}"
        )
    mean_kwh = sub["mean_kwh"].to_numpy(dtype=float)
    day_mean = float(mean_kwh.mean())
    if not np.isfinite(day_mean) or day_mean <= 0:
        raise RuntimeError(f"Non-positive daily mean for '{quantity}' year={year}")
    profile = mean_kwh / day_mean  # mean-1.0 by construction

    return {
        "profile": [float(x) for x in profile],
        "diagnostics": {
            "quantity": quantity,
            "year": int(year),
            "source": str(input_path),
            "seg3_var": "none",
            "daily_mean_kwh_per_hour": day_mean,
            "swing_max_over_min": float(profile.max() / profile.min()),
            "peak_hour": int(np.argmax(profile)),
            "trough_hour": int(np.argmin(profile)),
            "panel_n_home_days": float(sub["n_rounded"].mean()),
            "per_hour_kwh": [float(x) for x in mean_kwh],
            "method": (
                "Population-aggregate SERL diurnal mean (seg3_var='none'), "
                "normalised to mean 1.0 so it is mean-preserving when applied "
                "to the baseline anchor in model._reset_base_loads."
            ),
        },
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/serl_8963_targets/diurnal_targets_hourly_mean.csv"),
    )
    p.add_argument("--year", type=int, default=2023)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/calibration_serl_fits/diurnal_profile.yaml"),
    )
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: SERL input not found: {args.input}", file=sys.stderr)
        return 2

    elec = fit_profile(args.input, quantity="Electricity imports", year=args.year)

    result = {
        "base_profile_24h_electric": elec["profile"],
        # Gas baseline left flat — total-gas diurnal is heating-dominated and is
        # the province of heating_profile_24h, not the cooking baseline.
        "base_profile_24h_gas": [1.0] * 24,
        "diagnostics": {"electric": elec["diagnostics"]},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        yaml.safe_dump(result, f, sort_keys=False)

    if not args.quiet:
        d = elec["diagnostics"]
        print(f"Electricity baseline diurnal profile (mean-1.0), year {args.year}:")
        print(
            f"  swing {d['swing_max_over_min']:.2f}x  "
            f"peak h{d['peak_hour']}={elec['profile'][d['peak_hour']]:.3f}  "
            f"trough h{d['trough_hour']}={elec['profile'][d['trough_hour']]:.3f}"
        )
        print("  profile: " + " ".join(f"{x:.3f}" for x in elec["profile"]))
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
