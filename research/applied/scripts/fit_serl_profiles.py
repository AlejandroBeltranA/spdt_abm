#!/usr/bin/env python3
"""
Fit the full SERL diurnal "profile pack" for the schedule-retired architecture.

The model no longer simulates per-person leave/return/wake/sleep clocks. Instead
the diurnal shape comes straight from SERL, scaled by each dwelling's static
attributes (occupant count, area, type, fuel) and driven by climate for heat:

    electric(h) = E_base · base_profile_electric(h)
                + n_occ · per_person_electric(h)
    gas(h)      = G_base · base_profile_gas(h)              # cooking / DHW
                + slope · HD(h) · heating_profile(h | T)    # space heating

This script fits the four SERL profiles (no scalars; each is a normalised
empirical shape):

1. base_profile_electric(h) — population-aggregate electricity diurnal
   (seg3_var='none'), mean-1.0.
2. per_person_electric(h)   — marginal electricity per added occupant by hour,
   from the num_occupants weighted-OLS slope (reuses fit_presence_spikes), mean-1.0.
   Replaces the awake/sleep step.
3. base_profile_gas(h)      — the COOKING/DHW gas diurnal, isolated as the
   warm-band (15-20 C, above the ~16.5 C balance point → ~no space heat) gas
   shape, mean-1.0. Fixes the overnight gas floor.
4. heating_profile[band](h) — space-heating diurnal per outdoor-temperature
   band, isolated by subtracting the cooking baseline (in absolute Wh, since
   cooking is ~temperature-invariant) then normalising mean-1.0. Reproduces the
   cold-flat / mild-peaky HDD interaction by construction. Runtime interpolates
   by ambient temperature.

Output
------
``results/calibration_serl_fits/serl_profiles.yaml``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_THIS = Path(__file__).resolve()
REPO = _THIS.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(_THIS.parent))

import fit_presence_spikes as fps  # noqa: E402

DIURNAL = REPO / "data/serl_8963_targets/diurnal_targets_hourly_mean.csv"

# Outdoor-temperature bands (SERL seg3_value) → representative midpoint (C).
# 15_to_20 is above the ~16.5 C balance point → treated as cooking-only.
HEATING_BANDS = {"-5_to_0": -2.5, "0_to_5": 2.5, "5_to_10": 7.5, "10_to_15": 12.5}
COOKING_BAND = "15_to_20"


def _slice(df, quantity, seg3_var, seg3_value=None):
    s = df[(df.quantity == quantity) & (df.seg3_var == seg3_var)
           & (df.has_pv == "All") & (df.heating_fuel == "All")
           & (df.weekday_weekend == "both")]
    if seg3_value is not None:
        s = s[s.seg3_value == seg3_value]
    return s.sort_values("hour")


def _hourly(s) -> np.ndarray:
    v = s.set_index("hour")["mean_kwh"].reindex(range(24)).to_numpy(float)
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--output", type=Path,
                    default=REPO / "results/calibration_serl_fits/serl_profiles.yaml")
    args = ap.parse_args()
    df = pd.read_csv(DIURNAL)
    df = df[df.year == args.year]

    def norm(v):  # mean-1.0
        return v / v.mean()

    # 1. base_profile_electric
    elec_base = norm(_hourly(_slice(df, "Electricity imports", "none")))

    # 2. per_person_electric — from the num_occupants per-hour OLS slope.
    # Emitted in ABSOLUTE kWh/occupant/hour (not normalised): the model adds
    # (n_occ - panel_mean) * slope(h), a deviation-from-mean term consistent
    # with v5's anchor recentring (which baked average per-person into E_base).
    res = fps.fit(DIURNAL, year=args.year)
    ph = res["diagnostics"]["per_hour_occ"]
    slope = np.array([ph[h]["per_person_kwh_per_hour"] for h in range(24)])
    per_person = norm(slope)
    panel_mean_occupants = float(res.get("panel_mean_occupants", float("nan")))

    # 3. base_profile_gas — warm-band cooking/DHW shape (absolute → normalise)
    cooking_abs = _hourly(_slice(df, "Gas", "temperature_band", COOKING_BAND))
    gas_base = norm(cooking_abs)

    # 4. heating_profile per temperature band — cooking-subtracted, mean-1.0
    heating = {}
    for band, tmid in HEATING_BANDS.items():
        gas_b_abs = _hourly(_slice(df, "Gas", "temperature_band", band))
        heat_abs = np.maximum(gas_b_abs - cooking_abs, 0.0)
        if heat_abs.sum() <= 0:
            continue
        heating[band] = {
            "temp_mid_C": tmid,
            "profile": [float(x) for x in norm(heat_abs)],
            "swing": float(norm(heat_abs).max() / max(norm(heat_abs).min(), 1e-6)),
        }

    result = {
        "base_profile_24h_electric": [float(x) for x in elec_base],
        "per_person_profile_24h_electric": [float(x) for x in per_person],
        "per_person_slope_24h_electric": [float(x) for x in slope],
        "panel_mean_occupants": panel_mean_occupants,
        "base_profile_24h_gas": [float(x) for x in gas_base],
        "heating_temp_profile": {
            b: {"temp_mid_C": d["temp_mid_C"], "profile": d["profile"]}
            for b, d in heating.items()
        },
        "diagnostics": {
            "year": int(args.year),
            "cooking_band": COOKING_BAND,
            "heating_bands": list(heating.keys()),
            "electric_base_swing": float(elec_base.max() / elec_base.min()),
            "per_person_swing": float(per_person.max() / per_person.min()),
            "gas_cooking_swing": float(gas_base.max() / gas_base.min()),
            "heating_band_swings": {b: d["swing"] for b, d in heating.items()},
            "method": (
                "All profiles are mean-1.0 normalised SERL empirical shapes. "
                "Cooking isolated as the warm band (above balance point); "
                "per-band heating = (band gas - cooking) in absolute Wh, "
                "normalised. No scalars."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        yaml.safe_dump(result, f, sort_keys=False)

    print(f"electric base swing   {result['diagnostics']['electric_base_swing']:.2f}x")
    print(f"per-person swing      {result['diagnostics']['per_person_swing']:.2f}x")
    print(f"gas cooking swing     {result['diagnostics']['gas_cooking_swing']:.2f}x")
    print("heating profile by temperature band (mean-1.0):")
    print(f"{'h':>3} " + " ".join(f"{b:>9}" for b in heating))
    for h in range(24):
        print(f"{h:>3} " + " ".join(f"{heating[b]['profile'][h]:>9.2f}" for b in heating))
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
