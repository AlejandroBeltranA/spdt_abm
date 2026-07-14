#!/usr/bin/env python3
"""
Fit per-person electricity load and the awake/sleep diurnal multiplier
from SERL diurnal electricity-by-occupancy data.

Background
----------
``agent.py:add_person_load`` returns a kWh/h "load" for one person under
their current presence state:

    if at_home:
        load = energy_per_person_home          # default 0.06 kWh/h
        load *= awake_home_spike_mult          # default 1.0 (awake)
        load *= sleep_home_spike_mult          # default 0.3 (asleep)
    else:
        load = energy_per_person_away          # default 0.01 kWh/h

All four were unsourced. SERL exposes `Electricity imports` segmented
by `num_occupants` at hourly resolution; differencing across occupant
counts gives the per-person electricity contribution per hour.

What this script fits — cross-segmentation unconfounding
--------------------------------------------------------
A naive ``kwh = baseline + slope × n_persons`` regression per hour
confounds two things:

  (a) genuine behavioural per-person load (people moving around, using
      devices, cooking, etc.), and
  (b) household-size-correlated standby load (larger households live in
      larger homes with more always-on devices: fridges, freezers,
      modems, hot-water tanks, security systems).

A single household-level intercept across occupant bands can't soak up
(b) because larger-occupant bands have higher mean floor area in the
panel. The slope captures both, most visibly at night when nothing
varies behaviourally per person but standby keeps running.

SERL's diurnal data fortunately exposes ``Electricity imports`` also
segmented by ``floor_area_m2`` (4,320 rows across 4 years). Two key
empirical observations from 2023 SERL:

  - Per-m² slope is **behaviour-invariant**: 0.00234 kWh/h/m² at sleep
    hours, 0.00233 kWh/h/m² at awake hours (ratio 0.99). Exactly the
    signature of always-on standby that scales with home size, not with
    people moving around.
  - Per-occupant slope is **behaviour-dependent**: 0.067 kWh/h/person at
    sleep, 0.098 kWh/h/person at awake (ratio 0.68). The 32% awake
    surplus is the behavioural component; the 0.067 baseline is what's
    really being captured at night, which is mostly (b).

Unconfounding at sleep hours (behaviour ≈ 0):

    sleep_slope ≈ per_m²_standby × Δarea_per_added_person_in_panel
    0.067       ≈ 0.00234       × Δa/Δn
    → Δa/Δn ≈ 29 m²/added person  (panel-level area-per-occupant)

This is consistent with UK English Housing Survey data: 1-person ≈ 50
m², 4-person ≈ 130 m². The 29 m²/person panel slope is reasonable.

Pure behavioural per-person at awake hours then falls out:

    awake_slope = per_m²_standby × Δa/Δn + behavioural_awake
    0.098       = 0.067               + behavioural_awake
    → behavioural_awake ≈ 0.031 kWh/h/person

And at sleep behavioural ≈ 0 (consistent with everyone asleep — phones
charging on a per-person basis is real but small).

Outputs
-------
  - ``energy_per_person_home``   : 0.031 kWh/h/person (awake behavioural,
                                   partialled from per-m² standby)
  - ``awake_home_spike_mult``    : 1.0 by definition (reference)
  - ``sleep_home_spike_mult``    : 0.0 (sleep behavioural ≈ 0)
  - ``energy_per_person_away``   : 0.01 kWh/h/person (literature default)

Per-m² standby goes into the baseline anchor via
``fit_elec_baseline_area.py``, NOT here.

Method
------
1. Filter SERL diurnal to ``quantity='Electricity imports'``,
   ``seg3_var='num_occupants'``, ``has_pv='All'``, ``heating_fuel='All'``,
   year 2023 (panel coverage is whole-year by design here; SERL doesn't
   segment this view by weekday/weekend).
2. For each hour h in 0..23: weighted-OLS regress
   ``kwh = baseline + slope x num_occupants`` across the occupant bands
   {1, 2, 3, 4, 5, >=6}. >=6 is treated as 6 for the regression. Weights
   = sqrt(n_rounded) per band.
3. The fitted slope at hour h is the marginal kWh/h per added person.
4. Awake hours: 07-22 inclusive (UK awake-at-home window per ONS Time
   Use Survey 2014-15 round-trip averages). Sleep hours: 23-06.
   ``energy_per_person_home`` = weighted-mean (by panel n) of awake-hour
   slopes. ``sleep_home_spike_mult`` = mean sleep slope / mean awake slope.

Output
------
``results/calibration_serl_fits/presence_spikes.yaml`` with the three
fitted parameters and a per-hour diagnostic table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


# SERL occupant bands → integer counts for the per-person regression.
# >=6 is rare (~120/month panel) and is approximated as exactly 6.
BAND_TO_COUNT = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, ">=6": 6}

# Awake / sleep windows. Default 07-22 awake (16 hours), 23-06 sleep
# (8 hours), aligning with ONS Time Use Survey UK norms.
AWAKE_HOURS = list(range(7, 23))   # 7..22 inclusive
SLEEP_HOURS = [23] + list(range(0, 7))  # 23, 0..6

# Literature / practice default for the parameter SERL cannot identify.
ENERGY_PER_PERSON_AWAY_DEFAULT = 0.01  # kWh/h/person; small standing load.

# SERL Electricity × floor_area_m2 area-band midpoints (for the per-m²
# standby fit). Mirrors `fit_elec_baseline_area.py`.
AREA_BAND_MIDPOINTS = {
    "50 or less":  40.0,
    "51 to 100":   75.0,
    "101 to 150": 125.0,
    "151 to 200": 175.0,
    "Over 200":   240.0,
}


def load_occupant_diurnal(
    input_path: Path,
    *,
    year: int,
) -> pd.DataFrame:
    """Load Electricity imports x num_occupants hourly rows for one year."""
    df = pd.read_csv(input_path)
    sub = df[
        (df["quantity"] == "Electricity imports")
        & (df["year"] == year)
        & (df["seg3_var"] == "num_occupants")
        & (df["has_pv"] == "All")
        & (df["heating_fuel"] == "All")
    ].copy()
    if sub.empty:
        raise RuntimeError(
            f"No SERL Electricity-imports x num_occupants rows for year={year} "
            f"in {input_path}"
        )
    sub = sub[sub["seg3_value"].isin(BAND_TO_COUNT.keys())].copy()
    sub["n_persons"] = sub["seg3_value"].map(BAND_TO_COUNT).astype(int)
    return sub[["hour", "seg3_value", "n_persons", "mean_kwh", "n_rounded"]]


def load_area_diurnal(input_path: Path, *, year: int) -> pd.DataFrame:
    """Load Electricity imports × floor_area_m2 hourly rows (for the per-m² standby fit)."""
    df = pd.read_csv(input_path)
    sub = df[
        (df["quantity"] == "Electricity imports")
        & (df["year"] == year)
        & (df["seg3_var"] == "floor_area_m2")
        & (df["has_pv"] == "All")
        & (df["heating_fuel"] == "All")
    ].copy()
    if sub.empty:
        raise RuntimeError(
            f"No SERL Electricity-imports × floor_area_m2 rows for year={year} "
            f"in {input_path}"
        )
    sub = sub[sub["seg3_value"].isin(AREA_BAND_MIDPOINTS.keys())].copy()
    sub["area_m2"] = sub["seg3_value"].map(AREA_BAND_MIDPOINTS).astype(float)
    return sub[["hour", "seg3_value", "area_m2", "mean_kwh", "n_rounded"]]


def fit_hourly_area_slope(rows: pd.DataFrame) -> dict:
    """Weighted OLS of ``kwh = baseline + slope × area_m²`` for one hour."""
    x = rows["area_m2"].to_numpy(dtype=float)
    y = rows["mean_kwh"].to_numpy(dtype=float)
    w = np.sqrt(rows["n_rounded"].to_numpy(dtype=float))
    X = np.column_stack([np.ones_like(x), x])
    W2 = np.diag(w ** 2)
    beta = np.linalg.solve(X.T @ W2 @ X, X.T @ W2 @ y)
    return {
        "baseline_kwh_per_hour":      float(beta[0]),
        "per_m2_kwh_per_hour_per_m2": float(beta[1]),
        "n_panel_mean":               float(rows["n_rounded"].mean()),
    }


def fit_hourly_slope(rows: pd.DataFrame) -> dict:
    """Weighted OLS of ``kwh = baseline + slope x n_persons`` for one hour."""
    x = rows["n_persons"].to_numpy(dtype=float)
    y = rows["mean_kwh"].to_numpy(dtype=float)
    w = np.sqrt(rows["n_rounded"].to_numpy(dtype=float))
    X = np.column_stack([np.ones_like(x), x])
    W2 = np.diag(w ** 2)
    beta = np.linalg.solve(X.T @ W2 @ X, X.T @ W2 @ y)
    baseline, slope = float(beta[0]), float(beta[1])
    pred = X @ beta
    ss_res = float(np.sum(((y - pred) * w) ** 2))
    ybar = float(np.average(y, weights=w ** 2))
    ss_tot = float(np.sum(((y - ybar) * w) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "baseline_kwh_per_hour": baseline,
        "per_person_kwh_per_hour": slope,
        "r2_weighted": r2,
        "n_panel_mean": float(rows["n_rounded"].mean()),
    }


def _compute_from_data(occ_data: pd.DataFrame, area_data: pd.DataFrame) -> dict:
    """The unconfounding compute, factored out of :func:`fit` so the parametric
    bootstrap can re-run it on perturbed copies of the two diurnal tables.
    """
    per_hour_occ = {}
    for h in range(24):
        rows = occ_data[occ_data["hour"] == h]
        if len(rows) < 3:
            continue
        per_hour_occ[h] = fit_hourly_slope(rows)

    # ── Per-m² standby slope per hour (behaviour-invariant signature) ──
    per_hour_area = {}
    for h in range(24):
        rows = area_data[area_data["hour"] == h]
        if len(rows) < 3:
            continue
        per_hour_area[h] = fit_hourly_area_slope(rows)

    # Aggregate slopes across awake / sleep windows (weighted by panel n).
    def _wmean(per_hour, hours, key):
        vals = np.array([per_hour[h][key] for h in hours if h in per_hour])
        wts  = np.array([per_hour[h]["n_panel_mean"] for h in hours if h in per_hour])
        return float(np.average(vals, weights=wts))

    naive_per_person_awake = _wmean(per_hour_occ, AWAKE_HOURS, "per_person_kwh_per_hour")
    naive_per_person_sleep = _wmean(per_hour_occ, SLEEP_HOURS, "per_person_kwh_per_hour")
    per_m2_awake = _wmean(per_hour_area, AWAKE_HOURS, "per_m2_kwh_per_hour_per_m2")
    per_m2_sleep = _wmean(per_hour_area, SLEEP_HOURS, "per_m2_kwh_per_hour_per_m2")

    # ── Unconfounding ──
    # Sleep hours: behaviour ≈ 0, so naive_per_person_sleep ≈ per_m²_sleep × Δa/Δn.
    # Δa/Δn is the panel-level area-per-added-person.
    delta_area_per_person = naive_per_person_sleep / per_m2_sleep

    # Awake behavioural per-person = naive_awake − (per_m²_awake × Δa/Δn)
    behavioural_per_person_awake = (
        naive_per_person_awake - per_m2_awake * delta_area_per_person
    )

    # Sleep behavioural per-person is the residual after partialling. By
    # construction it's ≈ 0 (that was the identifying assumption); we
    # compute it anyway as a diagnostic. Clamp to ≥ 0.
    behavioural_per_person_sleep = max(
        0.0,
        naive_per_person_sleep - per_m2_sleep * delta_area_per_person,
    )

    energy_per_person_home = float(behavioural_per_person_awake)
    sleep_home_spike_mult = float(
        behavioural_per_person_sleep / behavioural_per_person_awake
    ) if behavioural_per_person_awake > 0 else 0.0

    # SERL panel-mean occupants — weighted by n_rounded per occupant band.
    # Needed downstream by calibrate_serl.py to recentre the electricity
    # baseline anchor when switching from naive to partialled per-person.
    panel_counts = occ_data.groupby("n_persons")["n_rounded"].sum().to_dict()
    n_panel = sum(n * w for n, w in panel_counts.items())
    w_panel = sum(panel_counts.values())
    panel_mean_occupants = n_panel / w_panel if w_panel > 0 else float("nan")

    return {
        "energy_per_person_home":   energy_per_person_home,
        "awake_home_spike_mult":    1.0,
        "sleep_home_spike_mult":    sleep_home_spike_mult,
        "energy_per_person_away":   ENERGY_PER_PERSON_AWAY_DEFAULT,
        # Surfaced for calibrate_serl.py recentring step:
        "naive_per_person_home_awake": naive_per_person_awake,
        "panel_mean_occupants":        float(panel_mean_occupants),
        "diagnostics": {
            "awake_hours":           AWAKE_HOURS,
            "sleep_hours":           SLEEP_HOURS,
            "naive_per_person_awake_kWh_per_h": naive_per_person_awake,
            "naive_per_person_sleep_kWh_per_h": naive_per_person_sleep,
            "per_m2_standby_awake_kWh_per_h_per_m2": per_m2_awake,
            "per_m2_standby_sleep_kWh_per_h_per_m2": per_m2_sleep,
            "delta_area_per_added_person_m2": delta_area_per_person,
            "behavioural_per_person_awake_kWh_per_h": behavioural_per_person_awake,
            "behavioural_per_person_sleep_kWh_per_h": behavioural_per_person_sleep,
            "per_hour_occ":  per_hour_occ,
            "per_hour_area": per_hour_area,
            "unconfounding_method": (
                "Cross-segmentation partial-out. Per-m² slope is "
                "behaviour-invariant (ratio ≈ 1.0 sleep/awake), so its "
                "sleep-hour value is interpreted as pure standby per m². "
                "Naive per-occupant sleep slope = per_m²_standby × Δa/Δn → "
                "solve for Δa/Δn (panel area-per-added-person). Awake "
                "behavioural = naive_awake − per_m²_standby × Δa/Δn. "
                "Per-m² standby itself goes into the baseline via "
                "fit_elec_baseline_area.py, not here."
            ),
            "energy_per_person_away_source": (
                "Literature default 0.01 kWh/h/person — SERL aggregate diurnal "
                "cannot identify per-person away load."
            ),
        },
    }


def fit(input_path: Path, *, year: int = 2023) -> dict:
    """Load the two diurnal tables and run the unconfounding compute."""
    occ_data = load_occupant_diurnal(input_path, year=year)
    area_data = load_area_diurnal(input_path, year=year)
    result = _compute_from_data(occ_data, area_data)
    result["diagnostics"]["year"] = int(year)
    result["diagnostics"]["source"] = str(input_path)
    return result


# ── Parametric-bootstrap support ──────────────────────────────────────────
#
# The diurnal mean file carries no published dispersion, so we borrow the
# *relative* sampling error SERL publishes for the same segmentation at the
# annual-daily level (daily_targets.csv has se/sd per occupant band and per
# floor-area band for Electricity imports) and map it onto each diurnal row by
# its band. The absolute SE on an hourly mean isn't published; the relative SE
# of the annual band mean is the best-available, SERL-grounded proxy for "how
# precisely is this band's electricity pinned down".

_COMBINED_VALUE_COL = "mean_kwh"


def _band_relative_se(daily_targets_path: Path, *, year: int, seg3_var: str) -> dict:
    """``{seg3_value: se/mean}`` from annual-daily Electricity rows for a segmentation."""
    df = pd.read_csv(daily_targets_path)
    sub = df[
        (df["quantity"] == "Electricity imports")
        & (df["year"] == year)
        & (df["seg3_var"] == seg3_var)
        & (df["has_pv"] == "All")
        & (df["heating_fuel"] == "All")
        & (df["period_type"] == "annual")
    ].copy()
    rel = {}
    for _, r in sub.iterrows():
        m = pd.to_numeric(r.get("mean"), errors="coerce")
        se = pd.to_numeric(r.get("se"), errors="coerce")
        if np.isfinite(m) and np.isfinite(se) and m > 0:
            rel[str(r["seg3_value"])] = float(se / m)
    return rel


def _combined_table(occ_data: pd.DataFrame, area_data: pd.DataFrame,
                    daily_targets_path: Path, *, year: int) -> pd.DataFrame:
    """Stack the occupant and area diurnal tables with a ``_kind`` tag and a
    per-row ``se`` borrowed from annual-band relative SE. The shared bootstrap
    helper perturbs the single ``mean_kwh`` column across both kinds at once.
    """
    rel_occ = _band_relative_se(daily_targets_path, year=year, seg3_var="num_occupants")
    rel_area = _band_relative_se(daily_targets_path, year=year, seg3_var="floor_area_m2")

    o = occ_data.copy()
    o["_kind"] = "occ"
    o["se"] = o["mean_kwh"] * o["seg3_value"].astype(str).map(rel_occ).fillna(0.0)

    a = area_data.copy()
    a["_kind"] = "area"
    a["se"] = a["mean_kwh"] * a["seg3_value"].astype(str).map(rel_area).fillna(0.0)

    return pd.concat([o, a], ignore_index=True)


def _leaf_values(combined: pd.DataFrame) -> dict:
    """Config-bound leaves from a combined (perturbed) diurnal table."""
    occ = combined[combined["_kind"] == "occ"]
    area = combined[combined["_kind"] == "area"]
    r = _compute_from_data(occ, area)
    return {
        "energy_per_person_home": r["energy_per_person_home"],
        "sleep_home_spike_mult":  r["sleep_home_spike_mult"],
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path,
                   default=Path("data/serl_8963_targets/diurnal_targets_hourly_mean.csv"))
    p.add_argument("--year", type=int, default=2023)
    p.add_argument("--output", type=Path,
                   default=Path("results/calibration_serl_fits/presence_spikes.yaml"))
    p.add_argument("--bootstrap", type=int, default=0,
                   help="If >0, parametric-bootstrap N draws of the per-person "
                        "leaves. Diurnal SE is borrowed from annual-band "
                        "relative SE in --se-input; writes a 'bootstrap' block.")
    p.add_argument("--bootstrap-seed", type=int, default=0)
    p.add_argument("--se-input", type=Path,
                   default=Path("data/serl_8963_targets/daily_targets.csv"),
                   help="daily_targets.csv — source of annual-band relative SE "
                        "for the diurnal bootstrap (it carries se; the diurnal "
                        "mean file does not).")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: SERL input not found: {args.input}", file=sys.stderr)
        return 2

    result = fit(args.input, year=args.year)

    if args.bootstrap > 0:
        from bootstrap_bands import bootstrap_leaves, merge_point_and_bands
        occ_data = load_occupant_diurnal(args.input, year=args.year)
        area_data = load_area_diurnal(args.input, year=args.year)
        combined = _combined_table(occ_data, area_data, args.se_input, year=args.year)
        bands = bootstrap_leaves(
            combined, _leaf_values, value_col=_COMBINED_VALUE_COL,
            se=combined["se"].to_numpy(float),
            n_boot=args.bootstrap, seed=args.bootstrap_seed,
        )
        point = {
            "energy_per_person_home": result["energy_per_person_home"],
            "sleep_home_spike_mult":  result["sleep_home_spike_mult"],
        }
        result["bootstrap"] = merge_point_and_bands(point, bands)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        yaml.safe_dump(result, f, sort_keys=False)

    if not args.quiet:
        diag = result["diagnostics"]
        print(f"Unconfounded via cross-segmentation (sleep-hour per-m² = pure standby).")
        print()
        print(f"  naive per-occupant awake (confounded) = {diag['naive_per_person_awake_kWh_per_h']:.4f} kWh/h")
        print(f"  naive per-occupant sleep (confounded) = {diag['naive_per_person_sleep_kWh_per_h']:.4f} kWh/h")
        print(f"  per-m² standby (sleep)                = {diag['per_m2_standby_sleep_kWh_per_h_per_m2']:.5f} kWh/h/m²")
        print(f"  Δ area per added person (SERL panel)  = {diag['delta_area_per_added_person_m2']:.1f} m²/person")
        print()
        print(f"  energy_per_person_home  = {result['energy_per_person_home']:.4f} kWh/h/person (partialled, awake behavioural)")
        print(f"  awake_home_spike_mult   = {result['awake_home_spike_mult']:.3f}  (reference)")
        print(f"  sleep_home_spike_mult   = {result['sleep_home_spike_mult']:.4f}  (partialled, ≈ 0 by identification)")
        print(f"  energy_per_person_away  = {result['energy_per_person_away']:.4f} kWh/h/person  (literature default)")
        print()
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
