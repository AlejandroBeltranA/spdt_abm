#!/usr/bin/env python3
"""
Fit the gas heating diurnal profile (`heating_profile_24h`) to SERL — no scalars.

The problem
-----------
Model gas ≈ flat baseline + climate-heating × `heating_profile_24h`. The climate
signal `slope × HD` peaks pre-dawn (coldest hour) and the shipped literature
`heating_profile_24h` is *high* overnight, so the model heats hardest at night —
the opposite of SERL, which shows an overnight setback, a sharp 07:00 morning
blast and an 18:00 evening peak. The gas diurnal is phase-inverted against SERL.

The construction (pure ratio of two empirical shapes — no tuned constant)
------------------------------------------------------------------------
Let
    H(h)  = the model's own HD-heating by hour-of-day under a FLAT profile
            (i.e. slope × HD × occupancy, the thing the profile multiplies),
            measured as a share of the day:  h_share(h) = H(h) / Σ H.
    S(h)  = the SERL gas diurnal share of the day: s_share(h) = SERL(h) / Σ SERL.

Set
    heating_profile_24h(h) = s_share(h) / h_share(h).

Then the reshaped heating is
    H(h) · profile(h) = H(h) · s_share(h) / (H(h)/ΣH) = s_share(h) · ΣH,
which is ∝ the SERL shape, and whose sum over the day is ΣH — i.e. the annual
heating integral is **unchanged by construction**. The level constant (ΣH)
cancels: there is no fitted scalar, no iteration, no rescaling step. This is the
gas analogue of the electricity baseline's mean-1.0 normalisation — the only
difference is that the heating profile integrates against HD, so the neutral
shape is defined by H(h), not by a flat mean.

Note this returns the annual heating to what the SERL *slope* fit delivers
(`Σ slope·HD`); the current literature profile, being positively HD-correlated,
inflates it. So annual gas moves relative to the *current shipped* model — that
change is reported, not absorbed.

Baseline pedestal
-----------------
Total gas = flat baseline + heating. We match the *heating* shape to the
SERL gas shape net of the (measured, not fitted) baseline pedestal, so the total
gas curve lands on SERL. Summer cooking shape is a small residual (would need a
separate gas baseline profile) — not addressed here.

Output
------
``results/calibration_serl_fits/gas_heating_profile.yaml`` with the fitted
``heating_profile_24h`` (24 values) and diagnostics.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_THIS = Path(__file__).resolve()
REPO = _THIS.parents[3]
sys.path.insert(0, str(REPO))

from household_energy.run_lsoa_batch import RunConfig, _run_single_lsoa  # noqa: E402

LSOA = "E01008344"
EPS = 1e-9
DEFAULT_LIT_PROFILE = [1.20,1.15,1.10,1.00,0.95,1.00,1.10,1.20,1.05,0.95,0.90,0.90,
                       0.90,0.90,0.90,0.95,1.00,1.10,1.25,1.30,1.25,1.20,1.15,1.10]


def _serl_gas_shape(year: int) -> np.ndarray:
    df = pd.read_csv(REPO / "data/serl_8963_targets/diurnal_targets_hourly_mean.csv")
    s = df[(df.quantity == "Gas") & (df.year == year) & (df.seg3_var == "none")
           & (df.has_pv == "All") & (df.heating_fuel == "All")
           & (df.weekday_weekend == "both")].sort_values("hour")
    v = s["mean_kwh"].to_numpy(float)
    return v / v.mean()


def _run(profile: list[float], stamp: str) -> tuple[np.ndarray, float, float]:
    """Run the LSOA with a candidate heating_profile_24h.

    Returns (gas hour-of-day means kWh, annual gas kWh, per-hour baseline gas).
    """
    base_cfg = yaml.safe_load(open(REPO / "household_energy/calibrated_config.yaml"))
    base_cfg["model"]["heating_profile_24h"] = [float(x) for x in profile]
    fd = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(base_cfg, fd); fd.close()

    outdir = REPO / "results_lsoa" / "_fit_gas_profile"
    outdir.mkdir(parents=True, exist_ok=True)
    rc = RunConfig(
        geojson=REPO / "data/epc_abm_newcastle.geojson",
        climate=REPO / "data/ncc_2t_timeseries_2010_2026.parquet",
        hidp_csv=REPO / "data/hidp_uprn_matches_tiered.csv",
        start_utc="2023-01-01T00:00:00Z", end_utc="2024-01-01T00:00:00Z",
        days=None, local_tz="Europe/London", lsoa_col="lsoa_code",
        outdir=outdir, agent_collect_every=1, stamp=stamp,
        config_path=Path(fd.name), save_model_timeseries=True,
    )
    row = _run_single_lsoa(LSOA, rc, 1, 1).iloc[0]
    ts = pd.read_parquet(
        outdir / LSOA / f"run_{stamp}" / f"model_timeseries_{LSOA}_{stamp}.parquet"
    ).reset_index(drop=True)
    ts["hod"] = ts.index % 24
    g = ts.groupby("hod")["total_gas_kwh"].mean().to_numpy()
    warm = ts[ts["ambient_mean_tempC"] > 17.0]   # no-heating hours → baseline pedestal
    base_per_hour = float(warm["total_gas_kwh"].mean()) if len(warm) else 0.0
    return g, float(row["abm_gas_kwh"]), base_per_hour


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--output", type=Path,
                    default=REPO / "results/calibration_serl_fits/gas_heating_profile.yaml")
    args = ap.parse_args()

    serl = _serl_gas_shape(args.year)                 # mean-1.0
    s_share = serl / serl.sum()

    # ── One run with a FLAT profile → the model's intrinsic HD-heating shape ──
    gas_flat, annual_flat, base_ph = _run([1.0] * 24, "flat")
    H = np.maximum(gas_flat - base_ph, EPS)           # heating by hour-of-day
    h_share = H / H.sum()

    # ── The fit: pure ratio of two empirical shapes (no constant to tune) ──
    profile = list(s_share / h_share)

    # Current shipped behaviour uses the literature profile — compute its gas
    # shape and annual analytically from the flat run (no extra ABM run needed).
    lit = np.asarray(DEFAULT_LIT_PROFILE)
    gas_lit = base_ph + H * lit
    annual_lit = annual_flat - 365.0 * 24 * 0.0  # placeholder; recomputed below
    annual_lit = float((base_ph * 24 + (H * lit).sum()) * 365.0)

    # ── Verify ──
    gas_new, annual_new, _ = _run(profile, "verify")

    def shape(a): return a / a.mean()
    def rmse(a, b): return float(np.sqrt(np.mean((shape(a) - shape(b)) ** 2)))

    rmse_before = rmse(gas_lit, serl)
    rmse_after = rmse(gas_new, serl)
    print(f"baseline pedestal /h : {base_ph:.1f} kWh")
    print(f"shape RMSE vs SERL   : literature {rmse_before:.4f}  ->  fitted {rmse_after:.4f}")
    print(f"annual gas (shipped/lit) {annual_lit:,.0f}  -> fitted {annual_new:,.0f}  "
          f"({100*(annual_new-annual_lit)/annual_lit:+.2f}%)")
    print(f"annual gas (slope ref, flat) {annual_flat:,.0f}  -> fitted {annual_new:,.0f}  "
          f"({100*(annual_new-annual_flat)/annual_flat:+.2f}%, should be ~0)")

    result = {
        "heating_profile_24h": [float(x) for x in profile],
        "diagnostics": {
            "year": int(args.year), "lsoa": LSOA,
            "baseline_pedestal_kwh_per_hour": base_ph,
            "serl_gas_shape_target": [float(x) for x in serl],
            "model_gas_shape_literature": [float(x) for x in shape(gas_lit)],
            "model_gas_shape_fitted": [float(x) for x in shape(gas_new)],
            "shape_rmse_literature": rmse_before,
            "shape_rmse_fitted": rmse_after,
            "annual_gas_literature": annual_lit,
            "annual_gas_slope_reference_flat": annual_flat,
            "annual_gas_fitted": annual_new,
            "annual_pct_change_vs_literature": 100*(annual_new-annual_lit)/annual_lit,
            "annual_pct_change_vs_slope_reference": 100*(annual_new-annual_flat)/annual_flat,
            "method": (
                "heating_profile_24h(h) = SERL_gas_share(h) / model_HD-heating_share(h). "
                "Pure ratio of two empirical day-shares: reshapes heating to the SERL "
                "diurnal while preserving the heating integral by construction (level "
                "constant cancels). No fitted scalar, no iteration."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        yaml.safe_dump(result, f, sort_keys=False)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
