"""Fit mean-1.0 MONTHLY profiles from SERL — the seasonal analogue of the diurnal
(hourly) profile. Two profiles, both mean-preserving so annual totals are unchanged:

  heating_month_profile_12   behavioural heating shape (SERL monthly heating per
                             degree-day) — replaces the hard heating-months on/off
                             gate with a smooth seasonal ramp.
  base_profile_12_electric   baseline-electricity seasonal shape (winter uplift)
                             from gas-heated homes' electricity (≈ pure baseline).

Usage:
  .venv/bin/python research/applied/scripts/fit_monthly_profile.py
Writes results/calibration_serl_fits/monthly_profile.yaml
"""
from __future__ import annotations
import numpy as np, pandas as pd, yaml
from pathlib import Path

DAYS = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
SUMMER = [6,7,8]
OUT = Path("results/calibration_serl_fits/monthly_profile.yaml")


def serl_monthly(d, hf, qty, col="mean"):
    s = d[(d.period_type=="monthly")&(d.year==2023)&(d.weekday_weekend=="both")&
          (d.heating_fuel==hf)&(d.has_pv.isin(["No","All"]))&(d.seg3_var=="none")&(d.quantity==qty)].copy()
    s[col] = pd.to_numeric(s[col], errors="coerce")
    return s.groupby("month")[col].mean()


def fit(write: bool = True):
    """Fit both mean-1.0 monthly profiles and return them. Set ``write=False`` to
    compute without overwriting the committed ``monthly_profile.yaml`` (e.g. when a
    notebook recomputes the profiles just to plot them)."""
    d = pd.read_csv("data/serl_8963_targets/daily_targets.csv")
    months = list(range(1, 13))
    days = np.array([DAYS[m] for m in months], float)

    # --- baseline-electricity seasonal profile (gas-heated electricity ≈ pure baseline) ---
    g_elec_day = serl_monthly(d, "Gas", "Electricity imports").reindex(months).values  # kWh/day
    base_prof = g_elec_day / np.average(g_elec_day, weights=days)                        # mean-1.0 (day-weighted)

    # --- heating behavioural profile: SERL monthly heating per degree-day ---
    g_gas_day = serl_monthly(d, "Gas", "Gas").reindex(months).values                    # kWh/day
    hdd_day   = serl_monthly(d, "Gas", "Gas", col="mean_hdd").reindex(months).values    # HDD/day (national)
    gas_floor = g_gas_day[[m-1 for m in SUMMER]].mean()                                 # summer baseline gas
    heat_day  = np.clip(g_gas_day - gas_floor, 0, None)                                 # heating kWh/day
    # effective heating intensity per degree-day, by month; normalise (HDD-weighted) to mean 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        intensity = np.where(hdd_day > 0.3, heat_day / hdd_day, np.nan)
    # fill summer (negligible heating) with the season's low value, then normalise
    intensity = pd.Series(intensity).interpolate(limit_direction="both").values
    heat_prof = intensity / np.average(intensity, weights=hdd_day)                       # mean-1.0 (HDD-weighted)

    out = {
        "heating_month_profile_12": [float(x) for x in heat_prof],
        "base_profile_12_electric": [float(x) for x in base_prof],
        "diagnostics": {
            "year": 2023, "source": "data/serl_8963_targets/daily_targets.csv",
            "gas_summer_floor_kwh_day": float(gas_floor),
            "heating_weighted_mean_check": float(np.average(heat_prof, weights=hdd_day)),
            "base_weighted_mean_check": float(np.average(base_prof, weights=days)),
        },
    }
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(yaml.safe_dump(out, sort_keys=False))
    return out, heat_prof, base_prof, hdd_day, heat_day, g_elec_day


if __name__ == "__main__":
    out, hp, bp, hdd, heat, gel = fit()
    M = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    print(f"{'month':6s}{'heat_profile':>13s}{'base_profile':>13s}")
    for i, m in enumerate(M):
        print(f"{m:6s}{hp[i]:13.3f}{bp[i]:13.3f}")
    print(f"\nheating profile HDD-weighted mean = {out['diagnostics']['heating_weighted_mean_check']:.3f} (→1.0)")
    print(f"baseline profile day-weighted mean = {out['diagnostics']['base_weighted_mean_check']:.3f} (→1.0)")
    print(f"wrote {OUT}")
