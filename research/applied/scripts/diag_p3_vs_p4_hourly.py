#!/usr/bin/env python3
"""Diagnostic: run E01008344 against P3 and P4 configs, decompose the lift.

Two questions:
  1. Hourly shape — does the P4 lift sit in a few hours (presence peaks)
     or spread across all hours (per-hour baseline shift)?
  2. Per-component annual — base_kwh vs heat_kwh vs spike_kwh: which
     bucket actually swelled?

Throwaway script. Not part of the calibration pipeline.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
REPO = _THIS.parents[3]
sys.path.insert(0, str(REPO))

from household_energy.run_lsoa_batch import RunConfig, _run_single_lsoa  # noqa: E402

LSOA = "E01008344"

CFGS = {
    "P3":  REPO / "results/calibration_v5_phase3_postfix/calibrated_config.yaml",
    "P4":  REPO / "results/calibration_v5_phase4/calibrated_config.yaml",
    "P4b": REPO / "results/calibration_v5_phase4b/calibrated_config.yaml",
}


def run_one(label: str, cfg_path: Path) -> tuple[pd.Series, dict]:
    """Run E01008344 with one config; return (hourly totals series, agent decomp dict)."""
    outdir = REPO / "results_lsoa" / "_diag_p3p4"
    outdir.mkdir(parents=True, exist_ok=True)
    rc = RunConfig(
        geojson=REPO / "data/epc_abm_newcastle.geojson",
        climate=REPO / "data/ncc_2t_timeseries_2010_2026.parquet",
        hidp_csv=REPO / "data/hidp_uprn_matches_tiered.csv",
        start_utc="2023-01-01T00:00:00Z",
        end_utc="2024-01-01T00:00:00Z",
        days=None, local_tz="Europe/London", lsoa_col="lsoa_code",
        outdir=outdir, agent_collect_every=1,
        stamp=f"diag_{label}",
        config_path=cfg_path.resolve(),
        save_model_timeseries=True,
    )
    annual_row = _run_single_lsoa(LSOA, rc, 1, 1)
    # Reload the saved hourly model timeseries
    ts_path = outdir / LSOA / f"run_diag_{label}" / f"model_timeseries_{LSOA}_diag_{label}.parquet"
    ts = pd.read_parquet(ts_path)
    # Reload agent-level annual decomposition via the agents themselves — easier:
    # the model_dc only has model-level totals. We need agent-level base/heat/spike
    # annual sums. Approach: read the per-agent CSV the run writes alongside,
    # if any; else just diff totals.
    r = annual_row.iloc[0]
    decomp = {
        "abm_kwh":       float(r["abm_kwh"]),
        "abm_gas_kwh":   float(r["abm_gas_kwh"]),
        "abm_elec_kwh":  float(r["abm_elec_kwh"]),
        "abm_other_kwh": float(r["abm_other_kwh"]),
        "run_dwellings": int(r["run_dwellings"]),
        "run_gas_dwellings": int(r["run_gas_dwellings"]),
    }
    return ts, decomp


def main() -> int:
    series, decomps = {}, {}
    for label, p in CFGS.items():
        if not p.exists():
            print(f"!! missing config: {p}", file=sys.stderr)
            return 2
        print(f"--- running {label}: {p.name} ---")
        ts, d = run_one(label, p)
        series[label] = ts
        decomps[label] = d
        print(f"   total: {d['abm_kwh']:>12,.0f}  gas: {d['abm_gas_kwh']:>12,.0f}  elec: {d['abm_elec_kwh']:>12,.0f}")
        print()

    # --- Annual decomposition ---
    print("=" * 72)
    print("ANNUAL TOTALS (E01008344, 481 dwellings)")
    print("=" * 72)
    print(f"{'metric':<25} {'P3':>14} {'P4':>14} {'Δ':>12} {'%Δ':>8}")
    for k in ["abm_kwh", "abm_gas_kwh", "abm_elec_kwh", "abm_other_kwh"]:
        p3 = decomps["P3"][k]; p4 = decomps["P4"][k]
        dlt = p4 - p3
        pct = 100 * dlt / p3 if p3 else float("nan")
        print(f"{k:<25} {p3:>14,.0f} {p4:>14,.0f} {dlt:>+12,.0f} {pct:>+7.1f}%")
    print()

    # --- Per-dwelling per-year ---
    print(f"Per-dwelling per year:")
    nd = decomps["P3"]["run_dwellings"]
    ng = decomps["P3"]["run_gas_dwellings"]
    p3e = decomps["P3"]["abm_elec_kwh"] / nd
    p4e = decomps["P4"]["abm_elec_kwh"] / nd
    p3g = decomps["P3"]["abm_gas_kwh"] / ng
    p4g = decomps["P4"]["abm_gas_kwh"] / ng
    print(f"  electric/dw : P3 {p3e:>8,.0f}  P4 {p4e:>8,.0f}  Δ {p4e-p3e:>+6,.0f}  ({100*(p4e-p3e)/p3e:+.1f}%)")
    print(f"  gas/gas-dw  : P3 {p3g:>8,.0f}  P4 {p4g:>8,.0f}  Δ {p4g-p3g:>+6,.0f}  ({100*(p4g-p3g)/p3g:+.1f}%)")
    print()

    # --- Hourly shape: diff curve aggregated to mean-of-hour-of-day ---
    p3 = series["P3"].copy()
    p4 = series["P4"].copy()
    for s in (p3, p4):
        s.index = pd.RangeIndex(len(s))
        s["hod"] = s.index % 24
    p3_hod = p3.groupby("hod")[["total_electric_kwh", "total_gas_kwh"]].mean()
    p4_hod = p4.groupby("hod")[["total_electric_kwh", "total_gas_kwh"]].mean()

    print("=" * 72)
    print("AVG kWh PER HOUR-OF-DAY (E01008344, full year, 481 dwellings)")
    print("=" * 72)
    print(f"{'h':>3} {'P3_elec':>9} {'P4_elec':>9} {'Δ_elec':>9} {'P3_gas':>9} {'P4_gas':>9} {'Δ_gas':>9}")
    for h in range(24):
        e3 = p3_hod.loc[h, "total_electric_kwh"]; e4 = p4_hod.loc[h, "total_electric_kwh"]
        g3 = p3_hod.loc[h, "total_gas_kwh"];      g4 = p4_hod.loc[h, "total_gas_kwh"]
        print(f"{h:>3} {e3:>9.1f} {e4:>9.1f} {e4-e3:>+9.1f} {g3:>9.1f} {g4:>9.1f} {g4-g3:>+9.1f}")
    print()
    print("Totals:")
    print(f"  Σ Δ_elec / hour-of-day × 365 ≈ "
          f"{(p4_hod['total_electric_kwh'].sum() - p3_hod['total_electric_kwh'].sum()) * 365:,.0f} kWh/yr (whole LSOA)")
    print(f"  Σ Δ_gas  / hour-of-day × 365 ≈ "
          f"{(p4_hod['total_gas_kwh'].sum() - p3_hod['total_gas_kwh'].sum()) * 365:,.0f} kWh/yr (whole LSOA)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
