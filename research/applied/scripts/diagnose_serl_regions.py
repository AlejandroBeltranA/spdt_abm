#!/usr/bin/env python3
"""
Quick diagnostic: per-region annual demand from SERL.

Tests the hypothesis I asserted but never verified — that Newcastle (North East)
consumes meaningfully less than the SERL national mean, which would explain
why the model (calibrated to SERL national) over-predicts Newcastle DESNZ.

If North East is close to the national mean, the hypothesis is wrong and
the model's over-prediction has a different source (denominator mismatch,
SERL respondent self-selection bias, etc.).

Approach:
  For each of SERL's 11 regions, compute mean annual per-dwelling demand
  for gas-heated homes (their gas + elec) and electric-heated homes (their
  elec), pooled across calibration years 2020-2023. Plot side by side and
  compute North East / national-mean ratios.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
SERL = REPO / "data" / "serl_8963_targets" / "daily_targets.csv"

REGIONS = [
    "North East", "North West", "Yorkshire", "East Midlands", "West Midlands",
    "East of England", "Greater London", "South East", "South West",
    "Wales", "Scotland",
]


def _annual_mean_kwh(daily_targets: pd.DataFrame, *, region: str,
                     heating_fuel: str, fuel_quantity: str,
                     years: list[int]) -> tuple[float, int]:
    """Annual per-dwelling kWh for one (region × heating_fuel × fuel) cell."""
    sub = daily_targets[
        (daily_targets["period_type"].astype(str) == "monthly")
        & (daily_targets["year"].astype(int).isin(years))
        & (daily_targets["weekday_weekend"].astype(str) == "both")
        & (daily_targets["heating_fuel"].astype(str) == heating_fuel)
        & (daily_targets["has_pv"].astype(str).isin(["No", "All"]))
        & (daily_targets["seg3_var"].astype(str) == "region")
        & (daily_targets["seg3_value"].astype(str) == region)
        & (daily_targets["quantity"].astype(str) == fuel_quantity)
    ].copy()
    if sub.empty:
        return float("nan"), 0
    sub["n_rounded"] = pd.to_numeric(sub["n_rounded"], errors="coerce").fillna(0.0)
    sub["mean"]      = pd.to_numeric(sub["mean"],      errors="coerce")
    # n-weighted monthly mean → sum the 12 monthly means × days-in-month
    sub["wx"]    = sub["mean"] * sub["n_rounded"]
    monthly = sub.groupby("month", as_index=False).agg(
        wx=("wx", "sum"), n=("n_rounded", "sum"))
    monthly["kwh_day"] = monthly["wx"] / monthly["n"].replace({0: np.nan})
    days_in_month = {1:31,2:28.25,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
    monthly["days"] = monthly["month"].map(days_in_month)
    monthly["kwh_month"] = monthly["kwh_day"] * monthly["days"]
    if monthly["kwh_month"].isna().any():
        return float("nan"), int(monthly["n"].sum())
    return float(monthly["kwh_month"].sum()), int(monthly["n"].sum())


def main() -> None:
    daily = pd.read_csv(SERL)
    years = [2020, 2021, 2022, 2023]
    print(f"Loaded SERL daily_targets: {len(daily):,} rows; years {years}")

    rows = []
    for region in REGIONS:
        gas_gas,  n_gas_h_gas = _annual_mean_kwh(daily, region=region,
            heating_fuel="Gas",      fuel_quantity="Gas", years=years)
        gas_elec, n_gas_h_e   = _annual_mean_kwh(daily, region=region,
            heating_fuel="Gas",      fuel_quantity="Electricity imports", years=years)
        elec,     n_elec_h    = _annual_mean_kwh(daily, region=region,
            heating_fuel="Electric", fuel_quantity="Electricity imports", years=years)
        rows.append({
            "region":              region,
            "gas_heated_gas_kwh":  gas_gas,
            "gas_heated_elec_kwh": gas_elec,
            "gas_heated_total_kwh": gas_gas + gas_elec if np.isfinite(gas_gas) and np.isfinite(gas_elec) else float("nan"),
            "elec_heated_elec_kwh": elec,
            "n_gas_heated":        max(n_gas_h_gas, n_gas_h_e),
            "n_elec_heated":       n_elec_h,
        })
    df = pd.DataFrame(rows).set_index("region")

    # National-mean (n-weighted across regions)
    def _wmean(col, n_col):
        ok = df[col].notna() & (df[n_col] > 0)
        return float(np.average(df[col][ok], weights=df[n_col][ok]))
    nat_gas_h_total = _wmean("gas_heated_total_kwh", "n_gas_heated")
    nat_gas_h_gas   = _wmean("gas_heated_gas_kwh",   "n_gas_heated")
    nat_gas_h_elec  = _wmean("gas_heated_elec_kwh",  "n_gas_heated")
    nat_elec_h      = _wmean("elec_heated_elec_kwh", "n_elec_heated")

    df["ratio_total_vs_national"] = df["gas_heated_total_kwh"] / nat_gas_h_total
    df["ratio_gas_vs_national"]   = df["gas_heated_gas_kwh"]   / nat_gas_h_gas
    df["ratio_elec_vs_national"]  = df["gas_heated_elec_kwh"]  / nat_gas_h_elec

    print("\n══ Annual per-dwelling demand by region (SERL 2020-2023 pooled) ══")
    print(f"\n  Gas-heated dwellings (gas + electric demand):")
    print(f"  {'Region':20s}  {'n':>7s}  {'Gas kWh':>10s}  {'Elec kWh':>10s}  {'Total kWh':>10s}  {'NE/Region':>9s}")
    for r in REGIONS:
        row = df.loc[r]
        print(f"  {r:20s}  {row['n_gas_heated']:>7,}  "
              f"{row['gas_heated_gas_kwh']:>10,.0f}  "
              f"{row['gas_heated_elec_kwh']:>10,.0f}  "
              f"{row['gas_heated_total_kwh']:>10,.0f}  "
              f"{row['ratio_total_vs_national']:>9.3f}")
    print(f"  {'NATIONAL n-weighted':20s}  {'':>7s}  "
          f"{nat_gas_h_gas:>10,.0f}  {nat_gas_h_elec:>10,.0f}  {nat_gas_h_total:>10,.0f}  1.000")

    print(f"\n  Electric-heated dwellings (electric only):")
    print(f"  {'Region':20s}  {'n':>7s}  {'Elec kWh':>10s}  {'vs national':>10s}")
    for r in REGIONS:
        row = df.loc[r]
        ratio = row["elec_heated_elec_kwh"] / nat_elec_h if np.isfinite(row["elec_heated_elec_kwh"]) else float("nan")
        print(f"  {r:20s}  {row['n_elec_heated']:>7,}  "
              f"{row['elec_heated_elec_kwh']:>10,.0f}  {ratio:>10.3f}")
    print(f"  {'NATIONAL n-weighted':20s}  {'':>7s}  {nat_elec_h:>10,.0f}  1.000")

    # Newcastle is North East
    ne = df.loc["North East"]
    print("\n══ North East (Newcastle) vs national ══")
    print(f"  Gas-heated total demand:  {ne['gas_heated_total_kwh']:,.0f} kWh/yr (national {nat_gas_h_total:,.0f}) → ratio {ne['ratio_total_vs_national']:.3f}")
    print(f"  Gas-heated gas only:      {ne['gas_heated_gas_kwh']:,.0f} kWh/yr (national {nat_gas_h_gas:,.0f}) → ratio {ne['ratio_gas_vs_national']:.3f}")
    print(f"  Gas-heated elec only:     {ne['gas_heated_elec_kwh']:,.0f} kWh/yr (national {nat_gas_h_elec:,.0f}) → ratio {ne['ratio_elec_vs_national']:.3f}")
    print(f"  Elec-heated dwelling:     {ne['elec_heated_elec_kwh']:,.0f} kWh/yr (national {nat_elec_h:,.0f}) → ratio {ne['elec_heated_elec_kwh']/nat_elec_h:.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    df_sorted = df.sort_values("gas_heated_total_kwh", ascending=True)

    ax = axes[0]
    colors = ['#d62728' if r == 'North East' else '#1f77b4' for r in df_sorted.index]
    y = np.arange(len(df_sorted))
    ax.barh(y, df_sorted["gas_heated_gas_kwh"], color=colors, alpha=0.85,
             label='Gas')
    ax.barh(y, df_sorted["gas_heated_elec_kwh"], left=df_sorted["gas_heated_gas_kwh"],
             color=colors, alpha=0.45, label='Electric (in gas-heated homes)')
    ax.set_yticks(y); ax.set_yticklabels(df_sorted.index)
    ax.axvline(nat_gas_h_total, color="black", lw=1, ls="--",
                label=f"national mean ({nat_gas_h_total:,.0f})")
    ax.set_xlabel("Annual kWh / dwelling")
    ax.set_title("Gas-heated dwellings: per-dwelling annual demand by region\n"
                  "(SERL 2020–2023, red = North East / Newcastle)")
    ax.legend(loc="lower right")

    ax = axes[1]
    df_e = df.sort_values("elec_heated_elec_kwh", ascending=True)
    colors_e = ['#d62728' if r == 'North East' else '#2ca02c' for r in df_e.index]
    y = np.arange(len(df_e))
    ax.barh(y, df_e["elec_heated_elec_kwh"], color=colors_e, alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(df_e.index)
    ax.axvline(nat_elec_h, color="black", lw=1, ls="--",
                label=f"national mean ({nat_elec_h:,.0f})")
    ax.set_xlabel("Annual kWh / dwelling")
    ax.set_title("Electric-heated dwellings: per-dwelling annual demand by region")
    ax.legend(loc="lower right")

    out_dir = REPO / "research" / "applied" / "results" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "serl_regional_demand.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure: {fig_path}")

    csv_path = out_dir / "serl_regional_demand.csv"
    df.to_csv(csv_path)
    print(f"CSV:    {csv_path}")


if __name__ == "__main__":
    main()
