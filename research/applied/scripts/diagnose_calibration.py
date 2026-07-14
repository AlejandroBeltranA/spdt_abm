#!/usr/bin/env python3
"""
Path-A2 diagnostic: decompose the per-dwelling over-prediction by source.

Reframed claim: the methodology paper predicts per-house demand. DESNZ is
supporting evidence at LSOA scale. Citywide ABM total matches DESNZ within
+2%, so the per-house calibration AGGREGATES correctly. But:

  * gas median ratio (1.49) >> elec median ratio (1.16) — heating side
    over-predicts more than baseload
  * gas shape correlation (0.61) << elec shape correlation (0.72) — model
    captures less of the per-LSOA gas variation than elec

This diagnostic answers:

  (1) Is the gas/elec imbalance consistent across LSOAs — i.e., a calibration
      bias — or driven by a subset?
  (2) Is the gas baseline anchor (0.289 kWh/h) consistent with what DESNZ
      implies for low-heating-load LSOAs?
  (3) What stock-side variables (floor area, SAP, age, dwelling-type mix)
      predict per-LSOA gas demand in DESNZ but NOT in ABM? That's the
      shape gap.

Reads the latest cmp_with_reliability_*.csv + the Newcastle stock; produces
a diagnostic report + figures.

Usage:
  python research/applied/scripts/diagnose_calibration.py \
      --cmp-csv notebooks/results/calibration/cmp_with_reliability_20260601_144919.csv \
      --city newcastle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS.parents[3]))

from utils import load_city_stock  # noqa: E402

# Anchor used in the notebook calibration (kWh/h baseline gas per dwelling).
# Convert to annual per-dwelling baseline.
ANCHOR_GAS_KWH_PER_HOUR  = 0.289   # from calibrated_config.yaml
ANCHOR_ELEC_KWH_PER_HOUR = 0.291
BASELINE_GAS_KWH_PER_YEAR  = ANCHOR_GAS_KWH_PER_HOUR  * 24 * 365  # ~2532
BASELINE_ELEC_KWH_PER_YEAR = ANCHOR_ELEC_KWH_PER_HOUR * 24 * 365  # ~2549


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.replace({0: np.nan})).replace([np.inf, -np.inf], np.nan)


def _age_to_decade_mid(age_str: pd.Series) -> pd.Series:
    """Map an 'YYYY-YYYY' age band to its midpoint year, NaN if unparseable."""
    out = pd.Series(np.nan, index=age_str.index)
    s = age_str.astype(str).str.strip()
    m = s.str.extract(r"(\d{4})-(\d{4})")
    a = pd.to_numeric(m[0], errors="coerce")
    b = pd.to_numeric(m[1], errors="coerce")
    out = (a + b) / 2.0
    # Handle "pre-1900" / "post-2003" style fallbacks
    pre = s.str.contains(r"pre[-\s]?1900", case=False, na=False, regex=True)
    post = s.str.contains(r"(?:post|after)[-\s]?\d{4}", case=False, na=False, regex=True)
    out = out.where(~pre, 1880)
    out = out.where(~post, 2010)
    return out


def diagnose(cmp_csv: Path, city: str, out_dir: Path) -> dict:
    cmp = pd.read_csv(cmp_csv)
    print(f"Loaded: {cmp_csv.name}  rows={len(cmp)}  LSOAs={cmp['lsoa_code'].nunique()}")

    stock = load_city_stock(city)

    pt = stock["property_type"].astype(str).str.lower().str.strip()
    stock["_flat_like"]     = pt.str.contains("flat", na=False)
    stock["_detached_like"] = pt.str.contains("detached", na=False) & ~pt.str.contains("semi", na=False)
    stock["_semi_like"]     = pt.str.contains("semi", na=False)
    stock["_terraced_like"] = pt.str.contains("terraced", na=False)
    stock["_decade_mid"]    = _age_to_decade_mid(stock["property_age"])

    stock_chars = stock.groupby("lsoa_code", as_index=False).agg(
        mean_floor_area_m2 = ("floor_area_m2", "mean"),
        median_floor_area  = ("floor_area_m2", "median"),
        mean_sap_band_ord  = ("sap_band_ord",  "mean"),
        mean_sap_rating    = ("sap_rating",    "mean"),
        mean_age_year      = ("_decade_mid",   "mean"),
        pct_detached       = ("_detached_like","mean"),
        pct_semi           = ("_semi_like",    "mean"),
        pct_terraced       = ("_terraced_like","mean"),
        pct_flat           = ("_flat_like",    "mean"),
    )
    for c in ["pct_detached", "pct_semi", "pct_terraced", "pct_flat"]:
        stock_chars[c] *= 100.0

    df = cmp.merge(stock_chars, on="lsoa_code", how="left")

    # ──────────────────────────────────────────────────────────────────────
    # (1) Gas-vs-elec balance: is heating over-predicted more than baseload?
    # ──────────────────────────────────────────────────────────────────────
    df["abm_gas_elec_ratio"]   = _safe_div(df["abm_gas_kwh"], df["abm_elec_kwh"])
    df["desnz_gas_elec_ratio"] = _safe_div(df["gas_kwh"],     df["elec_kwh"])
    df["gas_elec_misfit"]      = _safe_div(df["abm_gas_elec_ratio"], df["desnz_gas_elec_ratio"])

    abm_ge  = df["abm_gas_elec_ratio"].median()
    desnz_ge = df["desnz_gas_elec_ratio"].median()
    misfit_median = df["gas_elec_misfit"].median()

    print("\n── (1) Gas/elec balance ──")
    print(f"    ABM   median gas/elec ratio: {abm_ge:.3f}")
    print(f"    DESNZ median gas/elec ratio: {desnz_ge:.3f}")
    print(f"    Misfit (ABM/DESNZ ratio):    {misfit_median:.3f}")
    print(f"    → ABM over-predicts gas relative to elec by ~{(misfit_median - 1) * 100:+.0f}%")

    # Stratify gas/elec misfit by year — if uniform, it's calibration; if
    # 2022-spike, it's the energy-crisis behavioural response.
    print("\n    Misfit by year:")
    for yr, sub in df.groupby("year"):
        print(f"      {int(yr)}: median misfit = {sub['gas_elec_misfit'].median():.3f}")

    # ──────────────────────────────────────────────────────────────────────
    # (2) Baseline-vs-heating decomposition
    # ──────────────────────────────────────────────────────────────────────
    # ABM gas decomposes into baseline (anchor × n_gas_dwellings) + heating.
    df["abm_gas_baseline_kwh"] = BASELINE_GAS_KWH_PER_YEAR * df["run_gas_dwellings_strict"]
    df["abm_gas_heating_kwh"]  = (df["abm_gas_kwh"] - df["abm_gas_baseline_kwh"]).clip(lower=0)
    df["abm_baseline_share"]   = _safe_div(df["abm_gas_baseline_kwh"], df["abm_gas_kwh"])

    # If baseline > DESNZ_gas, the anchor is too high outright.
    df["anchor_vs_desnz"] = _safe_div(df["abm_gas_baseline_kwh"], df["gas_kwh"])

    print("\n── (2) Baseline-vs-heating decomposition ──")
    print(f"    Median ABM baseline share of total gas:    {df['abm_baseline_share'].median():.2%}")
    print(f"    Median ABM baseline / DESNZ gas:           {df['anchor_vs_desnz'].median():.3f}")
    print(f"      → if > 1.0, the baseline anchor alone exceeds DESNZ gas; anchor too high")
    print(f"    Share of LSOA-years where anchor > DESNZ:  {(df['anchor_vs_desnz'] > 1).mean() * 100:.1f}%")

    # The implied DESNZ heating fraction (sanity check, not a calibration test)
    # If we trust ANCHOR_GAS as the per-dwelling summer floor, DESNZ heating = total - baseline×n
    df["desnz_heating_kwh"] = (df["gas_kwh"] - BASELINE_GAS_KWH_PER_YEAR * df["gas_meters"]).clip(lower=0)
    df["desnz_baseline_share"] = _safe_div(
        BASELINE_GAS_KWH_PER_YEAR * df["gas_meters"], df["gas_kwh"]
    )
    print(f"    Implied DESNZ baseline share (using ABM anchor): {df['desnz_baseline_share'].median():.2%}")
    print(f"    → if MUCH higher than ABM's baseline share, the anchor is plausible")
    print(f"    → if similar to or lower than ABM's share, the anchor is too high")

    # ──────────────────────────────────────────────────────────────────────
    # (3) Heating slope per LSOA — implied from inter-year variation
    # ──────────────────────────────────────────────────────────────────────
    # Years 2021-2023 give us 3 demand points per LSOA. If we had per-year HDD
    # we could derive per-LSOA slopes. We don't have that lookup in this CSV,
    # so we approximate: heavier ABM-DESNZ slope deviation across years should
    # correlate with the gas slope being mis-calibrated.
    yoy_gas = (
        df.pivot_table(index="lsoa_code", columns="year",
                       values=["abm_gas_kwh", "gas_kwh"])
    )
    abm_gas_yoy_swing = (yoy_gas[("abm_gas_kwh", 2022)] - yoy_gas[("abm_gas_kwh", 2021)]).abs()
    desnz_gas_yoy_swing = (yoy_gas[("gas_kwh", 2022)] - yoy_gas[("gas_kwh", 2021)]).abs()
    yoy_swing_ratio = (abm_gas_yoy_swing / desnz_gas_yoy_swing.replace({0: np.nan})).median()
    print("\n── (3) Inter-year gas swing comparison ──")
    print(f"    Median |ΔABM 2022-2021| / |ΔDESNZ 2022-2021|: {yoy_swing_ratio:.3f}")
    print(f"    → > 1: ABM swings more across years than DESNZ (model is more climate-sensitive than data)")
    print(f"    → < 1: ABM under-responds to year-to-year climate variation")

    # ──────────────────────────────────────────────────────────────────────
    # (4) Per-LSOA shape regression: what predicts the residual?
    # ──────────────────────────────────────────────────────────────────────
    yr23 = df[df["year"] == 2023].copy()
    yr23["gas_residual_per_dw"] = (
        (yr23["abm_gas_kwh_per_gas_dw"] - yr23["gas_kwh_per_gas_dw"])
    )

    # Predictors of DESNZ gas per gas-meter (what reality looks like)
    feature_cols = ["mean_floor_area_m2", "mean_sap_band_ord", "mean_age_year",
                    "pct_detached", "pct_semi", "pct_terraced", "pct_flat"]
    avail = [c for c in feature_cols if yr23[c].notna().sum() > 30]

    print("\n── (4) What predicts per-LSOA gas demand? ──")
    print("    Correlation with DESNZ gas/gas_meter (per-LSOA, 2023):")
    for c in avail:
        r = yr23[c].corr(yr23["gas_kwh_per_gas_dw"])
        print(f"      {c:25s}: r = {r:+.3f}")

    print("\n    Correlation with ABM gas/gas_dwelling (per-LSOA, 2023):")
    for c in avail:
        r = yr23[c].corr(yr23["abm_gas_kwh_per_gas_dw"])
        print(f"      {c:25s}: r = {r:+.3f}")

    print("\n    Correlation with RESIDUAL (ABM - DESNZ) gas/dwelling (per-LSOA, 2023):")
    print("    Variables that correlate here are predictors DESNZ has but the model is missing:")
    for c in avail:
        r = yr23[c].corr(yr23["gas_residual_per_dw"])
        print(f"      {c:25s}: r = {r:+.3f}")

    # ──────────────────────────────────────────────────────────────────────
    # (5) Inferred recommendations
    # ──────────────────────────────────────────────────────────────────────
    print("\n── (5) Inferred recommendations ──")
    recs = []
    if misfit_median > 1.15:
        recs.append(f"GAS HEATING SLOPE: ABM gas/elec ratio is {misfit_median:.2f}× DESNZ ratio → "
                     f"heating slope is over-calibrated by ~{(misfit_median-1)*100:.0f}% relative to elec.")
    if df["anchor_vs_desnz"].median() < 0.5:
        recs.append("GAS ANCHOR: baseline anchor << DESNZ gas → anchor not the issue, heating is.")
    elif df["anchor_vs_desnz"].median() > 0.9:
        recs.append("GAS ANCHOR: baseline anchor close to/over DESNZ gas → anchor may be too high.")
    if yoy_swing_ratio > 1.3:
        recs.append(f"CLIMATE SENSITIVITY: ABM YoY swing is {yoy_swing_ratio:.2f}× DESNZ → "
                     f"model is too climate-responsive (heat_slope_max=5.0 likely too permissive).")
    elif yoy_swing_ratio < 0.7:
        recs.append("CLIMATE SENSITIVITY: ABM under-responds to YoY climate variation.")
    # Variables strongly correlated with residual but weakly with ABM = missing predictors
    for c in avail:
        r_desnz = yr23[c].corr(yr23["gas_kwh_per_gas_dw"])
        r_abm   = yr23[c].corr(yr23["abm_gas_kwh_per_gas_dw"])
        if abs(r_desnz) > 0.3 and abs(r_abm) < 0.2 and abs(r_desnz - r_abm) > 0.15:
            recs.append(f"MISSING PREDICTOR: DESNZ gas correlates with {c} (r={r_desnz:+.2f}) "
                         f"but ABM does not (r={r_abm:+.2f}). Model is missing this stock signal.")

    if not recs:
        recs.append("No single dominant calibration issue identified; over-prediction is broad.")
    for r in recs:
        print(f"    • {r}")

    # ──────────────────────────────────────────────────────────────────────
    # Figures
    # ──────────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.scatter(df["desnz_gas_elec_ratio"], df["abm_gas_elec_ratio"], alpha=0.3, s=12)
    lo = min(df["desnz_gas_elec_ratio"].min(), df["abm_gas_elec_ratio"].min())
    hi = max(df["desnz_gas_elec_ratio"].quantile(0.99), df["abm_gas_elec_ratio"].quantile(0.99))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="parity")
    ax.set_xlabel("DESNZ gas/elec ratio")
    ax.set_ylabel("ABM gas/elec ratio")
    ax.set_title(f"(a) Heat-vs-baseload imbalance — median misfit {misfit_median:.2f}")
    ax.legend()

    ax = axes[0, 1]
    pos = df["abm_baseline_share"].dropna()
    ax.hist(pos, bins=40, alpha=0.7, color="#2ca02c")
    ax.axvline(pos.median(), color="red", lw=1, ls=":",
                label=f"median={pos.median():.2%}")
    ax.set_xlabel("ABM baseline share of total gas")
    ax.set_ylabel("LSOA-years")
    ax.set_title("(b) Baseline anchor's share of model gas demand")
    ax.legend()

    ax = axes[1, 0]
    if "mean_floor_area_m2" in yr23.columns and yr23["mean_floor_area_m2"].notna().any():
        sub = yr23.dropna(subset=["mean_floor_area_m2", "gas_kwh_per_gas_dw"])
        ax.scatter(sub["mean_floor_area_m2"], sub["gas_kwh_per_gas_dw"],
                    alpha=0.5, s=14, label="DESNZ", color="#ff7f0e")
        ax.scatter(sub["mean_floor_area_m2"], sub["abm_gas_kwh_per_gas_dw"],
                    alpha=0.5, s=14, label="ABM", color="#1f77b4")
        ax.set_xlabel("LSOA mean floor area (m²)")
        ax.set_ylabel("Gas kWh per gas-connected dwelling")
        ax.set_title("(c) Does floor area predict gas demand?")
        ax.legend()

    ax = axes[1, 1]
    sub = df.dropna(subset=["pct_detached", "ratio_gas_assumed"])
    ax.scatter(sub["pct_detached"], sub["ratio_gas_assumed"], alpha=0.4, s=14)
    ax.axhline(1.0, color="black", lw=1, ls="--")
    ax.set_xlabel("LSOA % detached dwellings")
    ax.set_ylabel("ratio_gas_assumed (ABM/DESNZ)")
    ax.set_title("(d) Gas over-prediction vs detached share")

    fig.suptitle("Newcastle calibration diagnostic — where does the gas mis-fit come from?",
                 fontsize=12)
    figpath = out_dir / "calibration_diagnostic.png"
    fig.savefig(figpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure: {figpath}")

    summary = {
        "abm_gas_elec_ratio_median":      float(abm_ge),
        "desnz_gas_elec_ratio_median":    float(desnz_ge),
        "gas_elec_misfit_median":         float(misfit_median),
        "abm_baseline_share_median":      float(df["abm_baseline_share"].median()),
        "anchor_vs_desnz_median":         float(df["anchor_vs_desnz"].median()),
        "implied_desnz_baseline_share":   float(df["desnz_baseline_share"].median()),
        "yoy_swing_ratio_median":         float(yoy_swing_ratio),
        "recommendations":                recs,
    }
    with open(out_dir / "calibration_diagnostic_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary: {out_dir / 'calibration_diagnostic_summary.json'}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cmp-csv", type=Path, required=True)
    p.add_argument("--city", default="newcastle")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    out_dir = args.out_dir or args.cmp_csv.parent / "calibration_diagnostic"
    diagnose(args.cmp_csv, args.city, out_dir)


if __name__ == "__main__":
    main()
