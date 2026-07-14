#!/usr/bin/env python3
"""
Path-A diagnostic: characterise WHERE the per-dwelling over-prediction lives
in the Newcastle 185-LSOA validation result.

Reads the notebook's exported cmp_with_reliability_*.csv and the Newcastle
stock, and answers:

  1. Is the over-prediction uniform, or driven by a tail?
  2. Does the ratio scale with LSOA size (small LSOAs over-shoot more)?
  3. Does the ratio scale with unknown_share even below the 0.15 gate?
  4. Is the ratio stable year-over-year per LSOA, or drifting?
  5. Is the ratio driven by dwelling-type mix (detached vs flat-heavy)?
  6. What's the shape correlation between ABM and DESNZ per fuel?
  7. What does the reliable-only subset look like split by these dimensions?

Writes a diagnostic report + figures alongside the input CSV.

Usage:
  python research/applied/scripts/diagnose_newcastle_validation.py \
      --cmp-csv notebooks/results/calibration/cmp_with_reliability_20260601_141943.csv \
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mape(ratios: pd.Series) -> float:
    r = pd.to_numeric(ratios, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(((r - 1.0).abs().mean()) * 100.0) if len(r) else float("nan")


def _bin_stats(df: pd.DataFrame, group_col: str, ratio_cols: list[str]) -> pd.DataFrame:
    """Bin a continuous diagnostic into 5 quintiles and tabulate ratios per bin."""
    df = df.copy()
    try:
        df["bin"] = pd.qcut(df[group_col], q=5, duplicates="drop")
    except ValueError:
        # too few unique values for 5 bins
        df["bin"] = df[group_col]
    rows = []
    for b, sub in df.groupby("bin", observed=True):
        row = {
            "bin":             str(b),
            "n":               int(len(sub)),
            "n_lsoas":         int(sub["lsoa_code"].nunique()),
        }
        for rc in ratio_cols:
            row[f"median_{rc}"] = float(sub[rc].median())
            row[f"mape_{rc}"]   = _mape(sub[rc])
        rows.append(row)
    return pd.DataFrame(rows)


def _shape_corr(df: pd.DataFrame) -> dict:
    """Pearson correlations of ABM vs DESNZ per-dwelling rates."""
    out = {}
    for label, a, b in [
        ("total", "abm_kwh_per_dw",      "desnz_kwh_per_dw"),
        ("elec",  "abm_elec_kwh_per_dw", "elec_kwh_per_dw"),
        ("gas",   "abm_gas_kwh_per_gas_dw", "gas_kwh_per_gas_dw"),
    ]:
        sub = df[[a, b]].dropna()
        if len(sub) >= 3:
            out[label] = float(sub.corr().iloc[0, 1])
        else:
            out[label] = float("nan")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def diagnose(cmp_csv: Path, city: str, out_dir: Path) -> dict:
    cmp = pd.read_csv(cmp_csv)
    print(f"Loaded: {cmp_csv.name}  rows={len(cmp)}  LSOAs={cmp['lsoa_code'].nunique()}  years={sorted(cmp['year'].unique())}")

    # ── Stock-derived dwelling-type mix per LSOA ──
    stock = load_city_stock(city)
    mft = stock["main_fuel_type"].astype(str).str.lower().str.strip()
    pt  = stock["property_type"].astype(str).str.lower().str.strip()
    stock["_flat_like"]    = pt.str.contains("flat", na=False)
    stock["_detached_like"]= pt.str.contains("detached", na=False) & ~pt.str.contains("semi", na=False)
    type_mix = (
        stock.groupby("lsoa_code", as_index=False)
        .agg(
            n_dwellings_stock = ("UPRN",          "count"),
            pct_flat          = ("_flat_like",    "mean"),
            pct_detached      = ("_detached_like","mean"),
            pct_gas_heated    = ("main_fuel_type", lambda s: (s == "mains gas").mean()),
        )
    )
    type_mix["pct_flat"]       *= 100.0
    type_mix["pct_detached"]   *= 100.0
    type_mix["pct_gas_heated"] *= 100.0

    cmp = cmp.merge(type_mix, on="lsoa_code", how="left")

    # ── 0. Top-line distribution of ratio_total ──
    print("\n── (0) Distribution of ratio_total ──")
    for q in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
        print(f"    p{int(q*100):2d} = {cmp['ratio_total'].quantile(q):.3f}")
    print(f"   mean = {cmp['ratio_total'].mean():.3f}   std = {cmp['ratio_total'].std():.3f}")

    # ── 1. Stratify by LSOA size ──
    print("\n── (1) Ratio vs LSOA size (run_dwellings quintiles) ──")
    sz = _bin_stats(cmp, "run_dwellings", ["ratio_total", "ratio_elec", "ratio_gas_assumed"])
    print(sz.round(3).to_string(index=False))

    # ── 2. Stratify by unknown_share (continuous, not just gate) ──
    print("\n── (2) Ratio vs unknown_share (continuous, including below gate) ──")
    us = _bin_stats(cmp, "unknown_share", ["ratio_total", "ratio_gas_strict", "ratio_gas_assumed"])
    print(us.round(3).to_string(index=False))

    # ── 3. Year-over-year ratio stability per LSOA ──
    print("\n── (3) Year-over-year ratio drift per LSOA ──")
    yoy = (
        cmp.pivot_table(index="lsoa_code", columns="year", values="ratio_total")
           .rename(columns=lambda c: f"y{int(c)}")
    )
    yoy_cols = [c for c in yoy.columns if c.startswith("y")]
    if len(yoy_cols) >= 2:
        ymin, ymax = yoy_cols[0], yoy_cols[-1]
        yoy["delta_yoy"] = yoy[ymax] - yoy[ymin]
        print(f"   median delta {ymin}->{ymax}: {yoy['delta_yoy'].median():.3f}")
        print(f"   share of LSOAs with delta > 0.15 (crisis-effect threshold): {(yoy['delta_yoy'] > 0.15).mean()*100:.1f}%")
        print(f"   share of LSOAs with delta < -0.05 (improving): {(yoy['delta_yoy'] < -0.05).mean()*100:.1f}%")

    # ── 4. Ratio vs dwelling-type mix ──
    print("\n── (4) Ratio vs dwelling-type mix ──")
    print("    by % flat (quintiles):")
    mf = _bin_stats(cmp, "pct_flat", ["ratio_total", "ratio_elec", "ratio_gas_assumed"])
    print(mf.round(3).to_string(index=False))
    print("    by % detached (quintiles):")
    md = _bin_stats(cmp, "pct_detached", ["ratio_total", "ratio_elec", "ratio_gas_assumed"])
    print(md.round(3).to_string(index=False))
    print("    by % mains-gas-heated (quintiles):")
    mg = _bin_stats(cmp, "pct_gas_heated", ["ratio_total", "ratio_elec", "ratio_gas_assumed"])
    print(mg.round(3).to_string(index=False))

    # ── 5. Shape correlations ──
    print("\n── (5) Pearson shape correlations (ABM vs DESNZ per-dwelling rates) ──")
    print("    All 185 LSOAs:")
    all_corr = _shape_corr(cmp)
    for k, v in all_corr.items():
        print(f"      {k}: {v:+.3f}")
    rel = cmp[cmp["is_reliable_lsoa"] == True]
    print(f"    Reliable subset ({rel['lsoa_code'].nunique()} LSOAs):")
    rel_corr = _shape_corr(rel)
    for k, v in rel_corr.items():
        print(f"      {k}: {v:+.3f}")

    # ── 6. Reliable-only subgroup MAPEs ──
    print("\n── (6) Reliable-only MAPE by subgroup ──")
    rel_by_size = _bin_stats(rel, "run_dwellings", ["ratio_total", "ratio_elec", "ratio_gas_assumed"])
    print("    by size (quintiles):")
    print(rel_by_size.round(3).to_string(index=False))

    # ── Figures ──
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.hist(cmp["ratio_total"], bins=50, alpha=0.7, color="#1f77b4")
    ax.axvline(1.0, color="black", lw=1, ls="--", label="parity")
    ax.axvline(cmp["ratio_total"].median(), color="red", lw=1, ls=":", label=f"median={cmp['ratio_total'].median():.2f}")
    ax.set_xlabel("ratio_total (abm/desnz per-dwelling)")
    ax.set_ylabel("LSOA-years")
    ax.set_title("(a) Distribution of per-dwelling total ratio (185 LSOAs × 3 yr)")
    ax.legend()

    ax = axes[0, 1]
    ax.scatter(cmp["run_dwellings"], cmp["ratio_total"], alpha=0.4, s=12)
    ax.axhline(1.0, color="black", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("run_dwellings (log)")
    ax.set_ylabel("ratio_total")
    ax.set_title("(b) Over-prediction vs LSOA size")

    ax = axes[1, 0]
    ax.scatter(cmp["unknown_share"], cmp["ratio_gas_strict"], alpha=0.4, s=12, color="#d62728")
    ax.axvline(0.15, color="black", lw=1, ls="--", label="gate=0.15")
    ax.axhline(1.0, color="black", lw=1, ls=":", label="parity")
    ax.set_xlabel("unknown_share")
    ax.set_ylabel("ratio_gas_strict")
    ax.set_yscale("log")
    ax.set_title("(c) Gas-strict pathology vs unknown_share")
    ax.legend()

    ax = axes[1, 1]
    ax.scatter(cmp["abm_kwh_per_dw"], cmp["desnz_kwh_per_dw"], alpha=0.4, s=12)
    lo = min(cmp["abm_kwh_per_dw"].min(), cmp["desnz_kwh_per_dw"].min())
    hi = max(cmp["abm_kwh_per_dw"].max(), cmp["desnz_kwh_per_dw"].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="parity")
    ax.set_xlabel("ABM kWh/dwelling")
    ax.set_ylabel("DESNZ kWh/dwelling (per elec-meter)")
    ax.set_title(f"(d) Per-dwelling alignment (Pearson r={all_corr['total']:+.2f})")
    ax.legend()

    fig.suptitle("Newcastle 185-LSOA validation — diagnostic breakouts", fontsize=13)
    figpath = out_dir / "diagnostic_breakouts.png"
    fig.savefig(figpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure: {figpath}")

    # ── Persist the joined diagnostic CSV ──
    diag_csv = out_dir / "cmp_with_diagnostic.csv"
    cmp.to_csv(diag_csv, index=False)
    print(f"CSV:    {diag_csv}")

    return {
        "shape_corr_all":      all_corr,
        "shape_corr_reliable": rel_corr,
        "median_ratio_total":  float(cmp["ratio_total"].median()),
        "median_ratio_gas_strict": float(cmp["ratio_gas_strict"].median()),
        "n_lsoas_total":       int(cmp["lsoa_code"].nunique()),
        "n_lsoas_reliable":    int(rel["lsoa_code"].nunique()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cmp-csv", type=Path, required=True,
                   help="Path to cmp_with_reliability_*.csv from the validation notebook")
    p.add_argument("--city", default="newcastle",
                   help="City name for the stock loader (default: newcastle)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory for figures + diagnostic CSV "
                        "(default: alongside the cmp-csv)")
    args = p.parse_args()

    out_dir = args.out_dir or args.cmp_csv.parent / "diagnostic"
    summary = diagnose(args.cmp_csv, args.city, out_dir)
    with open(out_dir / "diagnostic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote diagnostic summary to {out_dir / 'diagnostic_summary.json'}")


if __name__ == "__main__":
    main()
