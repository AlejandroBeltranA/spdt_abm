#!/usr/bin/env python3
"""
Post-process a validate_run.py result through the framework already
established in `notebooks/energy-model-validation.ipynb`.

That notebook codified four decisions about how DESNZ comparison should
be done — none of which my initial `compute_validation_metrics` honoured:

  1. Compare **per-dwelling rates**, not total-kWh ratios. DESNZ
     per-dwelling uses **electricity meters** as the dwelling proxy.
  2. Gas denominator: **strict** (excluding unknown gas-connection) is
     the primary KPI; **assumed** (including unknowns) is diagnostic.
  3. Per-LSOA **unknown_share = run_unknown_gas_connection_dwellings /
     run_dwellings** — LSOAs with `unknown_share > 0.15` are flagged
     "high-uncertainty" for gas comparison and excluded from the
     reliable subset.
  4. Headline KPIs are reported on the reliable subset; pass thresholds
     are total MAPE 18%, strict gas MAPE 20%. Full-set numbers are a
     sensitivity check.

This script reproduces those KPIs from a single (city, year) validate_run
output — no model rerun needed. Compatible with `results/{sanity,
transfer, temporal}/<city>_<year>/`.

Usage:
  python research/applied/scripts/analyze_validation.py \
      --city newcastle --year 2023 \
      --kind sanity                    # or transfer / temporal
      [--unknown-share-gate 0.15]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS.parents[3]))

from utils import (  # noqa: E402
    CITY_CONVENTIONS, cache_dir, load_city_stock, load_desnz,
)


# ─────────────────────────────────────────────────────────────────────────────
# Per-LSOA dwelling counters from the stock
# ─────────────────────────────────────────────────────────────────────────────

def stock_dwelling_counts(stock: pd.DataFrame) -> pd.DataFrame:
    """Per-LSOA dwelling counts matching the notebook's columns.

    Definitions (verified against EPC `is_gas` / `is_off_gas` columns):
      run_gas_dwellings_strict           — is_gas == True (definitive)
      run_off_gas_dwellings              — is_off_gas == True (definitive)
      run_unknown_gas_connection_dwellings — is_gas is null (unknown)
      run_gas_dwellings   (assumed)      — strict + unknown
      run_dwellings                      — total
    """
    s = stock.copy()
    s["_is_gas_t"]      = (s["is_gas"]     == 1) | (s["is_gas"]     == True)
    s["_is_offgas_t"]   = (s["is_off_gas"] == 1) | (s["is_off_gas"] == True)
    s["_is_gas_unk"]    = s["is_gas"].isna()
    grouped = (
        s.groupby("lsoa_code", as_index=False)
        .agg(
            run_dwellings=("UPRN", "count"),
            run_gas_dwellings_strict=("_is_gas_t",   "sum"),
            run_off_gas_dwellings   =("_is_offgas_t","sum"),
            run_unknown_gas_connection_dwellings=("_is_gas_unk", "sum"),
        )
    )
    grouped["run_gas_dwellings"] = (
        grouped["run_gas_dwellings_strict"]
        + grouped["run_unknown_gas_connection_dwellings"]
    )
    return grouped


# ─────────────────────────────────────────────────────────────────────────────
# DESNZ per-dwelling rates
# ─────────────────────────────────────────────────────────────────────────────

def desnz_per_dwelling_rates(desnz: pd.DataFrame) -> pd.DataFrame:
    """Match the notebook's DESNZ per-fuel + combined per-meter metrics."""
    d = desnz.copy()
    for c in ["total_kwh_elec", "total_kwh_gas", "meters_elec", "meters_gas"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["elec_kwh_per_dw"]     = d["total_kwh_elec"] / d["meters_elec"].replace({0: np.nan})
    d["gas_kwh_per_gas_dw"]  = d["total_kwh_gas"]  / d["meters_gas"] .replace({0: np.nan})
    d["desnz_kwh"]           = d["total_kwh_elec"].fillna(0.0) + d["total_kwh_gas"].fillna(0.0)
    d["desnz_kwh_per_dw"]    = d["desnz_kwh"] / d["meters_elec"].replace({0: np.nan})
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Combine + ratios + KPIs
# ─────────────────────────────────────────────────────────────────────────────

def _mape_from_ratios(ratios: pd.Series) -> float:
    r = pd.to_numeric(ratios, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(((r - 1.0).abs().mean()) * 100.0) if len(r) else float("nan")


def build_cmp(residuals: pd.DataFrame, stock_counts: pd.DataFrame,
              desnz_rates: pd.DataFrame, *, unknown_share_gate: float = 0.15) -> pd.DataFrame:
    """Notebook cell 18 reproduced as a function — produce one row per LSOA
    with the strict/assumed gas split and the uncertainty flag."""

    # The validate_run residuals.parquet has abm_elec_kwh and abm_gas_kwh
    # already aggregated to LSOA.  Pull only what we need.
    abm = residuals[["lsoa_code", "abm_elec_kwh", "abm_gas_kwh", "abm_total_kwh"]].copy()

    cmp = (
        abm.merge(stock_counts, on="lsoa_code", how="left")
           .merge(desnz_rates,  on="lsoa_code", how="inner")
    )

    # ABM per-dwelling rates
    cmp["abm_kwh_per_dw"]      = cmp["abm_total_kwh"] / cmp["run_dwellings"].replace({0: np.nan})
    cmp["abm_elec_kwh_per_dw"] = cmp["abm_elec_kwh"]  / cmp["run_dwellings"].replace({0: np.nan})

    # Gas: two denominator choices
    cmp["abm_gas_kwh_per_gas_dw_assumed"] = (
        cmp["abm_gas_kwh"] / cmp["run_gas_dwellings"].replace({0: np.nan})
    )
    cmp["abm_gas_kwh_per_gas_dw_strict"]  = (
        cmp["abm_gas_kwh"] / cmp["run_gas_dwellings_strict"].replace({0: np.nan})
    )

    # Ratios (notebook convention: abm / desnz, MAPE = mean(|ratio - 1|) * 100)
    cmp["ratio_total"]        = cmp["abm_kwh_per_dw"]      / cmp["desnz_kwh_per_dw"]
    cmp["ratio_elec"]         = cmp["abm_elec_kwh_per_dw"] / cmp["elec_kwh_per_dw"]
    cmp["ratio_gas_assumed"]  = cmp["abm_gas_kwh_per_gas_dw_assumed"] / cmp["gas_kwh_per_gas_dw"]
    cmp["ratio_gas_strict"]   = cmp["abm_gas_kwh_per_gas_dw_strict"]  / cmp["gas_kwh_per_gas_dw"]

    # Uncertainty flag (notebook §18)
    cmp["unknown_share"] = (
        cmp["run_unknown_gas_connection_dwellings"] / cmp["run_dwellings"].replace({0: np.nan})
    )
    cmp["gas_denominator_quality"] = np.where(
        cmp["unknown_share"] <= unknown_share_gate, "low-uncertainty", "high-uncertainty",
    )
    return cmp


def kpi_scorecard(cmp: pd.DataFrame) -> dict:
    """Strict-first KPI table — matches notebook cell 28."""
    full = cmp
    reliable = cmp[cmp["gas_denominator_quality"] == "low-uncertainty"]

    def _scope(sub: pd.DataFrame, label: str) -> dict:
        return {
            "scope":                       label,
            "n_lsoas":                     int(sub["lsoa_code"].nunique()),
            "primary_mape_total_pct":      _mape_from_ratios(sub["ratio_total"]),
            "primary_mape_elec_pct":       _mape_from_ratios(sub["ratio_elec"]),
            "primary_mape_gas_strict_pct": _mape_from_ratios(sub["ratio_gas_strict"]),
            "sensitivity_mape_gas_assumed_pct": _mape_from_ratios(sub["ratio_gas_assumed"]),
            "median_ratio_total":          float(sub["ratio_total"].median()),
            "median_ratio_elec":           float(sub["ratio_elec"].median()),
            "median_ratio_gas_strict":     float(sub["ratio_gas_strict"].median()),
            "median_ratio_gas_assumed":    float(sub["ratio_gas_assumed"].median()),
        }

    return {
        "all_lsoas": _scope(full,      "all_lsoas"),
        "reliable":  _scope(reliable,  "reliable (low gas-connection uncertainty)"),
    }


def gate_status(scorecard: dict) -> dict:
    """Apply notebook cell 32's release-gate thresholds."""
    rel = scorecard["reliable"]
    total = rel["primary_mape_total_pct"]
    gas_strict = rel["primary_mape_gas_strict_pct"]
    TOTAL_PASS, TOTAL_WARN = 18.0, 22.0
    GAS_PASS, GAS_WARN    = 20.0, 24.0
    def _verdict(v: float, p: float, w: float) -> str:
        if v <= p: return "PASS"
        if v <= w: return "WARN"
        return "FAIL"
    return {
        "total_mape_reliable":   total,
        "gas_strict_reliable":   gas_strict,
        "total_verdict":   _verdict(total, TOTAL_PASS, TOTAL_WARN),
        "gas_verdict":     _verdict(gas_strict, GAS_PASS, GAS_WARN),
        "thresholds": {"total_pass": TOTAL_PASS, "total_warn": TOTAL_WARN,
                       "gas_pass": GAS_PASS,   "gas_warn":   GAS_WARN},
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--city", required=True, choices=sorted(CITY_CONVENTIONS.keys()))
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--kind", default="sanity",
                   choices=["sanity", "transfer", "temporal"],
                   help="Which result subdir to read (default: sanity)")
    p.add_argument("--unknown-share-gate", type=float, default=0.15,
                   help="LSOAs with unknown_share > this are flagged high-uncertainty (default 0.15)")
    args = p.parse_args()

    in_dir = cache_dir(args.kind, f"{args.city}_{args.year}")
    residuals_path = in_dir / "residuals.parquet"
    if not residuals_path.exists():
        raise FileNotFoundError(f"No residuals.parquet at {residuals_path} — run validate_run.py first.")

    print(f"Reading: {residuals_path}")
    residuals = pd.read_parquet(residuals_path)

    print(f"Loading stock ({args.city})…")
    stock = load_city_stock(args.city)
    print(f"Loading DESNZ ({args.city}, {args.year})…")
    desnz = load_desnz(args.city, args.year)

    stock_counts = stock_dwelling_counts(stock)
    desnz_rates  = desnz_per_dwelling_rates(desnz)
    cmp = build_cmp(residuals, stock_counts, desnz_rates,
                     unknown_share_gate=args.unknown_share_gate)
    scorecard = kpi_scorecard(cmp)
    gate = gate_status(scorecard)

    # Persist
    cmp_path = in_dir / "cmp_per_lsoa.parquet"
    cmp.to_parquet(cmp_path, index=False)

    score_path = in_dir / "kpi_scorecard.json"
    with open(score_path, "w") as f:
        json.dump({"scorecard": scorecard, "gate": gate,
                   "unknown_share_gate": args.unknown_share_gate,
                   "city": args.city, "year": args.year, "kind": args.kind},
                  f, indent=2, default=float)

    # Pretty-print
    print(f"\n── KPI scorecard ({args.city} {args.year}, {args.kind}) ──")
    for scope_label, scope in scorecard.items():
        print(f"\n  Scope: {scope['scope']}  (n_lsoas = {scope['n_lsoas']})")
        print(f"    primary MAPE total       : {scope['primary_mape_total_pct']:6.2f}%")
        print(f"    primary MAPE elec        : {scope['primary_mape_elec_pct']:6.2f}%")
        print(f"    primary MAPE gas (strict): {scope['primary_mape_gas_strict_pct']:6.2f}%")
        print(f"    sensitivity gas (assumed): {scope['sensitivity_mape_gas_assumed_pct']:6.2f}%")
        print(f"    median ratio total       : {scope['median_ratio_total']:6.3f}")
        print(f"    median ratio elec        : {scope['median_ratio_elec']:6.3f}")
        print(f"    median ratio gas strict  : {scope['median_ratio_gas_strict']:6.3f}")

    print(f"\n── Release gate (reliable subset) ──")
    print(f"    total  : {gate['total_mape_reliable']:6.2f}%   [{gate['total_verdict']}]"
          f"   (pass ≤ {gate['thresholds']['total_pass']}%, warn ≤ {gate['thresholds']['total_warn']}%)")
    print(f"    gas/str: {gate['gas_strict_reliable']:6.2f}%   [{gate['gas_verdict']}]"
          f"   (pass ≤ {gate['thresholds']['gas_pass']}%, warn ≤ {gate['thresholds']['gas_warn']}%)")

    print(f"\nWrote: {cmp_path}")
    print(f"       {score_path}")


if __name__ == "__main__":
    main()
