#!/usr/bin/env python3
"""
Naive regression baseline — the comparator the ABM has to beat.

Fits a simple OLS on the training city's LSOA-aggregated stock features:

    kWh_per_dwelling ~ a + b·mean_floor_area + c·mean_epc_band_ord + d·annual_HDD

then applies it to each test city/year and reports MAPE against DESNZ.
Trains separate models for electricity and gas (matching DESNZ's fuel split).

If the corresponding ABM validation result exists in
`results/transfer/<city>_<year>/metrics.json`, the script reports ABM MAPE
side by side so reviewers can see what fraction of the prediction skill
is attributable to the ABM beyond simple stock-feature regression.

This is the Alderete Peralta 2022 (AE) pattern translated to UK demand:
ABM must beat the naive baseline on out-of-sample MAPE.

Usage:
  python research/applied/scripts/regression_baseline.py \
      --train-city newcastle \
      --train-year 2023 \
      --test-cities sunderland waltham_forest \
      --test-year 2023
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS.parents[3]))

from utils import (  # noqa: E402
    cache_dir, climate_path, load_city_stock, load_desnz,
    CITY_CONVENTIONS, results_root,
)
from progress import ProgressTracker  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

HDD_BASE_TEMP_C = 15.5  # CIBSE convention


def _lsoa_features_from_stock(stock: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-dwelling stock attributes to LSOA-mean features.

    Returns DataFrame with columns:
      lsoa_code, n_dwellings, mean_floor_area, mean_epc_band_ord
    """
    # Tolerate either floor_area_m2 or floor_area
    floor_col = next((c for c in ["floor_area_m2", "floor_area", "total_floor_area_m2"]
                       if c in stock.columns), None)
    epc_col = next((c for c in ["sap_band_ord", "epc_band_ord", "current_energy_rating_ord"]
                     if c in stock.columns), None)
    if floor_col is None or epc_col is None:
        raise KeyError(
            f"Stock missing floor area / EPC band column. "
            f"floor_col={floor_col}, epc_col={epc_col}. "
            f"Available: {list(stock.columns)[:30]} …"
        )
    s = stock[["lsoa_code", floor_col, epc_col]].copy()
    s[floor_col] = pd.to_numeric(s[floor_col], errors="coerce")
    s[epc_col]   = pd.to_numeric(s[epc_col],   errors="coerce")
    out = (
        s.groupby("lsoa_code", as_index=False)
        .agg(n_dwellings=("lsoa_code", "size"),
             mean_floor_area=(floor_col, "mean"),
             mean_epc_band_ord=(epc_col, "mean"))
    )
    return out


def _city_annual_hdd(city: str, year: int) -> float:
    """Compute citywide annual HDD (single scalar) from the climate parquet.

    Uses the city-mean of the per-point temperature timeseries (i.e. average
    across all climate points belonging to that city), then sums HDD = max(0,
    15.5 - T) across all hours of the target year.
    """
    path = climate_path(city)
    df = pd.read_parquet(path)

    if "time" in df.columns:
        df = df.set_index("time")
    elif "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)

    df_yr = df[df.index.year == int(year)]
    # All remaining columns are temperature points
    temp_cols = [c for c in df_yr.columns if c != "time"]
    if not temp_cols:
        raise ValueError(f"No temperature columns found in {path}")
    citywide_mean = df_yr[temp_cols].mean(axis=1)
    hdd = (HDD_BASE_TEMP_C - citywide_mean).clip(lower=0.0).sum()
    return float(hdd)


def _lsoa_kwh_per_dwelling(desnz: pd.DataFrame) -> pd.DataFrame:
    """Convert DESNZ totals to per-dwelling rates."""
    out = desnz.copy()
    out["kwh_per_dw_elec"] = out["total_kwh_elec"] / out["meters_elec"].replace({0: np.nan})
    out["kwh_per_dw_gas"]  = out["total_kwh_gas"]  / out["meters_gas"] .replace({0: np.nan})
    out["kwh_per_dw_total"] = out["kwh_per_dw_elec"].fillna(0) + out["kwh_per_dw_gas"].fillna(0)
    return out[["lsoa_code", "kwh_per_dw_elec", "kwh_per_dw_gas", "kwh_per_dw_total"]]


def _build_design_matrix(features: pd.DataFrame, hdd_value: float) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, valid_mask) for OLS. X columns: 1, floor, epc_band, HDD."""
    n = len(features)
    floor = features["mean_floor_area"].values.astype(float)
    epc   = features["mean_epc_band_ord"].values.astype(float)
    hdd   = np.full(n, float(hdd_value))
    X = np.column_stack([np.ones(n), floor, epc, hdd])
    valid = np.isfinite(floor) & np.isfinite(epc)
    return X, valid


# ─────────────────────────────────────────────────────────────────────────────
# Fit + predict
# ─────────────────────────────────────────────────────────────────────────────

def _ols_fit(X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Standard OLS via lstsq. Returns (coefs, r_squared)."""
    Xv = X[mask]
    yv = y[mask]
    coefs, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    pred = Xv @ coefs
    ss_res = float(np.sum((yv - pred) ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coefs, r2


def _mape(pred: np.ndarray, truth: np.ndarray) -> float:
    ok = np.isfinite(pred) & np.isfinite(truth) & (truth > 0)
    return float(np.mean(np.abs((pred[ok] - truth[ok]) / truth[ok])) * 100) if ok.any() else float("nan")


def _read_abm_mape(city: str, year: int) -> Optional[dict]:
    """Try to find a matching ABM metrics.json from the transfer pipeline."""
    candidates = [
        results_root() / "transfer" / f"{city}_{year}" / "metrics.json",
        results_root() / "temporal" / f"{city}_{year}" / "metrics.json",
        results_root() / "sanity"   / f"{city}_{year}" / "metrics.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                m = json.load(f)
            return {
                "abm_mape_elec":  m.get("mape_elec"),
                "abm_mape_gas":   m.get("mape_gas"),
                "abm_mape_total": m.get("mape_total"),
                "source": str(p),
            }
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-city", required=True,
                   choices=sorted(CITY_CONVENTIONS.keys()))
    p.add_argument("--train-year", type=int, required=True)
    p.add_argument("--test-cities", nargs="+", required=True,
                   choices=sorted(CITY_CONVENTIONS.keys()))
    p.add_argument("--test-year", type=int, required=True)
    args = p.parse_args()

    out_dir = cache_dir("baseline", f"{args.train_city}_{args.train_year}")
    tracker = ProgressTracker(out_dir,
                                job_name=f"regression_baseline_{args.train_city}_{args.train_year}")
    tracker.start(f"train={args.train_city}/{args.train_year} "
                  f"test={args.test_cities}/{args.test_year}")

    # ── Train ────────────────────────────────────────────────────────────────
    with tracker.section("load train city stock + DESNZ"):
        train_stock = load_city_stock(args.train_city)
        train_desnz = load_desnz(args.train_city, args.train_year)
        train_hdd = _city_annual_hdd(args.train_city, args.train_year)
    tracker.milestone("train data",
                       n_dwellings=len(train_stock),
                       n_lsoas_desnz=len(train_desnz),
                       train_hdd=f"{train_hdd:.0f}")

    with tracker.section("aggregate to LSOA + fit OLS"):
        feats = _lsoa_features_from_stock(train_stock)
        truth = _lsoa_kwh_per_dwelling(train_desnz)
        merged = feats.merge(truth, on="lsoa_code", how="inner")
        X, mask = _build_design_matrix(merged, train_hdd)
        coefs_elec,  r2_elec  = _ols_fit(X, merged["kwh_per_dw_elec"].values,  mask)
        coefs_gas,   r2_gas   = _ols_fit(X, merged["kwh_per_dw_gas"].values,   mask)
        coefs_total, r2_total = _ols_fit(X, merged["kwh_per_dw_total"].values, mask)
    tracker.milestone("train fit",
                       r2_elec=f"{r2_elec:.3f}",
                       r2_gas=f"{r2_gas:.3f}",
                       r2_total=f"{r2_total:.3f}",
                       n_train_lsoas=int(mask.sum()))

    coef_names = ["intercept", "mean_floor_area", "mean_epc_band_ord", "annual_HDD"]
    coefs_payload = {
        "train_city": args.train_city,
        "train_year": args.train_year,
        "train_hdd":  train_hdd,
        "elec":  {"r_squared": r2_elec,  **dict(zip(coef_names, coefs_elec.tolist()))},
        "gas":   {"r_squared": r2_gas,   **dict(zip(coef_names, coefs_gas.tolist()))},
        "total": {"r_squared": r2_total, **dict(zip(coef_names, coefs_total.tolist()))},
    }
    with open(out_dir / "regression_coefs.json", "w") as f:
        json.dump(coefs_payload, f, indent=2, default=float)

    # ── Test ─────────────────────────────────────────────────────────────────
    test_rows = []
    for test_city in args.test_cities:
        with tracker.section(f"predict {test_city} {args.test_year}"):
            test_stock = load_city_stock(test_city)
            test_desnz = load_desnz(test_city, args.test_year)
            test_hdd = _city_annual_hdd(test_city, args.test_year)
            test_feats = _lsoa_features_from_stock(test_stock)
            test_truth = _lsoa_kwh_per_dwelling(test_desnz)
            test_merged = test_feats.merge(test_truth, on="lsoa_code", how="inner")
            Xt, mt = _build_design_matrix(test_merged, test_hdd)

            pred_elec  = Xt @ coefs_elec
            pred_gas   = Xt @ coefs_gas
            pred_total = Xt @ coefs_total

            mape_elec  = _mape(pred_elec[mt],  test_merged["kwh_per_dw_elec"].values[mt])
            mape_gas   = _mape(pred_gas[mt],   test_merged["kwh_per_dw_gas"].values[mt])
            mape_total = _mape(pred_total[mt], test_merged["kwh_per_dw_total"].values[mt])

            abm = _read_abm_mape(test_city, args.test_year)

        row = {
            "test_city":  test_city,
            "test_year":  args.test_year,
            "test_hdd":   test_hdd,
            "n_lsoas":    int(mt.sum()),
            "reg_mape_elec":  mape_elec,
            "reg_mape_gas":   mape_gas,
            "reg_mape_total": mape_total,
        }
        if abm is not None:
            row.update(abm)
            row["abm_beats_reg_total"] = (
                (abm["abm_mape_total"] < mape_total) if abm["abm_mape_total"] is not None else None
            )
        test_rows.append(row)

        tracker.milestone(f"{test_city} {args.test_year}",
                           reg_mape_total=f"{mape_total:.2f}%",
                           abm_mape_total=(f"{abm['abm_mape_total']:.2f}%"
                                            if abm and abm.get("abm_mape_total") is not None
                                            else "n/a"))

    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(out_dir / "test_mape.csv", index=False)

    tracker.finish(
        f"wrote regression_coefs.json + test_mape.csv ({len(test_df)} test cases) to {out_dir}"
    )


if __name__ == "__main__":
    main()
