#!/usr/bin/env python3
"""
Rebuild a `notebooks/results_lsoa/abm_year_all_<stamp>.parquet` rollup
from the per-LSOA `model_timeseries_*.parquet` files, with the full
22-column schema the validation notebook expects.

Why this exists:
The batch runner in `household_energy/run_lsoa_batch.py` was reduced
to a 5-column rollup schema in the March 2026 calibration revert.
This script reconstructs the full schema (per-fuel kWh + dwelling-counter
breakdown + per-dwelling rates) from the hourly model timeseries that
were saved alongside each per-LSOA run, so we don't need to re-execute
the 50+ model runs.

Usage:
  python research/applied/scripts/rebuild_rollup.py \
      --run-outdir notebooks/results_lsoa \
      --stamp 20260522 \
      --geojson data/epc_abm_newcastle.geojson \
      --start-utc 2021-01-01T00:00:00Z \
      --end-utc   2025-01-01T00:00:00Z
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS.parents[3]))

from progress import ProgressTracker  # noqa: E402


def _aggregate_one_lsoa(
    timeseries_path: Path,
    lsoa_code: str,
    gdf_subset: gpd.GeoDataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    """Convert one LSOA's hourly model timeseries → annual per-fuel rollup."""
    mdl = pd.read_parquet(timeseries_path)
    # The index is hour_start_utc as written by _run_single_lsoa
    if "hour_start_utc" in mdl.columns:
        mdl = mdl.set_index("hour_start_utc")
    mdl.index = pd.to_datetime(mdl.index, utc=True)
    mdl = mdl.loc[(mdl.index >= start_ts) & (mdl.index < end_ts)]

    fuel_cols = {
        "abm_kwh":       "total_energy",
        "abm_elec_kwh":  "total_electric_kwh",
        "abm_gas_kwh":   "total_gas_kwh",
        "abm_other_kwh": "total_other_kwh",
    }
    parts = []
    for out_name, src_col in fuel_cols.items():
        s = mdl[src_col].resample("YS").sum().rename(out_name) if src_col in mdl.columns \
            else pd.Series(0.0, index=mdl.index).resample("YS").sum().rename(out_name)
        parts.append(s)
    annual = pd.concat(parts, axis=1)
    annual["year"]      = annual.index.year
    annual["lsoa_code"] = lsoa_code

    # ── Dwelling counters from the stock subset for this LSOA ──
    n_total = len(gdf_subset)

    def _b(col: str) -> pd.Series:
        if col not in gdf_subset.columns:
            return pd.Series(False, index=gdf_subset.index)
        return pd.to_numeric(gdf_subset[col], errors="coerce") == 1

    is_gas_t      = _b("is_gas")
    is_offgas_t   = _b("is_off_gas")
    is_gas_unk    = gdf_subset["is_gas"].isna() if "is_gas" in gdf_subset.columns \
                    else pd.Series(True, index=gdf_subset.index)

    mft = (gdf_subset.get("main_fuel_type", pd.Series([""] * n_total))
                     .astype(str).str.lower().str.strip())

    run_gas_dwellings_strict             = int(is_gas_t.sum())
    run_off_gas_dwellings                = int(is_offgas_t.sum())
    run_unknown_gas_connection_dwellings = int(is_gas_unk.sum())
    run_gas_dwellings                    = run_gas_dwellings_strict + run_unknown_gas_connection_dwellings

    run_gas_heated_dwellings      = int((mft == "mains gas").sum())
    run_electric_heated_dwellings = int((mft == "electricity").sum())
    _non_heated = {"mains gas", "electricity", "no fuel", "nan", "unknown", "none", ""}
    run_other_heated_dwellings    = int((~mft.isin(_non_heated)).sum())
    run_electric_dwellings = n_total
    run_other_dwellings = 0

    annual["run_dwellings"]                          = n_total
    annual["run_gas_dwellings"]                      = run_gas_dwellings
    annual["run_gas_dwellings_strict"]               = run_gas_dwellings_strict
    annual["run_off_gas_dwellings"]                  = run_off_gas_dwellings
    annual["run_unknown_gas_connection_dwellings"]   = run_unknown_gas_connection_dwellings
    annual["run_gas_heated_dwellings"]               = run_gas_heated_dwellings
    annual["run_electric_heated_dwellings"]          = run_electric_heated_dwellings
    annual["run_other_heated_dwellings"]             = run_other_heated_dwellings
    annual["run_electric_dwellings"]                 = run_electric_dwellings
    annual["run_other_dwellings"]                    = run_other_dwellings

    annual["abm_kwh_per_dw"]              = annual["abm_kwh"]       / n_total
    annual["abm_elec_kwh_per_dw"]         = annual["abm_elec_kwh"]  / n_total
    annual["abm_gas_kwh_per_dw"]          = annual["abm_gas_kwh"]   / n_total
    annual["abm_gas_kwh_per_gas_dw"]      = annual["abm_gas_kwh"]   / (run_gas_dwellings or float("nan"))
    annual["abm_gas_kwh_per_gas_heated_dw"] = annual["abm_gas_kwh"] / (run_gas_heated_dwellings or float("nan"))
    annual["abm_other_kwh_per_dw"]        = annual["abm_other_kwh"] / n_total

    return annual.reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-outdir", type=Path,
                   default=Path("notebooks/results_lsoa"),
                   help="Base outdir containing <lsoa>/run_<stamp>/ subdirs")
    p.add_argument("--stamp", required=True,
                   help="Run stamp (e.g. 20260522). Determines which subdir to read.")
    p.add_argument("--geojson", type=Path, required=True,
                   help="Stock GeoJSON used by the batch (for dwelling counters)")
    p.add_argument("--lsoa-col", default="lsoa_code")
    p.add_argument("--start-utc", required=True)
    p.add_argument("--end-utc",   required=True)
    p.add_argument("--output", type=Path, default=None,
                   help="Output rollup path (default: <run-outdir>/abm_year_all_<stamp>.parquet)")
    args = p.parse_args()

    start_ts = pd.Timestamp(args.start_utc, tz="UTC")
    end_ts   = pd.Timestamp(args.end_utc,   tz="UTC")
    out_path = args.output or (args.run_outdir / f"abm_year_all_{args.stamp}.parquet")

    tracker = ProgressTracker(args.run_outdir,
                                job_name=f"rebuild_rollup_{args.stamp}")
    tracker.start(f"stamp={args.stamp} window=[{args.start_utc}, {args.end_utc}) "
                  f"geojson={args.geojson}")

    # Discover per-LSOA timeseries parquets
    pattern = str(args.run_outdir / "*" / f"run_{args.stamp}" / f"model_timeseries_*_{args.stamp}.parquet")
    ts_paths = sorted(args.run_outdir.glob(f"*/run_{args.stamp}/model_timeseries_*_{args.stamp}.parquet"))
    if not ts_paths:
        raise FileNotFoundError(f"No timeseries parquets matching: {pattern}")
    tracker.milestone(f"found {len(ts_paths)} LSOA timeseries files")

    # Load stock once
    with tracker.section("load stock"):
        gdf = gpd.read_file(args.geojson)
        gdf[args.lsoa_col] = gdf[args.lsoa_col].astype(str)
    tracker.milestone(f"stock loaded n_dwellings={len(gdf)} "
                       f"n_lsoas={gdf[args.lsoa_col].nunique()}")

    tracker.total = len(ts_paths)
    tracker._count = 0
    rows: list[pd.DataFrame] = []
    for ts_path in ts_paths:
        # Recover lsoa_code from the directory name (its parent's parent)
        lsoa_code = ts_path.parent.parent.name
        gdf_subset = gdf[gdf[args.lsoa_col] == lsoa_code]
        if gdf_subset.empty:
            tracker.warn(f"no stock rows for {lsoa_code}; skipping")
            tracker.tick()
            continue
        try:
            annual = _aggregate_one_lsoa(ts_path, lsoa_code, gdf_subset, start_ts, end_ts)
            rows.append(annual)
        except Exception as exc:
            tracker.warn(f"{lsoa_code}: {type(exc).__name__}: {exc}")
        tracker.tick(lsoa_code)

    if not rows:
        raise RuntimeError("No LSOA aggregations succeeded.")

    out = pd.concat(rows, ignore_index=True)
    # Provenance columns to match the runner's output
    out["run_start_utc"] = args.start_utc
    out["run_end_utc"]   = args.end_utc

    tracker.milestone(f"writing {len(out)} rows to {out_path}")
    out.to_parquet(out_path, index=False)
    out.to_csv(out_path.with_suffix(".csv"), index=False)
    tracker.finish(f"wrote {out_path.name}")


if __name__ == "__main__":
    main()
