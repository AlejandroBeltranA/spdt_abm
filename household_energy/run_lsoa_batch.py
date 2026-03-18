#!/usr/bin/env python
"""
run_lsoa_batch.py — run the Household-Energy ABM once per LSOA and export
hourly plus annual totals.

Purpose
    - Keep memory/runtime bounded by simulating one LSOA at a time.
    - Produce clean annual kWh and per-dwelling metrics ready to join with
      DESNZ LSOA statistics.

Inputs
    - GeoJSON with dwellings and an LSOA code column (default: data/epc_abm_newcastle.geojson).
    - Hourly climate parquet (default: data/ncc_2t_timeseries_2010_2039.parquet).
    - Optional HIDP CSV for enrichment (same join rules as run.py).

Outputs (per LSOA, under results_lsoa/<LSOA>/run_<YYYYMMDD>/)
    - model_timeseries_<LSOA>_<stamp>.parquet  — hourly model DataCollector (t0 dropped)
    - abm_year_<LSOA>_<stamp>.parquet/.csv     — annual totals + per-dwelling
Combined rollup
    - results_lsoa/abm_year_all_<stamp>.parquet/.csv

Usage
    energy-run-lsoa --geojson data/epc_abm_newcastle.geojson \
      --climate data/ncc_2t_timeseries_2010_2039.parquet \
      --start-utc 2020-01-01T00:00:00Z --end-utc 2025-01-01T00:00:00Z \
      --max-procs 4

Notes
    - Uses agent_collect_every=1 for accurate hourly totals; agent-level
      collection is disabled to keep runs light.
    - Per-dwelling denominators use only the dwellings included in that LSOA
      run (no EPC vs run-pop mismatch).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd
import pandas as pd

from household_energy.climate import ClimateField
from household_energy.config import load_config
from household_energy.model import EnergyModel


# ───────────────────────────── dataclass config ──────────────────────────────
@dataclass
class RunConfig:
    geojson: Path
    climate: Path
    hidp_csv: Optional[Path]
    start_utc: Optional[str]
    end_utc: Optional[str]
    days: Optional[int]
    local_tz: str
    lsoa_col: str
    outdir: Path
    agent_collect_every: int
    stamp: str


# ───────────────────────────── helper functions ──────────────────────────────
def _time_window(cf: ClimateField, start_utc: Optional[str], end_utc: Optional[str], days: Optional[int]):
    """Align requested window to climate index; return (start_ts, end_ts, T_hours)."""
    if (start_utc is None) ^ (end_utc is None):
        raise ValueError("Provide both --start-utc and --end-utc, or neither.")

    if start_utc and end_utc:
        start_ts = pd.Timestamp(start_utc, tz="UTC")
        end_ts = pd.Timestamp(end_utc, tz="UTC")
        i0 = cf.time_index_for(start_ts)
        i1 = cf.time_index_for(end_ts)
        if i1 <= i0:
            raise ValueError("end_utc must be after start_utc and within climate range")
        T = i1 - i0
        start_ts = pd.to_datetime(cf.times[i0], utc=True)  # snap to grid
    else:
        start_ts = pd.to_datetime(cf.times[0], utc=True)
        if days:
            T = int(days) * 24
            if T <= 0:
                raise ValueError("--days must be positive")
        else:
            T = len(cf.times)
        end_ts = start_ts + pd.to_timedelta(T, unit="h")
    return start_ts, end_ts, T


def _enrich_with_hidp(gdf: gpd.GeoDataFrame, hidp_csv: Optional[Path]) -> gpd.GeoDataFrame:
    """Optional HIDP merge (mirrors run.py but minimal)."""
    if not hidp_csv:
        return gdf
    if not hidp_csv.exists():
        raise FileNotFoundError(f"HIDP CSV not found: {hidp_csv}")

    cfg_defaults = load_config(None)
    merge_on_csv = cfg_defaults.households.get("merge_on", "uprn_chr")
    geo_uprn_field = cfg_defaults.households.get("geojson_uprn_field", "UPRN")
    for alt in ["UPRN", "uprn", "fid"]:
        if geo_uprn_field not in gdf.columns and alt in gdf.columns:
            geo_uprn_field = alt

    gdf[geo_uprn_field] = gdf[geo_uprn_field].astype(str).str.strip()
    hidp_df = pd.read_csv(hidp_csv, low_memory=False)
    hidp_df.columns = [c.strip() for c in hidp_df.columns]
    hidp_df[merge_on_csv] = hidp_df[merge_on_csv].astype(str).str.strip()
    hidp_df = hidp_df.drop_duplicates(subset=[merge_on_csv])

    merged = gdf.merge(
        hidp_df,
        how="left",
        left_on=geo_uprn_field,
        right_on=merge_on_csv,
        suffixes=("_geo", "_hidp"),
    )
    return merged


def _run_single_lsoa(lsoa_code: str, cfg: RunConfig, idx: int = 1, total: int = 1) -> Optional[pd.DataFrame]:
    """Run model for one LSOA; return annual summary frame."""
    print(f"[{idx}/{total}] LSOA {lsoa_code} – loading dwellings…", flush=True)
    gdf = gpd.read_file(cfg.geojson)
    if cfg.lsoa_col not in gdf.columns:
        raise KeyError(f"{cfg.lsoa_col} not found in {cfg.geojson}")
    gdf = gdf[gdf[cfg.lsoa_col].astype(str) == lsoa_code]
    if gdf.empty:
        print(f"⚠️  Skipping {lsoa_code}: no dwellings in source GeoJSON.")
        return None

    gdf = _enrich_with_hidp(gdf, cfg.hidp_csv)
    cf = ClimateField(cfg.climate)
    start_ts, end_ts, T_hours = _time_window(cf, cfg.start_utc, cfg.end_utc, cfg.days)

    model = EnergyModel(
        gdf=gdf,
        climate_parquet=str(cfg.climate),
        climate_start=start_ts,
        local_tz=cfg.local_tz,
        collect_agent_level=False,
        agent_collect_every=cfg.agent_collect_every,
    )

    for h in range(T_hours):
        model.step()
        if (h + 1) % max(1, T_hours // 10) == 0:
            print(f"    step {h+1:,}/{T_hours:,} ({(h+1)/T_hours: .0%}) for {lsoa_code}", flush=True)

    mdl = model.model_dc.get_model_vars_dataframe().copy()
    mdl["hour_start_utc"] = start_ts + pd.to_timedelta(mdl.index - 1, unit="h")
    mdl = mdl.set_index("hour_start_utc").iloc[1:]  # drop t0 snapshot
    mdl = mdl.loc[(mdl.index >= start_ts) & (mdl.index < end_ts)]

    annual = (
        mdl["total_energy"]
        .resample("YS")
        .sum()
        .rename("abm_kwh")
        .to_frame()
    )
    annual["year"] = annual.index.year
    annual["lsoa_code"] = lsoa_code
    dwellings = len(gdf)
    annual["run_dwellings"] = dwellings
    annual["abm_kwh_per_dw"] = annual["abm_kwh"] / dwellings

    outdir = cfg.outdir / lsoa_code / f"run_{cfg.stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    mdl.to_parquet(outdir / f"model_timeseries_{lsoa_code}_{cfg.stamp}.parquet")
    annual.to_parquet(outdir / f"abm_year_{lsoa_code}_{cfg.stamp}.parquet", index=False)
    annual.to_csv(outdir / f"abm_year_{lsoa_code}_{cfg.stamp}.csv", index=False)
    return annual.reset_index(drop=True)


# ──────────────────────────────── CLI ────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the ABM once per LSOA and export annual totals.")
    p.add_argument("--geojson", default="data/epc_abm_newcastle.geojson")
    p.add_argument("--climate", default="data/ncc_2t_timeseries_2010_2039.parquet")
    p.add_argument("--hidp-csv", default=None)
    p.add_argument("--start-utc", default="2020-01-01T00:00:00Z")
    p.add_argument("--end-utc", default="2025-01-01T00:00:00Z")
    p.add_argument("--days", type=int, default=None, help="Optional if start/end not supplied")
    p.add_argument("--local-tz", default="Europe/London")
    p.add_argument("--lsoa-col", default="lsoa_code")
    p.add_argument("--lsoas", nargs="*", default=None, help="LSOA codes to run; default = all in geojson")
    p.add_argument("--outdir", default="results_lsoa")
    p.add_argument("--agent-collect-every", type=int, default=1)
    p.add_argument("--max-procs", type=int, default=max(1, mp.cpu_count() // 2))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d")
    cfg = RunConfig(
        geojson=Path(args.geojson),
        climate=Path(args.climate),
        hidp_csv=Path(args.hidp_csv).resolve() if args.hidp_csv else None,
        start_utc=args.start_utc,
        end_utc=args.end_utc,
        days=args.days,
        local_tz=args.local_tz,
        lsoa_col=args.lsoa_col,
        outdir=Path(args.outdir).resolve(),
        agent_collect_every=args.agent_collect_every,
        stamp=stamp,
    )

    cfg.outdir.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_file(cfg.geojson)
    all_lsoas = sorted(gdf[cfg.lsoa_col].astype(str).unique())
    target_lsoas: Iterable[str] = args.lsoas if args.lsoas else all_lsoas

    target_lsoas = list(target_lsoas)
    total = len(target_lsoas)
    print(f"Running {total} LSOAs → {cfg.outdir} (stamp {stamp})")

    tasks = [(lsoa, cfg, i + 1, total) for i, lsoa in enumerate(target_lsoas)]
    if args.max_procs == 1:
        rows = [_run_single_lsoa(*task) for task in tasks]
    else:
        with mp.Pool(processes=max(1, args.max_procs)) as pool:
            rows = pool.starmap(_run_single_lsoa, tasks)

    annual_all = pd.concat([r for r in rows if r is not None], ignore_index=True)
    if not annual_all.empty:
        annual_all.to_parquet(cfg.outdir / f"abm_year_all_{stamp}.parquet", index=False)
        annual_all.to_csv(cfg.outdir / f"abm_year_all_{stamp}.csv", index=False)
        print(f"Saved combined annual table → abm_year_all_{stamp}.parquet/csv")
    else:
        print("No LSOA runs produced output (check inputs).")


if __name__ == "__main__":
    main()
