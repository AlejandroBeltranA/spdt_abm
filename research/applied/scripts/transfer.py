#!/usr/bin/env python3
"""Spatial-transfer validation: run a calibrated config on a city's EPC stock
for a target year, then merge DESNZ + apply the coverage-aware confidence tiers
(see utils.compute_confidence_tiers). No per-city refit — that's the point.

Runs the production code path (run_lsoa_batch._run_single_lsoa) over LSOAs in
a multiprocessing pool. Output rollup CSV is the one the validation notebook
and the confidence-layer figure consume.

Usage:
  python research/applied/scripts/transfer.py --city sunderland
  python research/applied/scripts/transfer.py --city waltham_forest --limit 2
  python research/applied/scripts/transfer.py --city newcastle \\
      --config results/calibration_v7_cohort/calibrated_config.yaml --year 2024
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

_THIS = Path(__file__).resolve()
REPO = _THIS.parents[3]
sys.path.insert(0, str(_THIS.parent))      # research/applied/scripts (utils, etc.)
sys.path.insert(0, str(REPO))              # repo root (household_energy)

from household_energy.config import CALIBRATED_PATH  # noqa: E402
from household_energy.run_lsoa_batch import RunConfig, _run_single_lsoa  # noqa: E402
from utils import (  # noqa: E402
    CITY_CONVENTIONS, city_convention, epc_stock_path, hidp_path_for,
    compute_confidence_tiers, summarize_confidence_tiers, CONFIDENCE_OUT_COLS,
)

DEFAULT_CONFIG = CALIBRATED_PATH  # the shipped v7_cohort calibration in household_energy/
LSOA_COL = "lsoa_code"


def _city_paths(city: str):
    conv = city_convention(city)
    # EPC stock + HIDP resolution is shared with utils.load_city_stock via the
    # centralized resolvers, so validation and transfer agree on availability.
    geo = epc_stock_path(city)                 # raises if neither .geojson/.gpkg present
    hidp = hidp_path_for(city)
    climate = REPO / "data" / f"{conv.climate_prefix}_2t_timeseries_2010_2026.parquet"
    if not climate.exists():
        raise FileNotFoundError(climate)
    return conv, geo, climate, hidp


def run_city(city: str, year: int, config: Path, limit: int | None, max_procs: int) -> pd.DataFrame:
    conv, geo, climate, hidp = _city_paths(city)
    outdir = REPO / "results_lsoa" / f"transfer_{conv.epc_slug}"
    cfg = RunConfig(
        geojson=geo, climate=climate, hidp_csv=hidp,
        start_utc=f"{year}-01-01T00:00:00Z",
        end_utc=f"{year + 1}-01-01T00:00:00Z",
        days=None, local_tz="Europe/London", lsoa_col=LSOA_COL, outdir=outdir,
        agent_collect_every=1, stamp=f"{conv.epc_slug}_{year}",
        config_path=config.resolve(), save_model_timeseries=False,
    )
    outdir.mkdir(parents=True, exist_ok=True)

    # Filter out NaN/missing lsoa_code (Manchester gpkg has 3 such rows out
    # of 205k); they can't be located spatially anyway, and sorted() chokes
    # on the mixed-dtype output of astype(str).unique() when NaNs are present.
    _lsoa_series = gpd.read_file(geo)[LSOA_COL].dropna().astype(str)
    lsoas = sorted(_lsoa_series[_lsoa_series != "nan"].unique())
    if limit:
        lsoas = lsoas[:limit]
    total = len(lsoas)
    print(f"[{city}] {total} LSOAs | config {cfg.config_path.name} | year {year}", flush=True)

    tasks = [(code, cfg, i + 1, total) for i, code in enumerate(lsoas)]
    if max_procs == 1:
        rows = [_run_single_lsoa(*t) for t in tasks]
    else:
        with mp.Pool(processes=max(1, max_procs)) as pool:
            rows = pool.starmap(_run_single_lsoa, tasks)

    abm = pd.concat([r for r in rows if r is not None], ignore_index=True)
    # Smoke runs (--limit) write to a distinct filename so they can never
    # silently overwrite a full-city rollup that downstream consumers read.
    tag = f"smoke{limit}" if limit else "all"
    out_csv = outdir / f"abm_year_{tag}_{conv.epc_slug}_{year}.csv"
    abm.to_csv(out_csv, index=False)
    print(f"[{city}] saved rollup {out_csv.name} ({len(abm)} LSOAs)", flush=True)
    return abm


def write_confidence(city: str, abm: pd.DataFrame, year: int) -> pd.DataFrame:
    c = compute_confidence_tiers(abm, city, year=year)
    summarize_confidence_tiers(c, city)
    out = REPO / "research/applied" / f"transfer_confidence_{city}_{year}.csv"
    c.sort_values(["confidence", "coverage"])[CONFIDENCE_OUT_COLS].to_csv(out, index=False)
    print(f"  wrote {out}")
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", required=True, choices=sorted(CITY_CONVENTIONS.keys()))
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help="path to calibrated_config.yaml (default: the shipped "
                         "v7_cohort calibration in household_energy/)")
    ap.add_argument("--limit", type=int, default=None, help="cap LSOA count for smoke tests")
    ap.add_argument("--max-procs", type=int, default=5)
    a = ap.parse_args()
    if not a.config.exists():
        raise FileNotFoundError(a.config)
    abm = run_city(a.city, a.year, a.config, a.limit, a.max_procs)
    if not a.limit:
        write_confidence(a.city, abm, a.year)


if __name__ == "__main__":
    main()
