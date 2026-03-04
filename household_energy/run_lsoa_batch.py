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
import random

import geopandas as gpd
import pandas as pd

from household_energy.climate import ClimateField
from household_energy.config import load_config
from household_energy.model import EnergyModel

_GDF_ENRICHED_CACHE: dict[tuple[str, str | None], gpd.GeoDataFrame] = {}
_CLIMATE_CACHE: dict[str, ClimateField] = {}


# ───────────────────────────── dataclass config ──────────────────────────────
@dataclass
class RunConfig:
    geojson: Path
    climate: Path
    hidp_csv: Optional[Path]
    config_path: Optional[Path]
    start_utc: Optional[str]
    end_utc: Optional[str]
    days: Optional[int]
    local_tz: str
    lsoa_col: str
    outdir: Path
    agent_collect_every: int
    save_model_timeseries: bool
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
    # Preserve canonical columns that may be suffixed by the merge.
    for base in ["lsoa_code", "ward_code", "local_authority"]:
        geo_col = f"{base}_geo"
        hidp_col = f"{base}_hidp"
        if base not in merged.columns and (geo_col in merged.columns or hidp_col in merged.columns):
            if geo_col in merged.columns and hidp_col in merged.columns:
                merged[base] = merged[geo_col].combine_first(merged[hidp_col])
            elif geo_col in merged.columns:
                merged[base] = merged[geo_col]
            else:
                merged[base] = merged[hidp_col]
    # Wealth: prefer direct hh_income_band (already 5 buckets). Fallback is deterministic.
    income_col = merged.get("hh_income_band")
    if income_col is not None:
        merged["wealth_bucket"] = income_col.astype(str).str.strip().str.lower()
    else:
        bands = ["q1_lowest", "q2_low", "q3_mid", "q4_high", "q5_highest"]
        merged["wealth_bucket"] = [
            random.Random(hash(str(k)) & 0xFFFFFFFF).choice(bands)
            for k in merged.get(geo_uprn_field, merged.index).astype(str)
        ]
    return merged


def _load_lsoa_gdf(cfg: RunConfig, lsoa_code: str) -> gpd.GeoDataFrame:
    cache_key = (str(cfg.geojson.resolve()), str(cfg.hidp_csv.resolve()) if cfg.hidp_csv else None)
    gdf_all = _GDF_ENRICHED_CACHE.get(cache_key)
    if gdf_all is None:
        gdf_all = gpd.read_file(cfg.geojson)
        gdf_all = _enrich_with_hidp(gdf_all, cfg.hidp_csv)
        _GDF_ENRICHED_CACHE[cache_key] = gdf_all

    if cfg.lsoa_col not in gdf_all.columns:
        geo_col = f"{cfg.lsoa_col}_geo"
        hidp_col = f"{cfg.lsoa_col}_hidp"
        if geo_col in gdf_all.columns or hidp_col in gdf_all.columns:
            if geo_col in gdf_all.columns and hidp_col in gdf_all.columns:
                gdf_all[cfg.lsoa_col] = gdf_all[geo_col].combine_first(gdf_all[hidp_col])
            elif geo_col in gdf_all.columns:
                gdf_all[cfg.lsoa_col] = gdf_all[geo_col]
            else:
                gdf_all[cfg.lsoa_col] = gdf_all[hidp_col]
    if cfg.lsoa_col not in gdf_all.columns:
        raise KeyError(f"{cfg.lsoa_col} not found in {cfg.geojson}")
    gdf = gdf_all[gdf_all[cfg.lsoa_col].astype(str) == lsoa_code].copy()
    if gdf.empty:
        print(f"⚠️  Skipping {lsoa_code}: no dwellings in source GeoJSON.")
        return gdf
    return gdf


def _build_model(gdf: gpd.GeoDataFrame, cfg: RunConfig):
    clim_key = str(cfg.climate.resolve())
    cf = _CLIMATE_CACHE.get(clim_key)
    if cf is None:
        cf = ClimateField(cfg.climate)
        _CLIMATE_CACHE[clim_key] = cf
    start_ts, end_ts, T_hours = _time_window(cf, cfg.start_utc, cfg.end_utc, cfg.days)
    model = EnergyModel(
        gdf=gdf,
        climate_parquet=str(cfg.climate),
        climate_start=start_ts,
        local_tz=cfg.local_tz,
        collect_agent_level=False,
        agent_collect_every=cfg.agent_collect_every,
        config_path=str(cfg.config_path) if cfg.config_path else None,
    )
    return model, T_hours, start_ts, end_ts


def _run_model_hours(model: EnergyModel, T_hours: int, lsoa_code: str, start_ts: pd.Timestamp):
    annual_acc: dict[int, dict[str, float]] = {}
    for h in range(T_hours):
        model.step()
        ts = start_ts + pd.to_timedelta(h, unit="h")
        y = int(ts.year)
        acc = annual_acc.setdefault(y, {"abm_kwh": 0.0, "abm_elec_kwh": 0.0, "abm_gas_kwh": 0.0, "abm_other_kwh": 0.0})
        acc["abm_kwh"] += float(getattr(model, "total_energy", 0.0))
        acc["abm_elec_kwh"] += float(getattr(model, "total_electric_kwh", 0.0))
        acc["abm_gas_kwh"] += float(getattr(model, "total_gas_kwh", 0.0))
        acc["abm_other_kwh"] += float(getattr(model, "total_other_kwh", 0.0))
    print(f"    completed {T_hours:,} steps for {lsoa_code}", flush=True)
    return annual_acc


def _count_gas_connected_dwellings(gdf: gpd.GeoDataFrame) -> dict[str, int]:
    """Count dwellings with a gas connection (best-effort from EPC flags/fields).

    Signals (combined best-effort):
      - `is_off_gas == 1` => off-gas
      - `is_off_gas == 0` => gas-connected
      - `is_gas == 1` => gas-connected
      - `main_fuel_type` contains "gas" => gas-connected

    Notes:
      - Many EPC extracts have sparse `is_off_gas`; when it's missing/NaN we
        fall back to the other signals rather than classifying as unknown.
    """
    n = int(len(gdf))
    if n == 0:
        return {"gas_connected": 0, "off_gas": 0, "unknown": 0}

    off = pd.Series(False, index=gdf.index)
    on_strict = pd.Series(False, index=gdf.index)

    if "is_off_gas" in gdf.columns:
        s = pd.to_numeric(gdf["is_off_gas"], errors="coerce")
        off |= (s == 1)
        on_strict |= (s == 0)

    if "is_gas" in gdf.columns:
        s = pd.to_numeric(gdf["is_gas"], errors="coerce")
        on_strict |= (s == 1)

    if "main_fuel_type" in gdf.columns:
        s = gdf["main_fuel_type"].astype(str).str.lower()
        on_strict |= s.str.contains("gas", na=False)

    # Ensure mutual exclusivity (off-gas wins if explicitly flagged).
    on_strict &= ~off
    unk = ~(off | on_strict)

    # Assumed-connection count: treat unknown as connected (common when EPC flags are sparse).
    on_assumed = ~off
    return {
        "gas_connected": int(on_assumed.sum()),
        "gas_connected_strict": int(on_strict.sum()),
        "off_gas": int(off.sum()),
        "unknown": int(unk.sum()),
    }


def _count_heating_fuel_buckets(model: EnergyModel) -> dict[str, int]:
    """Count dwellings by the model's heating-fuel bucket classifier (electric/gas/other)."""
    counts: dict[str, int] = {"electric": 0, "gas": 0, "other": 0}
    for hh in getattr(model, "household_agents", []):
        bucket = None
        try:
            bucket = hh._heating_fuel_bucket()  # HouseholdAgent private helper
        except Exception:
            bucket = None
        if bucket not in counts:
            bucket = "other"
        counts[bucket] += 1
    return counts


def _run_single_lsoa(lsoa_code: str, cfg: RunConfig, idx: int = 1, total: int = 1) -> Optional[pd.DataFrame]:
    """Run model for one LSOA; return annual summary frame."""
    print(f"[{idx}/{total}] LSOA {lsoa_code} – loading dwellings…", flush=True)
    gdf = _load_lsoa_gdf(cfg, lsoa_code)
    if gdf.empty:
        return None

    model, T_hours, start_ts, end_ts = _build_model(gdf, cfg)
    conn_counts = _count_gas_connected_dwellings(gdf)
    heat_counts = _count_heating_fuel_buckets(model)
    annual_acc = _run_model_hours(model, T_hours, lsoa_code, start_ts)

    if cfg.save_model_timeseries and model.model_dc is not None:
        mdl = model.model_dc.get_model_vars_dataframe().copy()
        mdl["hour_start_utc"] = start_ts + pd.to_timedelta(mdl.index - 1, unit="h")
        mdl = mdl.set_index("hour_start_utc").iloc[1:]  # drop t0 snapshot
        mdl = mdl.loc[(mdl.index >= start_ts) & (mdl.index < end_ts)]
    else:
        mdl = None

    annual = pd.DataFrame(
        [{"year": y, **vals} for y, vals in sorted(annual_acc.items(), key=lambda kv: kv[0])]
    )
    annual["lsoa_code"] = lsoa_code
    dwellings = len(gdf)
    annual["run_dwellings"] = dwellings
    # Gas connection (EPC sample) — intended to be comparable to DESNZ gas meter counts.
    annual["run_gas_dwellings"] = int(conn_counts.get("gas_connected", 0))
    annual["run_gas_dwellings_strict"] = int(conn_counts.get("gas_connected_strict", 0))
    annual["run_off_gas_dwellings"] = int(conn_counts.get("off_gas", 0))
    annual["run_unknown_gas_connection_dwellings"] = int(conn_counts.get("unknown", 0))

    # Heating fuel bucket (model classification) — useful for internal diagnostics (not meters).
    annual["run_gas_heated_dwellings"] = int(heat_counts.get("gas", 0))
    annual["run_electric_heated_dwellings"] = int(heat_counts.get("electric", 0))
    annual["run_other_heated_dwellings"] = int(heat_counts.get("other", 0))
    # Back-compat (older exports used these names for *heated* buckets)
    annual["run_electric_dwellings"] = annual["run_electric_heated_dwellings"]
    annual["run_other_dwellings"] = annual["run_other_heated_dwellings"]
    annual["abm_kwh_per_dw"] = annual["abm_kwh"] / dwellings
    if "abm_elec_kwh" in annual.columns:
        annual["abm_elec_kwh_per_dw"] = annual["abm_elec_kwh"] / dwellings
    if "abm_gas_kwh" in annual.columns:
        annual["abm_gas_kwh_per_dw"] = annual["abm_gas_kwh"] / dwellings
        gas_connected_dw = int(conn_counts.get("gas_connected", 0))
        gas_heated_dw = int(heat_counts.get("gas", 0))
        # For "gas connected homes" lens:
        annual["abm_gas_kwh_per_gas_dw"] = annual["abm_gas_kwh"] / gas_connected_dw if gas_connected_dw > 0 else float("nan")
        # For model-internal "gas heated homes" lens:
        annual["abm_gas_kwh_per_gas_heated_dw"] = annual["abm_gas_kwh"] / gas_heated_dw if gas_heated_dw > 0 else float("nan")
    if "abm_other_kwh" in annual.columns:
        annual["abm_other_kwh_per_dw"] = annual["abm_other_kwh"] / dwellings

    outdir = cfg.outdir / lsoa_code / f"run_{cfg.stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    if cfg.save_model_timeseries and mdl is not None:
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
    p.add_argument("--config-path", default=None, help="Optional YAML config override")
    p.add_argument("--start-utc", default="2020-01-01T00:00:00Z")
    p.add_argument("--end-utc", default="2025-01-01T00:00:00Z")
    p.add_argument("--days", type=int, default=None, help="Optional if start/end not supplied")
    p.add_argument("--local-tz", default="Europe/London")
    p.add_argument("--lsoa-col", default="lsoa_code")
    p.add_argument("--lsoas", nargs="*", default=None, help="LSOA codes to run; default = all in geojson")
    p.add_argument("--outdir", default="results_lsoa")
    p.add_argument("--agent-collect-every", type=int, default=1)
    p.add_argument("--no-model-timeseries", action="store_true", help="Skip hourly model parquet export per LSOA")
    p.add_argument("--max-procs", type=int, default=max(1, mp.cpu_count() // 2))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d")
    cfg = RunConfig(
        geojson=Path(args.geojson),
        climate=Path(args.climate),
        hidp_csv=Path(args.hidp_csv).resolve() if args.hidp_csv else None,
        config_path=Path(args.config_path).resolve() if args.config_path else None,
        start_utc=args.start_utc,
        end_utc=args.end_utc,
        days=args.days,
        local_tz=args.local_tz,
        lsoa_col=args.lsoa_col,
        outdir=Path(args.outdir).resolve(),
        agent_collect_every=args.agent_collect_every,
        save_model_timeseries=not args.no_model_timeseries,
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
