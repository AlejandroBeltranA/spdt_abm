#!/usr/bin/env python
"""
run.py – batch executor for the Household-Energy ABM
===================================================

Headless (non-GUI) entry point that:

1) reads a GeoJSON of building polygons,
2) instantiates `EnergyModel` with a climate parquet,
3) runs either for a wall-clock window (UTC) or for `days × 24` hours,
4) writes artefacts to `--outdir`:

   ┌──────────────────────────────────────────┐
   │ energy_timeseries.csv                    │  – simple per-hour totals/avg
   │ model_timeseries.parquet                 │  – DataCollector (model-level)
   │ agent_timeseries.parquet (optional)      │  – DataCollector (agent-level)
   │ model_hourly.parquet                     │  – model-level w/ UTC index
   │ model_daily.parquet                      │  – daily aggregates
   │ energy_model.pkl (optional)              │  – full pickled model
   └──────────────────────────────────────────┘

USAGE EXAMPLES
--------------
# 7-day quick run (uses climate start index if provided)
python -m household_energy.run \
  data/ncc_neighborhood.geojson \
  --climate data/ncc_2t_timeseries_2010_2039.parquet \
  --days 7 \
  --outdir results \
  --no-agent-level
or
energy-run \
  data/ncc_neighborhood.geojson \
  --climate data/ncc_2t_timeseries_2010_2039.parquet \
  --days 7 \
  --outdir results \
  --no-agent-level

# Windowed run aligned to climate (recommended for validation)
python run.py data/ncc_neighborhood.geojson \
  --climate data/hourly_climate.parquet \
  --start-utc 2020-01-01T00:00:00Z 
  --end-utc 2025-01-01T00:00:00Z \
  --outdir results_2020_2024 
  --no-agent-level
or
energy-run
 data/ncc_neighborhood.geojson \
  --climate data/hourly_climate.parquet \
  --start-utc 2020-01-01T00:00:00Z 
  --end-utc 2025-01-01T00:00:00Z \
  --outdir results_2020_2024 
  --no-agent-level
"""


from __future__ import annotations

import argparse
from pathlib import Path
import time

import cloudpickle as pickle
import geopandas as gpd
import pandas as pd

from household_energy.model import EnergyModel
from household_energy.climate import ClimateField


# ────────────────────────── CLI parser ────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the ABM and export time-series.")
    p.add_argument("geojson", help="Path to neighbourhood GeoJSON")

    # Climate/time window
    p.add_argument("--climate", required=True,
                   help="Path to hourly climate parquet prepared by 01-climate-prep.ipynb")
    p.add_argument("--start-utc", default=None,
                   help="ISO UTC start timestamp (e.g., 2020-01-01T00:00:00Z). If omitted, uses climate start.")
    p.add_argument("--end-utc", default=None,
                   help="ISO UTC end timestamp (exclusive). If omitted and --days not set, uses full climate span.")
    p.add_argument("--days", type=int, default=None,
                   help="Optional duration in days (24h steps). Ignored if both --start-utc and --end-utc provided.")
    p.add_argument("--local-tz", default="Europe/London",
                   help="IANA local timezone for model agents (default: Europe/London)")

    # Performance / outputs
    p.add_argument("--no-agent-level", action="store_true",
                   help="Disable agent-level DataCollector for speed (recommended for long runs).")
    p.add_argument("--agent-collect-every", type=int, default=24,
                   help="Collect agent-level data every N hours (default: 24).")
    p.add_argument("--no-pickle", action="store_true",
                   help="Skip writing energy_model.pkl to save disk/time.")
    p.add_argument("--outdir", default=".",
                   help="Output folder for CSV / Parquet / pickle (default: .)")
    p.add_argument("--print-every-hours", type=int, default=24*7,
                   help="Progress print frequency (in hours). Default: 168 (weekly).")

    return p.parse_args()


# ──────────────────────────── helpers ─────────────────────────────
def _safe_sum_household_energy(model: EnergyModel) -> float:
    # Mirrors DataCollector 'total_energy' but avoids assumptions.
    return float(sum(getattr(h, "energy_consumption", 0.0) for h in model.household_agents))


# ──────────────────────────── main ────────────────────────────────
def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Sanity: both or neither for explicit window
    if (args.start_utc is None) ^ (args.end_utc is None):
        raise ValueError("Provide both --start-utc and --end-utc, or neither.")

    # 1 ─ Load geometry and climate metadata
    gdf = gpd.read_file(args.geojson)
    cf = ClimateField(args.climate)

    # Determine time window (in hours) aligned to climate indices
    if args.start_utc and args.end_utc:
        start_utc = pd.Timestamp(args.start_utc, tz="UTC")
        end_utc   = pd.Timestamp(args.end_utc,   tz="UTC")
        i0 = cf.time_index_for(start_utc)
        i1 = cf.time_index_for(end_utc)
        if i1 <= i0:
            raise ValueError("Invalid window: end <= start, or climate does not cover the requested range.")
        T_hours   = i1 - i0
        start_utc = pd.to_datetime(cf.times[i0], utc=True)  # align to exact grid
    else:
        # Fall back to a days-based duration from the climate file's beginning
        i0 = 0
        start_utc = pd.to_datetime(cf.times[i0], utc=True)
        if args.days is None:
            # Use the whole climate span
            T_hours = int(len(cf.times)) - i0
        else:
            T_hours = int(args.days) * 24
            if T_hours <= 0:
                raise ValueError("--days must be positive.")

    # 2 ─ Build model (optionally disable agent-level collection; set cadence)
    t0 = time.perf_counter()
    model = EnergyModel(
        gdf=gdf,
        climate_parquet=args.climate,
        climate_start=start_utc,
        local_tz=args.local_tz,
        collect_agent_level=not args.no_agent_level,
        agent_collect_every=args.agent_collect_every,
    )
    init_s = time.perf_counter() - t0
    print(f"Init: {init_s:.2f}s | households={len(model.household_agents):,}, persons={len(model.person_agents):,}")
    print(f"Window hours: {T_hours:,} | start: {start_utc} | "
          f"end(excl): {start_utc + pd.to_timedelta(T_hours, unit='h')}")

    # 3 ─ Run
    records = []  # simple per-hour summary (CSV)
    t_run = time.perf_counter()
    print_every = max(1, int(args.print_every_hours))
    for h in range(T_hours):
        model.step()

        # per-hour rollup for compatibility CSV
        tot = _safe_sum_household_energy(model)
        records.append(
            dict(
                step=h,
                hour=h % 24,
                day=h // 24,
                total_energy=tot,
                avg_energy=tot / len(model.household_agents) if len(model.household_agents) else 0.0,
            )
        )
        if (h + 1) % print_every == 0:
            print(f" progressed {h+1:,}/{T_hours:,} hours")
    print(f"Run done: {time.perf_counter() - t_run:.1f}s")

    # 4 ─ Pull model-level hourly frame (label by hour start)
    mdl = model.model_dc.get_model_vars_dataframe().copy()
    # DataCollector has a t=0 snapshot at index 0; the first step writes at index 1.
    mdl["hour_start_utc"] = start_utc + pd.to_timedelta(mdl.index - 1, unit="h")
    mdl = mdl.set_index("hour_start_utc").iloc[1:]  # drop t=0 snapshot

    # Clamp exactly like the notebook (timestamp range if provided, else count)
    if args.start_utc and args.end_utc:
        end_utc = pd.Timestamp(args.end_utc, tz="UTC")
        mdl = mdl.loc[(mdl.index >= start_utc) & (mdl.index < end_utc)]
    else:
        mdl = mdl.iloc[:T_hours]

    # 5 ─ Build daily aggregates (sum energy; average ambient temperature)
    prop_cols = [c for c in mdl.columns
                 if hasattr(model, "energy_by_type") and c in getattr(model, "energy_by_type", {}).keys()]
    wealth_cols = [c for c in ("high", "medium", "low") if c in mdl.columns]

    daily = pd.DataFrame({
        "total_energy_kWh":       mdl["total_energy"].resample("D").sum(),
        "ambient_mean_tempC_avg": mdl["ambient_mean_tempC"].resample("D").mean()
                                   if "ambient_mean_tempC" in mdl.columns else pd.NA,
    })
    for c in prop_cols:
        daily[f"{c}_kWh"] = mdl[c].resample("D").sum()
    for c in wealth_cols:
        daily[f"wealth_{c}_kWh"] = mdl[c].resample("D").sum()

    # Match the ipynb’s explicit calendar slice when a window is provided
    if args.start_utc and args.end_utc:
        start_date = start_utc.normalize().date()
        end_date_excl = pd.Timestamp(args.end_utc, tz="UTC").normalize().date()
        last_inclusive = (pd.Timestamp(end_date_excl) - pd.Timedelta(days=1)).date()
        daily = daily.loc[str(start_date):str(last_inclusive)]

    # 6 ─ Write outputs
    # 6-a: simple CSV with hourly totals
    csv_path = outdir / "energy_timeseries.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)

    # 6-b: Parquet tables
    model_parquet = outdir / "model_timeseries.parquet"
    mdl.to_parquet(model_parquet)

    # Agent-level export (only if enabled and something was collected)
    agent_parquet = None
    if not args.no_agent_level:
        if getattr(model, "agent_dc", None) is None:
            print("ℹ️ Agent-level collection is disabled in the model; skipping agent_timeseries.parquet.")
        else:
            try:
                agent_df = model.agent_dc.get_agent_vars_dataframe()
                if isinstance(agent_df, pd.DataFrame) and not agent_df.empty:
                    agent_parquet = outdir / "agent_timeseries.parquet"
                    agent_df.to_parquet(agent_parquet)
                    print(f"Saved agent-level → {agent_parquet}")
                else:
                    print("ℹ️ Agent DataCollector returned empty frame; skipping agent_timeseries.parquet.")
            except Exception as e:
                print(f"ℹ️ Skipping agent_timeseries.parquet: {e.__class__.__name__}: {e}")

    # 6-c: “window runner” convenience outputs
    hourly_out = outdir / "model_hourly.parquet"
    daily_out  = outdir / "model_daily.parquet"
    mdl.to_parquet(hourly_out)
    daily.to_parquet(daily_out)

    # 6-d: Optional pickle of entire model
    pickle_path = outdir / "energy_model.pkl"
    if not args.no_pickle:
        with open(pickle_path, "wb") as fh:
            pickle.dump(model, fh)
    else:
        pickle_path = None

    # 7 ─ Console summary
    written = [
        f"• {csv_path.name}",
        f"• {model_parquet.name}",
        f"• {hourly_out.name}",
        f"• {daily_out.name}",
    ]
    if agent_parquet:
        written.append(f"• {agent_parquet.name}")
    if pickle_path:
        written.append(f"• {pickle_path.name}")

    total_energy_kwh = float(pd.DataFrame(records)["total_energy"].sum())
    print(
        "✅ Simulation complete\n"
        f"   Hours simulated   : {T_hours:,}\n"
        f"   Overall energy    : {total_energy_kwh:,.2f} kWh\n"
        f"   Files written     :\n   " + "\n   ".join(written)
    )


# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
