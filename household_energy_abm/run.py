#!/usr/bin/env python
"""
run.py – batch executor for the Household-Energy ABM
===================================================

This script is the *headless* (non-GUI) entry point that:

1. reads a GeoJSON of building polygons,
2. instantiates `EnergyModel`,
3. runs it for `days × 24` hourly steps,
4. writes three data artefacts to `--outdir`:

   ┌───────────────────────────────┐
   │ energy_timeseries.csv         │  – per-hour total & average kWh
   │ model_timeseries.parquet      │  – DataCollector (model-level vars)
   │ agent_timeseries.parquet      │  – DataCollector (agent-level vars)
   └───────────────────────────────┘

Optionally the **entire model object** is pickled with *cloudpickle* so you can
reload the exact agent graph for debugging or additional analysis::

    import cloudpickle, pprint
    with open("energy_model.pkl", "rb") as fh:
        model = cloudpickle.load(fh)
    pprint.pprint(model.household_agents[0].__dict__)

Example usage::

    python run.py data/ncc_neighborhood.geojson --days 7 --outdir results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cloudpickle as pickle
import geopandas as gpd
import pandas as pd

from model import EnergyModel


# ────────────────────────── CLI parser ────────────────────────────
def parse_args() -> argparse.Namespace:
    """Return user-provided command-line parameters."""
    p = argparse.ArgumentParser(description="Run the ABM and export time-series.")
    p.add_argument("geojson", help="Path to neighbourhood GeoJSON")
    p.add_argument("--days", type=int, default=1,
                   help="Number of days to simulate (default: 1)")
    p.add_argument("--outdir", default=".",
                   help="Output folder for CSV / Parquet / pickle (default: .)")
    return p.parse_args()


# ──────────────────────────── main ────────────────────────────────
def main() -> None:
    args   = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # 1 ─ load geometry + build model ---------------------------------
    gdf   = gpd.read_file(args.geojson)
    model = EnergyModel(gdf)

    # 2 ─ run simulation ----------------------------------------------
    steps   = args.days * 24
    records = []                                    # per-hour summary rows

    for step in range(steps):
        model.step()

        tot = sum(h.energy_consumption for h in model.household_agents)
        records.append(
            dict(
                step=step,
                hour=step % 24,
                day=step // 24,
                total_energy=tot,
                avg_energy=tot / len(model.household_agents),
            )
        )

    # 3 ─ write outputs ----------------------------------------------
    # 3-a CSV with hourly totals
    df = pd.DataFrame(records)
    csv_path = outdir / "energy_timeseries.csv"
    df.to_csv(csv_path, index=False)

    # 3-b Parquet tables from Mesa’s DataCollector
    model_parquet  = outdir / "model_timeseries.parquet"
    agent_parquet  = outdir / "agent_timeseries.parquet"
    model.datacollector.get_model_vars_dataframe().to_parquet(model_parquet)
    model.datacollector.get_agent_vars_dataframe().to_parquet(agent_parquet)

    # 3-c Optional: pickle entire model for replay
    pickle_path = outdir / "energy_model.pkl"
    with open(pickle_path, "wb") as fh:
        pickle.dump(model, fh)

    # 4 ─ console summary --------------------------------------------
    print(
        "✅ Simulation complete\n"
        f"   Steps simulated   : {steps}\n"
        f"   Overall energy    : {df['total_energy'].sum():,.2f} kWh\n"
        f"   Files written     :\n"
        f"     • {csv_path.name}\n"
        f"     • {model_parquet.name}\n"
        f"     • {agent_parquet.name}\n"
        f"     • {pickle_path.name}"
    )


# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
