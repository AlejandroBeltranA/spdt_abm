"""Full-city per-home decomposition with the v7 calibrated model, RESUMABLE.

A thin checkpoint driver: it loops over every Newcastle LSOA and calls the shared
``decompose_demand.run_decomposition`` (the same code path notebook 3 imports),
appending each LSOA's per-dwelling rows to one CSV. It carries no model logic of
its own. The only thing it adds over calling ``run_decomposition`` with the full
LSOA list is resumability: it skips LSOAs already in the CSV, so a kill just pauses
it and re-running continues. That matters because the full-city run is long and gets
killed on machine sleep.

  .venv/bin/python research/applied/scripts/run_city_decomp.py
Writes results_lsoa/decomp_city_newcastle_2023_v7.csv (read by the policy notebook).
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))                                       # household_energy
sys.path.insert(0, str(REPO / "research" / "applied" / "scripts"))  # decompose_demand
os.chdir(REPO)

import geopandas as gpd, pandas as pd
from decompose_demand import run_decomposition

YEAR = 2023
GEO = "data/epc_abm_newcastle.geojson"
CFG = "results/calibration_v7_cohort/calibrated_config.yaml"
OUT = Path("results_lsoa/decomp_city_newcastle_2023_v7.csv")
# The committed schema the policy notebook reads; keep it exactly so appends stay compatible.
COLS = ["unique_id", "lsoa_code", "heating_bucket", "base_kwh", "heat_kwh", "spike_kwh",
        "electric_kwh", "gas_kwh"]


def main():
    gdf = gpd.read_file(GEO); gdf["lsoa_code"] = gdf["lsoa_code"].astype(str)
    all_lsoas = sorted(gdf.lsoa_code.unique())
    done = set()
    if OUT.exists():
        done = set(pd.read_csv(OUT, usecols=["lsoa_code"]).lsoa_code.astype(str).unique())
    todo = [l for l in all_lsoas if l not in done]
    print(f"{len(all_lsoas)} LSOAs total | {len(done)} done | {len(todo)} to run", flush=True)

    t0 = time.time()
    for k, lsoa in enumerate(todo):
        df = run_decomposition([lsoa], year=YEAR, config_path=CFG)   # shared decomposition
        df["lsoa_code"] = lsoa                                        # guarantee the cohort label
        df[COLS].to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        el = time.time() - t0
        print(f"[{k+1}/{len(todo)}] {lsoa} ({len(df)} dw) | {el:.0f}s elapsed, "
              f"~{el/(k+1)*len(todo)/60:.0f} min total", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
