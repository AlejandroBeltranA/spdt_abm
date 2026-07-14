"""Per-component, per-fuel, per-heating-type decomposition of annual demand.

Built for the pipeline-transparency notebook (Goal 1 of HANDOFF_2026-06-19):
run the live model on a Newcastle sample and decompose each dwelling's annual
demand into baseline / heating / occupancy, split by heating fuel bucket
(gas-heated vs electric-heated) and by energy fuel (electric vs gas kWh), so no
error can hide inside a coverage-cancelled total.

The model resets per-agent ``base_kwh``/``heat_kwh``/``spike_kwh`` at the start
of every step (``_reset_base_loads`` -> ``reset_energy``), so the component split
is per-hour. We accumulate it across the year ourselves; the model already
accumulates per-agent ``electric_kwh``/``gas_kwh`` annual totals via
``_accumulate_annual_kwh``.

Usage (CLI smoke test):
    .venv/bin/python research/applied/scripts/decompose_demand.py E01008397
"""
from __future__ import annotations

import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd

from household_energy.model import EnergyModel
from household_energy.config import CALIBRATED_PATH

GEOJSON = "data/epc_abm_newcastle.geojson"
CLIMATE = "data/ncc_2t_timeseries_2010_2026.parquet"


def run_decomposition(
    lsoa_codes: list[str] | None = None,
    year: int = 2023,
    hours: int = 8760,
    config_path: str | None = None,
    geojson: str = GEOJSON,
    climate: str = CLIMATE,
    progress_every: int = 0,
) -> pd.DataFrame:
    """Run the live model and return a per-dwelling decomposition DataFrame.

    Columns: unique_id, lsoa_code, heating_bucket (gas/electric/other),
    base_kwh, heat_kwh, spike_kwh (annual, all fuels combined),
    electric_kwh, gas_kwh (annual fuel totals), total_kwh.
    """
    if config_path is None:
        config_path = str(CALIBRATED_PATH)

    gdf = gpd.read_file(geojson)
    gdf["lsoa_code"] = gdf["lsoa_code"].astype(str)
    if lsoa_codes is not None:
        gdf = gdf[gdf["lsoa_code"].isin([str(c) for c in lsoa_codes])].copy()
    if len(gdf) == 0:
        raise ValueError("No dwellings selected — check lsoa_codes.")

    # The agent does not carry lsoa_code, so map it back by the same id the model uses for
    # unique_id (UPRN, else uprn/fid). Without this, multi-LSOA / direct calls return blank
    # lsoa_code (see PIPELINE_AUDIT_2026-06-24). run_city_decomp's per-LSOA patch is then
    # redundant rather than load-bearing.
    _idcol = "UPRN" if "UPRN" in gdf.columns else ("uprn" if "uprn" in gdf.columns else "fid")
    id2lsoa = dict(zip(gdf[_idcol].astype(str), gdf["lsoa_code"].astype(str))) if _idcol in gdf.columns else {}

    m = EnergyModel(
        gdf=gdf,
        climate_parquet=climate,
        climate_start=pd.Timestamp(f"{year}-01-01", tz="UTC"),
        local_tz="Europe/London",
        collect_agent_level=False,
        agent_collect_every=1,
        config_path=config_path,
    )

    agents = m.household_agents
    n = len(agents)
    base = np.zeros(n)
    heat = np.zeros(n)
    spike = np.zeros(n)

    t0 = time.time()
    for step in range(hours):
        m.step()
        # Read this hour's component split before the next step resets it.
        for i, h in enumerate(agents):
            base[i] += h.base_kwh
            heat[i] += h.heat_kwh
            spike[i] += h.spike_kwh
        if progress_every and (step + 1) % progress_every == 0:
            el = time.time() - t0
            print(f"  step {step+1}/{hours}  ({el:.0f}s, {el/(step+1)*1000:.1f} ms/step)")

    rows = []
    for i, h in enumerate(agents):
        rows.append(
            {
                "unique_id": str(h.unique_id),
                "lsoa_code": str(getattr(h, "lsoa_code", "")),
                "heating_bucket": h._resolve_heating_fuel_bucket(),
                "main_fuel_type": getattr(h, "main_fuel_type", None),
                "main_heating_system": getattr(h, "main_heating_system", None),
                "base_kwh": base[i],
                "heat_kwh": heat[i],
                "spike_kwh": spike[i],
                "electric_kwh": float(h.annual_electric_kwh_by_year.get(year, 0.0)),
                "gas_kwh": float(h.annual_gas_kwh_by_year.get(year, 0.0)),
                "total_kwh": float(h.annual_kwh_by_year.get(year, 0.0)),
            }
        )
    df = pd.DataFrame(rows)
    # Fill lsoa_code from the input gdf (the agent attribute is blank); keep any non-empty
    # value the agent did carry.
    mapped = df["unique_id"].map(id2lsoa)
    df["lsoa_code"] = mapped.where(mapped.notna() & (mapped != ""), df["lsoa_code"])
    df["component_sum"] = df["base_kwh"] + df["heat_kwh"] + df["spike_kwh"]
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-dwelling means by heating bucket: components + fuel totals."""
    g = df.groupby("heating_bucket")
    out = g.agg(
        n=("unique_id", "size"),
        base=("base_kwh", "mean"),
        heat=("heat_kwh", "mean"),
        spike=("spike_kwh", "mean"),
        electric=("electric_kwh", "mean"),
        gas=("gas_kwh", "mean"),
        total=("total_kwh", "mean"),
    )
    return out


if __name__ == "__main__":
    codes = sys.argv[1:] or ["E01008397"]
    print(f"Running decomposition on {len(codes)} LSOA(s): {codes}")
    df = run_decomposition(lsoa_codes=codes, progress_every=2000)
    print(f"\n{len(df)} dwellings\n")
    print("Per-dwelling means by heating bucket (kWh/yr):")
    print(summarize(df).round(0).to_string())
    print("\nIdentity check (component_sum vs electric+gas):")
    diff = (df["component_sum"] - (df["electric_kwh"] + df["gas_kwh"])).abs()
    print(f"  max abs diff = {diff.max():.3f} kWh, mean = {diff.mean():.4f} kWh")
