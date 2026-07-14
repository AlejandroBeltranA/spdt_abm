"""Run the decomposed retune on the 30-LSOA sample and capture BOTH:
  - per-dwelling annual split (baseline/heating/occupancy by heating cohort)
  - monthly per-cohort fuel totals (model elec & gas by month, per dwelling)
so we can see how it lands monthly and annually against SERL.
"""
from __future__ import annotations
import sys, time
import geopandas as gpd, numpy as np, pandas as pd

from household_energy.model import EnergyModel
from run_decomp_sample import pick_sample, YEAR

GEO = "data/epc_abm_newcastle.geojson"
CLIM = "data/ncc_2t_timeseries_2010_2026.parquet"
BUCKETS = ["gas", "electric", "other"]


def run_monthly_retune(tag: str = "retune2", n_lsoa: int | None = None,
                       config: str | None = None, *, write: bool = True):
    """Run the decomposed retune and return (annual_df, monthly_df).

    Importable so the validation notebook and the CLI share one code path:
        from run_monthly_retune import run_monthly_retune
        annual, monthly = run_monthly_retune("v7", 30, config_path)

    With ``write=True`` the two CSVs are written to results_lsoa/ under ``tag``
    exactly as the CLI does; the notebook then reads them back.
    """
    cfg = config or f"results_lsoa/_tmp_config_{tag}.yaml"
    out_annual = f"results_lsoa/decomp_sample_newcastle_2023_{tag}.csv"
    out_monthly = f"results_lsoa/monthly_cohort_newcastle_2023_{tag}.csv"
    sample = pick_sample()
    if n_lsoa is not None:
        sample = sample[:n_lsoa]
        print(f"(subset to {len(sample)} LSOAs)")
    gdf = gpd.read_file(GEO); gdf["lsoa_code"] = gdf["lsoa_code"].astype(str)
    gdf = gdf[gdf["lsoa_code"].isin(sample)].copy()
    m = EnergyModel(gdf=gdf, climate_parquet=CLIM,
                    climate_start=pd.Timestamp(f"{YEAR}-01-01", tz="UTC"),
                    local_tz="Europe/London", collect_agent_level=False,
                    agent_collect_every=1, config_path=cfg)
    agents = m.household_agents
    n = len(agents)
    bidx = np.array([BUCKETS.index(a._resolve_heating_fuel_bucket()) for a in agents])
    base = np.zeros(n); heat = np.zeros(n); spike = np.zeros(n)
    # monthly per-cohort fuel totals: [month 1..12][bucket] for elec and gas
    elec_mc = np.zeros((13, 3)); gas_mc = np.zeros((13, 3))
    start = pd.Timestamp(f"{YEAR}-01-01", tz="UTC")

    t0 = time.time()
    for step in range(8760):
        m.step()
        mo = (start + pd.to_timedelta(m.current_hour - 1, unit="h")).month
        for i, a in enumerate(agents):
            base[i] += a.base_kwh; heat[i] += a.heat_kwh; spike[i] += a.spike_kwh
            elec_mc[mo, bidx[i]] += a.electric_kwh
            gas_mc[mo, bidx[i]] += a.gas_kwh
        if (step + 1) % 2000 == 0:
            print(f"  step {step+1}/8760 ({time.time()-t0:.0f}s)")

    # annual per-dwelling
    rows = []
    for i, a in enumerate(agents):
        rows.append(dict(unique_id=str(a.unique_id), heating_bucket=BUCKETS[bidx[i]],
                         base_kwh=base[i], heat_kwh=heat[i], spike_kwh=spike[i],
                         electric_kwh=float(a.annual_electric_kwh_by_year.get(YEAR, 0.0)),
                         gas_kwh=float(a.annual_gas_kwh_by_year.get(YEAR, 0.0))))
    annual_df = pd.DataFrame(rows)

    # monthly per-cohort, divided by dwelling count in each cohort -> per-dwelling kWh/month
    counts = {b: int((bidx == j).sum()) for j, b in enumerate(BUCKETS)}
    mrows = []
    for mo in range(1, 13):
        for j, b in enumerate(BUCKETS):
            mrows.append(dict(month=mo, cohort=b, n=counts[b],
                              elec_kwh_per_dw=elec_mc[mo, j]/counts[b],
                              gas_kwh_per_dw=gas_mc[mo, j]/counts[b]))
    monthly_df = pd.DataFrame(mrows)

    if write:
        annual_df.to_csv(out_annual, index=False)
        monthly_df.to_csv(out_monthly, index=False)
        print(f"\ncounts: {counts}")
        print(f"saved {out_annual} and {out_monthly} ({time.time()-t0:.0f}s)")
    return annual_df, monthly_df


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "retune2"
    n_lsoa = int(sys.argv[2]) if len(sys.argv) > 2 else None
    config = sys.argv[3] if len(sys.argv) > 3 else None
    run_monthly_retune(tag, n_lsoa, config, write=True)


if __name__ == "__main__":
    main()
