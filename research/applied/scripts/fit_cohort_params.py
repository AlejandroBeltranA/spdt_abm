"""Fit the per-cohort baseline anchors and the profile-paired heating slopes.

These three parameters can't be read off a SERL regression alone — they depend on
the dwelling stock and the climate's degree-hour accumulation — so they're fitted
with ONE short model measurement, then computed from SERL cohort targets:

  baseline_anchor_elec_kwh_per_hour_electric  electric-heated baseline anchor,
        fitted so electric-heated dwellings reproduce SERL electric-heated
        baseline electricity (all-electric water + cooking → higher than gas).
  heating_slope_kWh_per_deg            gas heating slope = SERL HDD slope ÷ the
        seasonal-profile annual inflation (keeps the SERL-calibrated annual when
        the heating-months gate is replaced by the smooth monthly profile).
  heating_slope_kWh_per_deg_electric   electric heating slope, fitted so
        electric-heated dwellings reproduce SERL electric-heated heating
        (absorbs the HDD-base vs setpoint-hinge convention gap; documented).

Measurement config: monthly profiles ON, the gas anchor + gas slope applied to
BOTH cohorts (so we can read what the model gives each cohort at a known
reference), then invert against the SERL targets.

  .venv/bin/python research/applied/scripts/fit_cohort_params.py [n_lsoa]
Writes results/calibration_serl_fits/cohort_params.yaml
"""
from __future__ import annotations
import sys, copy
from pathlib import Path
import geopandas as gpd, numpy as np, pandas as pd, yaml

from household_energy.model import EnergyModel
from run_decomp_sample import pick_sample, YEAR

REPO = Path(__file__).resolve().parents[3]
GEO = "data/epc_abm_newcastle.geojson"
CLIM = "data/ncc_2t_timeseries_2010_2026.parquet"
SHIPPED = "household_energy/calibrated_config.yaml"
PROFILES = "results/calibration_serl_fits/monthly_profile.yaml"
DIAG = "results/calibration_v5_phase5b/diagnostics.json"
OUT = "results/calibration_serl_fits/cohort_params.yaml"
DAYS = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
HEATING_MONTHS = [10,11,12,1,2,3,4]


def serl_targets():
    d = pd.read_csv("data/serl_8963_targets/daily_targets.csv")
    def mon(hf, qty):
        s = d[(d.period_type=="monthly")&(d.year==2023)&(d.weekday_weekend=="both")&
              (d.heating_fuel==hf)&(d.has_pv.isin(["No","All"]))&(d.seg3_var=="none")&(d.quantity==qty)]
        return pd.to_numeric(s["mean"],errors="coerce").groupby(s["month"]).mean().reindex(range(1,13))
    ele = mon("Electric","Electricity imports")
    floor = ele[[6,7,8]].mean()
    gas_elec = mon("Gas","Electricity imports")            # gas-heated electricity ≈ all baseline
    return {
        "gas_heated_baseline":  float(sum(gas_elec[m]*DAYS[m] for m in range(1,13))),
        "elec_heated_baseline": float(floor*365),
        "elec_heated_heating":  float(sum((ele-floor).clip(lower=0)[m]*DAYS[m] for m in range(1,13))),
    }


def profile_inflation(setpoint, profile12):
    """Analytic: how much the monthly heating profile changes the annual relative
    to the hard heating-months gate, on this climate. No simulation needed."""
    clim = pd.read_parquet(CLIM); clim["timestamp"] = pd.to_datetime(clim["timestamp"], utc=True)
    t = clim[(clim.timestamp>=f"{YEAR}-01-01")&(clim.timestamp<f"{YEAR+1}-01-01")].groupby("timestamp")["temp_C"].mean()
    dh = np.maximum(0.0, setpoint - t.values)
    months = t.index.month
    dh_by_m = np.array([dh[months==m].sum() for m in range(1,13)])
    profiled = float((np.array(profile12)*dh_by_m).sum())
    gated = float(dh_by_m[[m-1 for m in HEATING_MONTHS]].sum())
    return profiled/gated


def fit_cohort_params(n_lsoa: int = 10, *, write: bool = True) -> dict:
    """Fit the per-cohort anchors + profile-paired slopes and return them as a dict.

    Runs ONE short model measurement (``n_lsoa`` LSOAs, ~6 min at 30) with the
    seasonal profiles on and the raw SERL anchor/slope applied to both cohorts,
    reads what each cohort consumes, then inverts against the SERL cohort targets.
    The returned dict is exactly what is written to ``cohort_params.yaml`` and what
    notebook §1.5 consumes. Set ``write=False`` to compute without touching the
    committed artifact.

    Importable so the notebook and the CLI share one code path:
        from fit_cohort_params import fit_cohort_params
        cp = fit_cohort_params(30, write=False)
    """
    import json
    cfg = yaml.safe_load(open(SHIPPED))
    mp = yaml.safe_load(open(PROFILES))
    diag = json.load(open(DIAG))
    ref_anchor = float(diag["anchors"]["electric"])    # raw SERL summer baseline (no recentring)
    ref_slope = float(diag["hdd_slopes"]["gas"])        # raw SERL HDD gas slope

    # measurement config: profiles on; raw anchor + raw gas slope for BOTH cohorts,
    # so we read what the model gives each cohort at a known reference point.
    meas = copy.deepcopy(cfg); m = meas["model"]
    m["base_profile_12_electric"] = mp["base_profile_12_electric"]
    m["heating_month_profile_12"] = mp["heating_month_profile_12"]
    m["baseline_anchor_elec_kwh_per_hour"] = ref_anchor
    m["heating_slope_kWh_per_deg"] = ref_slope
    m.pop("heating_slope_kWh_per_deg_electric", None)
    m.pop("electric_heated_baseline_mult", None)
    m.pop("baseline_anchor_elec_kwh_per_hour_electric", None)
    meas_path = "results_lsoa/_tmp_config_cohort_measure.yaml"
    yaml.safe_dump(meas, open(meas_path,"w"), sort_keys=False)

    # run the measurement
    sample = pick_sample()[:n_lsoa]
    gdf = gpd.read_file(GEO); gdf["lsoa_code"] = gdf["lsoa_code"].astype(str)
    gdf = gdf[gdf.lsoa_code.isin(sample)].copy()
    mdl = EnergyModel(gdf=gdf, climate_parquet=CLIM, climate_start=pd.Timestamp(f"{YEAR}-01-01",tz="UTC"),
                      local_tz="Europe/London", collect_agent_level=False, agent_collect_every=1, config_path=meas_path)
    ag = mdl.household_agents
    is_e = np.array([a._resolve_heating_fuel_bucket()=="electric" for a in ag])
    is_g = np.array([a._resolve_heating_fuel_bucket()=="gas" for a in ag])
    heat = np.zeros(len(ag))
    for _ in range(8760):
        mdl.step()
        for i,a in enumerate(ag):
            heat[i]+=a.heat_kwh
    # Non-heating ELECTRICITY per cohort (the target is electricity, not total energy):
    # gas-heated heat is gas, so their electric total is all non-heating; electric-
    # heated heat is electric, so subtract it.
    elec = np.array([float(a.annual_electric_kwh_by_year.get(YEAR, 0.0)) for a in ag])
    nonheat = elec.copy(); nonheat[is_e] = elec[is_e] - heat[is_e]
    g_nonheat = float(nonheat[is_g].mean()); e_nonheat = float(nonheat[is_e].mean())
    e_heat = float(heat[is_e].mean())

    tgt = serl_targets()
    setpoint = float(cfg["model"].get("heating_trigger_temp_C", 16.47))
    infl = profile_inflation(setpoint, mp["heating_month_profile_12"])

    anchor_gas = round(ref_anchor * tgt["gas_heated_baseline"]  / g_nonheat, 4)
    anchor_e   = round(ref_anchor * tgt["elec_heated_baseline"] / e_nonheat, 4)
    slope_gas  = round(ref_slope / infl, 4)
    slope_e    = round(ref_slope * tgt["elec_heated_heating"] / e_heat, 4)

    out = {
        "baseline_anchor_elec_kwh_per_hour": anchor_gas,
        "baseline_anchor_elec_kwh_per_hour_electric": anchor_e,
        "heating_slope_kWh_per_deg": slope_gas,
        "heating_slope_kWh_per_deg_electric": slope_e,
        "diagnostics": {
            "n_lsoa": n_lsoa, "ref_anchor": ref_anchor, "ref_slope": ref_slope,
            "profile_annual_inflation": round(infl,4),
            "measured_gas_nonheat": round(g_nonheat), "measured_elec_nonheat": round(e_nonheat),
            "measured_elec_heat": round(e_heat),
            "serl_targets": {k: round(v) for k,v in tgt.items()},
            "note": ("electric slope fit so electric-heated reproduces SERL national heating on this "
                     "climate; climate-independent (per-city) derivation is a documented follow-up."),
        },
    }
    if write:
        Path(OUT).parent.mkdir(parents=True, exist_ok=True)
        yaml.safe_dump(out, open(OUT, "w"), sort_keys=False)
    return out


def main():
    n_lsoa = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    out = fit_cohort_params(n_lsoa, write=True)
    print(yaml.safe_dump({k: v for k, v in out.items() if k != "diagnostics"}, sort_keys=False))
    print("diagnostics:", out["diagnostics"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
