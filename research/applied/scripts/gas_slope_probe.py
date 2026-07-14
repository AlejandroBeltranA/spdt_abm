#!/usr/bin/env python3
"""Iterate on the gas heating slope (and optional SAP-multiplier scale) by
running a single representative LSOA. Targets SERL national-mean per-home gas
consumption as the calibration anchor; Newcastle deviation from SERL emerges
from the stock composition + climate, not from a tuned target.

Defaults: median Newcastle LSOA E01008344 (gas dw=450, v4 produces 16,718
kWh/yr per gas dwelling vs SERL national 2023 mean ~10,577).

Usage:
  # Try a 22% slope cut (PROGRESS.md 2026-06-01 diagnostic):
  .venv/bin/python research/applied/scripts/gas_slope_probe.py --slope 0.164
  # Add a SAP renormalisation on top:
  .venv/bin/python research/applied/scripts/gas_slope_probe.py --slope 0.164 --sap-scale 1.075
  # Different LSOA:
  .venv/bin/python research/applied/scripts/gas_slope_probe.py --slope 0.164 --lsoa E01008353
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

_THIS = Path(__file__).resolve()
REPO = _THIS.parents[3]
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(REPO))

from household_energy.run_lsoa_batch import RunConfig, _run_single_lsoa  # noqa: E402
from utils import city_convention  # noqa: E402

V4_CONFIG = REPO / "results/calibration_v4_elecslope_20260604_094157/calibrated_config.yaml"
SERL_TARGET_2023 = 10_577  # kWh/yr, SERL national mean gas, gas-heated households
DESNZ_NEWCASTLE_PER_METER_2023 = 12_314  # kWh/yr per gas meter


def build_probe_config(slope_gas: float | None, sap_scale: float, base_config: Path) -> Path:
    cfg = yaml.safe_load(base_config.read_text())
    m = cfg["model"]
    old_slope = m.get("heating_slope_kWh_per_deg")
    if slope_gas is not None:
        m["heating_slope_kWh_per_deg"] = float(slope_gas)
    if sap_scale != 1.0:
        # v5 phase-1 configs don't carry sap_band_mult_heating_gas; guard.
        sap_map = m.get("sap_band_mult_heating_gas")
        if sap_map:
            m["sap_band_mult_heating_gas"] = {k: v * sap_scale for k, v in sap_map.items()}
    cfg.setdefault("meta", {})["probe"] = {
        "base_config": str(base_config),
        "old_gas_slope": old_slope,
        "new_gas_slope": slope_gas if slope_gas is not None else old_slope,
        "sap_scale": sap_scale,
    }
    out = Path(tempfile.mkstemp(prefix="probe_v5_", suffix=".yaml")[1])
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def run_probe(lsoa: str, slope_gas: float | None, sap_scale: float, city: str = "newcastle", base_config: Path = V4_CONFIG):
    conv = city_convention(city)
    geo = REPO / "data" / f"epc_abm_{conv.epc_slug}.geojson"
    climate = REPO / "data" / f"{conv.climate_prefix}_2t_timeseries_2010_2026.parquet"
    hidp_candidates = [
        REPO / "data" / f"{conv.epc_slug}_hidp_uprn_matches_tiered.csv",
        REPO / "data" / "hidp_uprn_matches_tiered.csv",
    ]
    hidp = next((c for c in hidp_candidates if c.exists()), None)

    probe_cfg = build_probe_config(slope_gas, sap_scale, base_config)
    outdir = REPO / "results_lsoa" / "_probe"
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = RunConfig(
        geojson=geo, climate=climate, hidp_csv=hidp,
        start_utc="2023-01-01T00:00:00Z",
        end_utc="2024-01-01T00:00:00Z",
        days=None, local_tz="Europe/London", lsoa_col="lsoa_code", outdir=outdir,
        agent_collect_every=1,
        stamp=f"probe_{lsoa}_slope{slope_gas:.4f}_sap{sap_scale:.3f}" if slope_gas is not None
              else f"probe_{lsoa}_cfg_sap{sap_scale:.3f}",
        config_path=probe_cfg.resolve(), save_model_timeseries=False,
    )
    row = _run_single_lsoa(lsoa, cfg, 1, 1)
    if row is None or row.empty:
        print("FAILED: no output")
        return

    r = row.iloc[0]
    gas = r["abm_gas_kwh"]
    gas_dw = r["run_gas_dwellings"]
    elec = r["abm_elec_kwh"]
    dw_all = r["run_dwellings"]
    per_gas = gas / gas_dw if gas_dw else float("nan")
    per_elec = elec / dw_all
    print()
    print(f"=== probe result | LSOA {lsoa} | slope={slope_gas} sap_scale={sap_scale} ===")
    print(f"  run_dwellings        : {int(dw_all):,}")
    print(f"  gas dwellings        : {int(gas_dw):,}")
    print(f"  ABM gas per gas-dw   : {per_gas:>8,.0f} kWh/yr")
    print(f"  ABM elec per dw      : {per_elec:>8,.0f} kWh/yr")
    print()
    print(f"  SERL national 2023   : {SERL_TARGET_2023:>8,} kWh/yr  (calibration anchor)")
    print(f"  DESNZ Newcastle/meter: {DESNZ_NEWCASTLE_PER_METER_2023:>8,} kWh/yr  (independent benchmark)")
    print(f"  v4 LSOA baseline     : ~16,700 kWh/yr  (current overshoot)")
    print()
    if per_gas:
        print(f"  ratio model / SERL   : {per_gas / SERL_TARGET_2023:.3f}  "
              f"(>1 means Newcastle stock skew above national, plausible up to ~1.2)")
        print(f"  ratio model / DESNZ  : {per_gas / DESNZ_NEWCASTLE_PER_METER_2023:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slope", type=float, default=None,
                    help="gas heating slope kWh/h/°C override (default: use the value in --config; v4=0.2097)")
    ap.add_argument("--sap-scale", type=float, default=1.0, help="multiply ALL sap_band_mult_heating_gas entries by this scalar")
    ap.add_argument("--lsoa", default="E01008344", help="LSOA to probe (default median Newcastle)")
    ap.add_argument("--city", default="newcastle")
    ap.add_argument("--config", type=Path, default=V4_CONFIG,
                    help=f"Base config to probe (default: {V4_CONFIG}). Point at the v5 phase-1 config to test new setpoint + linear heating.")
    a = ap.parse_args()
    run_probe(a.lsoa, a.slope, a.sap_scale, a.city, base_config=a.config)


if __name__ == "__main__":
    main()
