#!/usr/bin/env python3
"""
Assemble the sensitivity-analysis parameter table from a bootstrapped v5
calibration directory.

Each row is one SA *knob*: a parameter (or coherent group of band multipliers)
that the Morris screen perturbs, with its central value, its uncertainty, and
the config key(s) the perturbation writes to. Three provenance classes:

  - SERL-fitted   : central + SE from the fit script's / diagnostics' bootstrap
                    block. Perturbed over ±k_sigma·SE (default k=2).
  - literature    : central + an explicit [low, high] from the cited source.
  - (neutralised / physical-cap channels are NOT knobs — zero range by
     construction, so they can't move the output and don't belong in the screen.)

A deliberate scoping call: ``boiler_efficiency`` and ``heatpump_cop_ref`` enter
only via ``hp_effect_mult`` for heat-pump-converted dwellings, so under the
*baseline* (no-adoption) transfer they have exactly zero effect on citywide
energy. They are policy-scenario knobs (Paper 2), not Paper-1 baseline-energy
knobs, and are excluded here to avoid a misleading zero-sensitivity row. The
night ``setpoint_setback_C`` *does* affect baseline heating and is included.

Outputs (under results/sensitivity_analysis/):
  sa_param_table.yaml  — machine-readable knob specs for sa_morris.py
  sa_param_table.csv   — human/paper-readable one-row-per-knob summary

Usage:
  python research/applied/scripts/build_sa_param_table.py \
      --calib-dir results/calibration_v5_phase5b_boot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

_THIS = Path(__file__).resolve()
REPO = _THIS.parents[3]

K_SIGMA = 2.0  # SERL-fitted knobs are screened over central ± K_SIGMA · SE


def _load(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build(calib_dir: Path) -> list[dict]:
    cfg = _load(calib_dir / "calibrated_config.yaml")["model"]
    diag = _load(calib_dir / "diagnostics.json")
    setpoint = _load(calib_dir / "heating_setpoint_fit.yaml")
    area = _load(calib_dir / "area_scaling_fit.yaml")
    sap = _load(calib_dir / "sap_band_mult_fit.yaml")
    age = _load(calib_dir / "age_mult_fit.yaml")
    eba = _load(calib_dir / "elec_baseline_area_fit.yaml")
    presence = _load(calib_dir / "presence_spikes_fit.yaml")

    for name, d in [("diagnostics.json", diag), ("heating_setpoint_fit", setpoint),
                    ("area_scaling_fit", area), ("sap_band_mult_fit", sap),
                    ("age_mult_fit", age), ("elec_baseline_area_fit", eba),
                    ("presence_spikes_fit", presence)]:
        if "bootstrap" not in d:
            raise RuntimeError(
                f"{name} has no 'bootstrap' block — re-run calibrate_serl.py "
                f"with --bootstrap N (calib dir: {calib_dir})."
            )

    db = diag["bootstrap"]
    knobs: list[dict] = []

    # ── SERL-fitted scalars ──────────────────────────────────────────────
    knobs.append({
        "name": "heating_setpoint_C", "kind": "scalar",
        # config sets heating_trigger_temp_C and the legacy alias to the same
        # value; the model reads heating_trigger_temp_C. Perturb both together.
        "config_keys": ["heating_trigger_temp_C", "heating_setpoint_C"],
        "central": float(setpoint["bootstrap"]["heating_setpoint_C"]["value"]),
        "se":      float(setpoint["bootstrap"]["heating_setpoint_C"]["se"]),
        "k_sigma": K_SIGMA,
        "source":  "SERL temperature-band hinge fit (parametric bootstrap)",
    })
    knobs.append({
        "name": "heating_slope_kWh_per_deg", "kind": "scalar",
        "config_keys": ["heating_slope_kWh_per_deg"],
        "central": float(db["heating_slope_kWh_per_deg"]["value"]),
        "se":      float(db["heating_slope_kWh_per_deg"]["se"]),
        "k_sigma": K_SIGMA,
        "source":  "SERL HDD OLS-through-origin (in-process bootstrap)",
    })
    knobs.append({
        "name": "baseline_anchor_gas_kwh_per_hour", "kind": "scalar",
        "config_keys": ["baseline_anchor_gas_kwh_per_hour"],
        "central": float(db["baseline_anchor_gas_kwh_per_hour"]["value"]),
        "se":      float(db["baseline_anchor_gas_kwh_per_hour"]["se"]),
        "k_sigma": K_SIGMA,
        "source":  "SERL summer gas baseline (in-process bootstrap)",
    })
    knobs.append({
        "name": "baseline_anchor_elec_kwh_per_hour", "kind": "scalar",
        "config_keys": ["baseline_anchor_elec_kwh_per_hour"],
        # central = the recentred config value; SE = the calibrated-anchor
        # bootstrap SE (the +ΔB recentre is a fixed offset on top, so its
        # sampling SE carries through to the recentred level).
        "central": float(cfg["baseline_anchor_elec_kwh_per_hour"]),
        "se":      float(db["baseline_anchor_elec_kwh_per_hour"]["se"]),
        "k_sigma": K_SIGMA,
        "source":  "SERL summer elec baseline, recentred (in-process bootstrap)",
    })
    knobs.append({
        "name": "energy_per_person_home", "kind": "scalar",
        "config_keys": ["energy_per_person_home"],
        "central": float(presence["bootstrap"]["energy_per_person_home"]["value"]),
        "se":      float(presence["bootstrap"]["energy_per_person_home"]["se"]),
        "k_sigma": K_SIGMA,
        "source":  "SERL diurnal per-person (cross-seg unconfounded, bootstrap)",
    })

    # ── SERL-fitted band-lookup knobs (one coherent z-shift per lookup) ──
    for name, cfg_key, fit_yaml, src in [
        ("heat_slope_area_bands", "heat_slope_area_bands", area,
         "SERL gas slope × floor-area band (bootstrap)"),
        ("sap_band_mult_heating_gas", "sap_band_mult_heating_gas", sap,
         "SERL gas slope × SAP band (bootstrap)"),
        ("building_age_mult_heating_gas", "building_age_mult_heating_gas", age,
         "SERL gas slope × building-age band (bootstrap)"),
        ("baseline_elec_area_bands", "baseline_elec_area_bands", eba,
         "SERL baseline elec × floor-area band (bootstrap)"),
    ]:
        bb = fit_yaml["bootstrap"]
        central = {b: float(v["value"]) for b, v in bb.items()}
        se = {b: float(v["se"]) for b, v in bb.items()}
        knobs.append({
            "name": name, "kind": "lookup", "config_key": cfg_key,
            "central": central, "se": se, "k_sigma": K_SIGMA,
            "source": src,
        })

    # ── Literature knob ──────────────────────────────────────────────────
    knobs.append({
        "name": "setpoint_setback_C", "kind": "scalar_literature",
        "config_keys": ["setpoint_setback_C"],
        "central": float(cfg.get("setpoint_setback_C", 2.0)),
        "low": 1.0, "high": 3.0,
        "source": "CIBSE Domestic Heating Design Guide — typical night setback 1–3 °C",
    })

    return knobs


def _csv_rows(knobs: list[dict]) -> pd.DataFrame:
    rows = []
    for k in knobs:
        if k["kind"] == "lookup":
            # summarise the lookup by its mean relative SE across bands
            cen = k["central"]
            se = k["se"]
            rel = [se[b] / cen[b] for b in cen if cen[b] != 0]
            mean_rel = sum(rel) / len(rel) if rel else 0.0
            rows.append({
                "knob": k["name"], "kind": k["kind"],
                "central": f"{len(cen)}-band lookup",
                "uncertainty": f"±{K_SIGMA:g}·SE (mean {mean_rel*100:.1f}% per band)",
                "source": k["source"],
            })
        elif k["kind"] == "scalar_literature":
            rows.append({
                "knob": k["name"], "kind": k["kind"],
                "central": f"{k['central']:g}",
                "uncertainty": f"[{k['low']:g}, {k['high']:g}] (literature)",
                "source": k["source"],
            })
        else:
            rel = k["se"] / k["central"] * 100 if k["central"] else 0.0
            rows.append({
                "knob": k["name"], "kind": k["kind"],
                "central": f"{k['central']:.5g}",
                "uncertainty": f"±{K_SIGMA:g}·SE = ±{K_SIGMA*k['se']:.4g} ({rel:.1f}%)",
                "source": k["source"],
            })
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calib-dir", type=Path,
                   default=REPO / "results/calibration_v5_phase5b_boot")
    p.add_argument("--out-dir", type=Path,
                   default=REPO / "results/sensitivity_analysis")
    args = p.parse_args(argv)

    if not (args.calib_dir / "calibrated_config.yaml").exists():
        print(f"ERROR: no calibrated_config.yaml in {args.calib_dir}", file=sys.stderr)
        return 2

    knobs = build(args.calib_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    table = {
        "meta": {
            "calib_dir": str(args.calib_dir),
            "k_sigma": K_SIGMA,
            "n_knobs": len(knobs),
            "note": (
                "SERL-fitted knobs screened over central ± k_sigma·SE from the "
                "parametric bootstrap of SERL published sampling error; the "
                "literature knob over its cited range. boiler_efficiency and "
                "heatpump_cop_ref excluded — baseline-energy-inert (policy-only)."
            ),
        },
        "knobs": knobs,
    }
    with open(args.out_dir / "sa_param_table.yaml", "w") as f:
        yaml.safe_dump(table, f, sort_keys=False)

    df = _csv_rows(knobs)
    df.to_csv(args.out_dir / "sa_param_table.csv", index=False)

    print(f"Wrote {len(knobs)} knobs to {args.out_dir}/sa_param_table.{{yaml,csv}}")
    print()
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
