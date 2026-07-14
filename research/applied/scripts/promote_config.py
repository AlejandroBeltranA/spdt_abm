#!/usr/bin/env python3
"""
Promote a calibration output into the engine's shipped config.

The calibration pipeline (calibrate_serl.py) writes a full
`results/calibration_<label>/calibrated_config.yaml` that also carries a `meta`
provenance block and a deprecated `heating_setpoint_C` alias. The engine only
reads the `model:` block, so this script copies that block alone into
`household_energy/calibrated_config.yaml` with a provenance header — keeping the
shipped config clean and re-promotable without dead keys creeping back.

Usage:
  .venv/bin/python research/applied/scripts/promote_config.py            # v5 default
  .venv/bin/python research/applied/scripts/promote_config.py \
      --source results/calibration_<label>/calibrated_config.yaml --label <label>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = REPO / "results/calibration_v5_phase5b/calibrated_config.yaml"
DEST = REPO / "household_energy/calibrated_config.yaml"

HEADER = """\
# ============================================================================
# Calibrated configuration — the parameters that make the ABM accurate.
#
# This is the canonical, shipped calibration the engine runs with by default.
# It overlays household_energy/config_defaults.yaml (structural defaults); the
# values below are the fitted ones. Every entry point (run.py, run_lsoa_batch.py,
# research/applied/scripts/transfer.py) defaults to this file, so the model runs
# accurately on just the data, with no extra flags.
#
# PROVENANCE
#   Calibration label : {label}
#   Calibration year  : {year} (SERL national panel)
#   Produced by       : research/applied/scripts/calibrate_serl.py
#   Promoted by       : research/applied/scripts/promote_config.py
#   Source artifact   : {source}
#
# DO NOT hand-edit the fitted values here. Edit the calibration pipeline and
# re-promote (see research/applied/RUNBOOK.md §0.5). The `model:` block only is
# shipped; the calibration's `meta` provenance and the deprecated
# `heating_setpoint_C` alias are dropped on promotion.
# ============================================================================
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--label", type=str, default=None,
                    help="provenance label (default: inferred from source dir)")
    a = ap.parse_args()
    a.source = a.source.resolve()   # accept relative paths; relative_to(REPO) below needs absolute
    if not a.source.exists():
        raise FileNotFoundError(a.source)

    raw = yaml.safe_load(a.source.read_text()) or {}
    model = dict(raw.get("model") or {})
    if not model:
        raise ValueError(f"No `model:` block in {a.source}")
    model.pop("heating_setpoint_C", None)  # deprecated alias of heating_trigger_temp_C

    label = a.label or a.source.parent.name.replace("calibration_", "")
    years = (raw.get("meta") or {}).get("calibration_years")
    year = ", ".join(str(y) for y in years) if years else "unknown"

    try:
        src_str = str(a.source.relative_to(REPO))
    except ValueError:
        src_str = str(a.source)   # source outside the repo: record the absolute path
    header = HEADER.format(label=label, year=year, source=src_str)
    body = yaml.safe_dump({"model": model}, sort_keys=False, default_flow_style=False)
    DEST.write_text(header + body)
    print(f"promoted {a.source.relative_to(REPO)} -> {DEST.relative_to(REPO)} "
          f"(label={label}, {len(model)} model keys, meta + alias dropped)")


if __name__ == "__main__":
    main()
