# Model Calibration Recovery Log (2026-03-04)

## Summary
This document records the notebook/code recovery actions taken after calibration files appeared missing on `dev_levers`.

The loss was caused by branch history rewrite (rebase), not permanent deletion.

## Root Cause
- A snapshot commit existed on `dev_levers`: `9c1dafd` (`chore: snapshot dev_levers work including docs backup`).
- `dev_levers` was later rebased onto a different commit chain (`d585428 -> 4e151f7 -> ed2e2d1`).
- Files that only existed in `9c1dafd` disappeared from branch tip after the rewrite.
- Git reflog still referenced `9c1dafd`, so recovery was possible.

## Recovery Actions Performed
1. Verified missing assets via git history and working tree.
2. Created safety branch:
   - `backup/notebooks_9c1dafd` -> `9c1dafd`
3. Restored missing notebooks from backup commit.
4. Committed restored notebooks:
   - `d201db3` (`restore: recover calibration notebooks from backup snapshot`)
5. Push initially failed due to GitHub 100MB limit (`notebooks/scenario_testing_v2_1.ipynb`).
6. Cleared outputs in the oversized notebook and amended recovery commit.
7. Pushed branch successfully using explicit non-thin push:
   - `origin/dev_levers` updated to `d201db3`
8. Restored calibration engine code from backup snapshot for the coupled runtime files:
   - `household_energy/agent.py`
   - `household_energy/analyze.py`
   - `household_energy/climate.py`
   - `household_energy/config_defaults.yaml`
   - `household_energy/make_animation.py`
   - `household_energy/model.py`
   - `household_energy/run.py`
   - `household_energy/run_lsoa_batch.py`
   - `household_energy/server.py`
9. Committed restored engine files on `codex/model_calibration`:
   - `d727ad7` (`restore: recover calibration engine files from backup snapshot`)

## Branch State
- `dev_levers` and `origin/dev_levers` share recovered notebook history at `d201db3`.
- `codex/model_calibration` is based on the recovered state and adds:
  - `d727ad7` (calibration engine file restore)

## Notes
- Local uncommitted/untracked items not included in the recovery commit were intentionally left untouched:
  - `README.md` (modified)
  - `household_energy/config.py` (modified)
  - `household_energy/serl_calibration_pipeline.py` (untracked)
  - `notebooks/recovery/` (untracked)

## What Was Missing In `agent.py` And `model.py`
The recovery commit restored major calibration/runtime logic, not minor formatting changes.

### `household_energy/agent.py`
- Restored split baseline computation (`electric` + `gas`) instead of single scalar base load:
  - `_compute_hourly_base_components` and related helpers.
  - File refs: `agent.py:293`, `agent.py:642`, `agent.py:645`.
- Restored SAP-calibration parameterization and multipliers used by slope/cap/spike behavior:
  - `_sap_params`, `_sap_index`, `_sap_multiplier`.
  - File refs: `agent.py:464`, `agent.py:480`, `agent.py:486`.
- Restored fuel-bucket resolution and gas-share routing used by calibration scenarios:
  - `_resolve_heating_fuel_bucket`, `_resolve_base_gas_share`, `_resolve_gas_spike_share`, cache refresh.
  - File refs: `agent.py:539`, `agent.py:570`, `agent.py:582`, `agent.py:593`.
- Restored per-fuel accounting on each tick:
  - `electric_kwh`, `gas_kwh`, `other_kwh` reset + update paths.
  - File refs: `agent.py:517`, `agent.py:670`, `agent.py:735`.
- Restored annual accumulation container used for year-level calibration summaries:
  - `annual_kwh_by_year`.
  - File ref: `agent.py:186`.

### `household_energy/model.py`
- Restored intraday calibration controls and profiles:
  - `heating_profile_24h`, `dhw_profile_24h`, AM/PM/winter peak multipliers, occupancy-aware DHW.
  - File refs: `model.py:126`, `model.py:131`, `model.py:140`, `model.py:727`.
- Restored separate fuel baseline anchors and baseline split integration with household fuel routing:
  - `baseline_anchor_elec_kwh_per_hour`, `baseline_anchor_gas_kwh_per_hour`, `use_separate_fuel_baseline_anchors`.
  - File refs: `model.py:152`, `model.py:155`, `model.py:158`.
- Restored richer schedule calibration controls:
  - configurable `schedule_defs`, `schedules.type_map`, `wfh_share`, `jitter_hours`.
  - File refs: `model.py:175`, `model.py:179`, `model.py:213`, `model.py:488`.
- Restored SERL multiplier pipeline support:
  - load profiles CSV, apply hourly/monthly multipliers per seg3 group.
  - File refs: `model.py:240`, `model.py:663`, `model.py:698`.
- Restored segmentation aggregation for calibration outputs:
  - `calibration_segmentations`, `_collect_segmentation_aggregates`, `get_segmentation_timeseries`.
  - File refs: `model.py:257`, `model.py:791`, `model.py:838`.
- Restored aggregate fuel metrics on model outputs:
  - `total_electric_kwh`, `total_gas_kwh`, `total_other_kwh` and their reporters.
  - File refs: `model.py:228`, `model.py:576`, `model.py:985`.
- Restored annual kWh accumulation per household at model level:
  - `_accumulate_annual_kwh`.
  - File ref: `model.py:861`.

## Recommended Safeguards
- Before rebases on working branches, create a backup branch (`backup/<name>`).
- For large notebooks, clear outputs before commit or use Git LFS.
- Keep recovery checkpoints as small, focused commits (not mixed with unrelated edits).
