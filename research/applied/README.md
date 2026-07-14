# research/applied — calibration · sensitivity · validation

The model's parameter pipeline. Calibrates household-energy demand from SERL
smart-meter data, screens parameter sensitivity, and validates against independent
DESNZ meter data. Paper 1 (*Applied Energy*) draws on this.

## Notebooks (`notebooks/`, run top-to-bottom)

| notebook | does | key outputs |
|---|---|---|
| `1_calibration.ipynb` | SERL fits → per-cohort baseline anchors + profile-paired heating slopes → assembles the config → (optional) promote | `results/calibration_v7_cohort/calibrated_config.yaml` |
| `2_sensitivity.ipynb` | Morris elementary-effects screen; cross-city robustness | influence ranking |
| `3_validation.ipynb` | per-cohort split vs SERL, seasonal shape vs SERL, per-unit vs DESNZ (held out) | validation tables/figures |

**The seam is the config:** notebook 1 *writes* it, notebooks 2 & 3 *read* it.
Notebooks detect the repo root and `chdir`, so they run from anywhere.

## Running
Open in Jupyter on the `.venv` kernel and **Run All**. By default the heavy steps
read cached artifacts (fast, deterministic). Flags to recompute live:
`RUN_CALIB` (re-run SERL fits), `RUN_COHORT_FIT` (re-measure cohort params, ~17 min),
`RUN_SA` (re-run Morris), `RUN_LIVE` (re-run the validation model). `PROMOTE=True`
in notebook 1 ships the config to `household_energy/calibrated_config.yaml`.

## Key scripts (`scripts/`, shared across the pipeline)
`calibrate_serl.py` (orchestrator), `fit_*.py` (setpoint, area, SAP, age, elec
baseline, presence spikes, monthly + diurnal profiles), `fit_cohort_params.py`
(per-cohort anchors + slopes), `run_monthly_retune.py` / `run_city_decomp.py`
(decomposition runs), `decompose_demand.py`, `transfer.py`, `utils.py`,
`build_nb_{calibration,sensitivity,validation}.py` (regenerate the notebooks).

## Current state (v7)
De-fudged calibration: per-cohort electricity baseline **anchors** (gas 0.300 /
electric 0.402, no correction multiplier), mean-1.0 **monthly profiles** (replacing
the hard heating-months gate), profile-paired heating slopes (gas 0.1846 / electric
0.0693). Reproduces SERL exactly per cohort; held-out DESNZ gas **0.94**,
electricity **1.12** (the SERL-panel-vs-population residual, deliberately not
closed). Config is **promoted** (2026-06-24) as the canonical shipped calibration in
`household_energy/calibrated_config.yaml`, so every entry point defaults to v7. The
`transfer_confidence_*.csv` are being regenerated to v7 (`transfer.py --config ...`,
or notebook 3 `RUN_TRANSFER`).
